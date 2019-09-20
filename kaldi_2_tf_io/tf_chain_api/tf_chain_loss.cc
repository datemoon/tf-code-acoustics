
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <sys/time.h>

#include "tf-2-kaldi-api.h"

namespace tf = tensorflow;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("ChainLoss")
	.Input("inputs: float")
	.Input("indexs: int32")
	.Input("in_labels: int32")
	.Input("weights: float")
	.Input("statesinfo: int32")
	.Input("num_states: int32")
	//.Input("supervision_weights: float")
	//.Input("num_sequences: int32")
	//.Input("frames_per_sequence: int32")
	.Attr("label_dim: int = 0")
	.Attr("den_indexs: tensor = { dtype: DT_INT32 }")
	.Attr("den_in_labels: tensor = { dtype: DT_INT32 }")
	.Attr("den_weights: tensor = { dtype: DT_INT32 }")
	.Attr("den_statesinfo: tensor = { dtype: DT_INT32 }")
	.Attr("den_num_states: int = 0")
	.Attr("den_start_state: int = 0")
	.Attr("delete_laststatesuperfinal: bool = true")
	.Attr("l2_regularize: float = 0.0")
	.Attr("leaky_hmm_coefficient: float = 0.0")
	.Attr("xent_regularize: float = 0.0")
	.Output("objf: float")
	.Output("gradient: float")
	.SetShapeFn([](InferenceContext* c)
	{
		ShapeHandle inputs;         // nnet forward output
		ShapeHandle indexs;         // next inputs it's lattice info
		ShapeHandle in_labels;
		ShapeHandle weights;
		ShapeHandle statesinfo;
		ShapeHandle num_states;

		// check shape
		TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &inputs));
		TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &indexs));
		TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &in_labels));
		TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &weights));
		TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 3, &statesinfo));
		TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 1, &num_states));

		// Get batch size from inputs and sequence_length, and update inputs
		// with the merged batch_size since it is returned.
		DimensionHandle batch_size;
		TF_RETURN_IF_ERROR(
			c->Merge(c->Dim(inputs, 1), c->Dim(num_states, 0), &batch_size));

		TF_RETURN_IF_ERROR(c->ReplaceDim(inputs, 1, batch_size, &inputs));

		c->set_output(0, c->Vector(batch_size));
		c->set_output(1, inputs);

		return tf::Status::OK();
	});


namespace chain_loss
{
class ChainLossOp: public tf::OpKernel
{
public:
	explicit ChainLossOp(tf::OpKernelConstruction* ctx) : tf::OpKernel(ctx)
	{
		// den fst data
		OP_REQUIRES_OK(ctx, ctx->GetAttr("den_indexs", &_den_indexs));
		OP_REQUIRES(ctx, _den_indexs.dtype() == tf::DT_INT32,
				tf::errors::InvalidArgument("_den_indexs must be int32, got ",
					tf::DataTypeString(_den_indexs.dtype())));

		OP_REQUIRES_OK(ctx, ctx->GetAttr("den_in_labels", &_den_in_labels));
		OP_REQUIRES(ctx, _den_in_labels.dtype() == tf::DT_INT32,
				tf::errors::InvalidArgument("_den_in_labels must be int32, got ",
					tf::DataTypeString(_den_in_labels.dtype())));

		OP_REQUIRES_OK(ctx, ctx->GetAttr("den_weights", &_den_weights));
		OP_REQUIRES(ctx, _den_weights.dtype() == tf::DT_FLOAT,
				tf::errors::InvalidArgument("_den_weights must be float, got ",
					tf::DataTypeString(_den_weights.dtype())));

		OP_REQUIRES_OK(ctx, ctx->GetAttr("den_statesinfo", &_den_statesinfo));
		OP_REQUIRES(ctx, _den_statesinfo.dtype() == tf::DT_INT32,
				tf::errors::InvalidArgument("_den_statesinfo must be , got ",
					tf::DataTypeString(_den_statesinfo.dtype())));

		OP_REQUIRES_OK(ctx, ctx->GetAttr("den_num_states", &_den_num_states));
		OP_REQUIRES_OK(ctx, ctx->GetAttr("den_start_state", &_den_start_state));
		OP_REQUIRES_OK(ctx, ctx->GetAttr("label_dim", &_label_dim));

		OP_REQUIRES_OK(ctx, ctx->GetAttr("delete_laststatesuperfinal", &_delete_laststatesuperfinal));

		// loss config
		OP_REQUIRES_OK(ctx, ctx->GetAttr("l2_regularize", &_l2_regularize));
		OP_REQUIRES_OK(ctx, ctx->GetAttr("leaky_hmm_coefficient", &_leaky_hmm_coefficient));
		OP_REQUIRES_OK(ctx, ctx->GetAttr("xent_regularize", &_xent_regularize));

		auto den_indexs_t = _den_indexs.matrix<int>();
		auto den_in_labels_t = _den_in_labels.vec<int>();
		auto den_weights_t = _den_weights.vec<float>();
		auto den_statesinfo_t = _den_statesinfo.matrix<int>();

		_den_graph_saver.Init(den_indexs_t.data(), den_in_labels_t.data(), 
				den_in_labels_t.data(), den_weights_t.data(), den_statesinfo_t.data(),
				_den_num_states, _label_dim, 
				_delete_laststatesuperfinal, _den_start_state);

	}

	void Compute(tf::OpKernelContext* ctx) override
	{
#ifdef DEBUG_SPEED
		struct timeval start;
		struct timeval end;
		gettimeofday(&start, NULL);
#endif

		const tf::Tensor* inputs;     // tensor<float, 3>
		const tf::Tensor* indexs;     // tensor<int, 3>
		const tf::Tensor* in_labels;  // matrix<int>
		const tf::Tensor* weights;    // matrix<float>
		const tf::Tensor* statesinfo; // tensor<int, 3>
		const tf::Tensor* num_states; // vector<int>
		//const tf::Tensor* supervision_weights; // float
		//const tf::Tensor* num_sequences; // int
		//const tf::Tensor* frames_per_sequence; // int

		OP_REQUIRES_OK(ctx, ctx->input("inputs", &inputs));
		OP_REQUIRES_OK(ctx, ctx->input("indexs", &indexs));
		OP_REQUIRES_OK(ctx, ctx->input("in_labels", &in_labels));
		OP_REQUIRES_OK(ctx, ctx->input("weights", &weights));
		OP_REQUIRES_OK(ctx, ctx->input("statesinfo", &statesinfo));
		OP_REQUIRES_OK(ctx, ctx->input("num_states", &num_states));
		
		OP_REQUIRES(ctx, inputs->shape().dims() == 3,
				tf::errors::InvalidArgument("inputs is not a 3-Tensor"));

		OP_REQUIRES(ctx, indexs->shape().dims() == 3,
				tf::errors::InvalidArgument("indexs is not a 3-Tensor"));

		OP_REQUIRES(ctx, tf::TensorShapeUtils::IsMatrix(in_labels->shape()),
				tf::errors::InvalidArgument("in_labels is not a matrix"));

		OP_REQUIRES(ctx, tf::TensorShapeUtils::IsMatrix(weights->shape()),
				tf::errors::InvalidArgument("weights is not a matrix"));

		OP_REQUIRES(ctx, statesinfo->shape().dims() == 3,
				tf::errors::InvalidArgument("statesinfo is not 3-Tensor"));

		OP_REQUIRES(ctx, tf::TensorShapeUtils::IsVector(num_states->shape()),
				tf::errors::InvalidArgument("num_states should be vector "
					"but received shapes: ",num_states->shape().DebugString()));

		const tf::TensorShape& inputs_shape = inputs->shape();
		const tf::int64 max_time = inputs_shape.dim_size(0);
		const tf::int64 batch_size = inputs_shape.dim_size(1);
		const tf::int64 num_classes_raw = inputs_shape.dim_size(2);
		
		const tf::TensorShape& indexs_shape = indexs->shape();
		const tf::int32 max_num_arcs = indexs_shape.dim_size(1);

		const tf::TensorShape& statesinfo_shape = statesinfo->shape();
		const tf::int32 max_num_states = statesinfo_shape.dim_size(1);

		// check num_classes_raw less then std::numeric_limits<int>::max()
		OP_REQUIRES(
				ctx, tf::FastBoundsCheck(num_classes_raw, std::numeric_limits<int>::max()),
				tf::errors::InvalidArgument("num_classes cannot exceed max int"));
		const int num_classes = static_cast<const int>(num_classes_raw);

		OP_REQUIRES(ctx, num_classes == _label_dim,
				tf::errors::InvalidArgument("input feature dim and label dim should be equal"));
		// malloc loss space
		//Tensor* loss = nullptr;
		//OP_REQUIRES_OK(ctx, ctx->allocate_output("loss", sequence_length->shape(), &loss));
		//auto loss_t = loss->vec<float>();

		tf::Tensor* objf = nullptr;
		OP_REQUIRES_OK(ctx, 
				ctx->allocate_output("objf", tf::TensorShape({3}), &objf));
		// malloc gradient space
		tf::Tensor* gradient;
		OP_REQUIRES_OK(ctx,
				ctx->allocate_output("gradient", inputs_shape, &gradient));


		auto inputs_t = inputs->tensor<float, 3>();
		auto objf_t = objf->vec<float>();
		auto gradient_t = gradient->tensor<float, 3>();

		// gradient set zero
		// the setZero is so slow,so I decide zet zero in hubo::MMILoss
		// gradient_t.setZero();

		auto indexs_t = indexs->tensor<int, 3>();
		auto in_labels_t = in_labels->matrix<int>();
		auto weights_t = weights->matrix<float>();
		auto statesinfo_t = statesinfo->tensor<int, 3>();
		auto num_states_t = num_states->vec<int>();

		// every sequences must be equal length.
		float supervision_weights = 1.0;
		int supervision_num_sequences = 1.0;
		int supervision_frames_per_sequence = max_time;

		bool ret = ChainLossDen(indexs_t.data(), in_labels_t.data(), in_labels_t.data(), 
				weights_t.data(), statesinfo_t.data(), num_states_t.data(),
				max_num_arcs, max_num_states,
				supervision_weights, supervision_num_sequences, supervision_frames_per_sequence, 
				_label_dim,
				inputs_t.data(),
				max_time, batch_size, num_classes_raw,
				_den_graph_saver,
				gradient_t.data(),
				objf_t.data(),
				_l2_regularize, _leaky_hmm_coefficient, _xent_regularize);

#ifdef DEBUG_SPEED
		gettimeofday(&end, NULL);
		std::cout << "DEBUG_SPEED : " << __FILE__ " : mmi_loss_op process data time:"
			<< (end.tv_sec - start.tv_sec)+(end.tv_usec-start.tv_usec)*1.0/1e6<< std::endl;
#endif

	}
private:
	tf::Tensor _den_indexs;
	tf::Tensor _den_in_labels;
	tf::Tensor _den_weights;
	tf::Tensor _den_statesinfo;
	int	_den_num_states;
	int _den_start_state;
	int _label_dim;
	bool _delete_laststatesuperfinal;
	// l2 regularization constant on the 'chain' output; the actual term added to
	// the objf will be -0.5 times this constant times the squared l2 norm.
	// (squared so it's additive across the dimensions).  e.g. try 0.0005.
	float _l2_regularize;
	
	// Coefficient for 'leaky hmm'.  This means we have an epsilon-transition from
	// each state to a special state with probability one, and then another
	// epsilon-transition from that special state to each state, with probability
	// leaky_hmm_coefficient times [initial-prob of destination state].  Imagine
	// we make two copies of each state prior to doing this, version A and version
	// B, with transition from A to B, so we don't have to consider epsilon loops-
	// or just imagine the coefficient is small enough that we can ignore the
	// epsilon loops.
	float _leaky_hmm_coefficient;
	
	// Cross-entropy regularization constant.  (e.g. try 0.1).  If nonzero,
	// the network is expected to have an output named 'output-xent', which
	// should have a softmax as its final nonlinearity.
	float _xent_regularize;

	hubo::DenominatorGraphSaver _den_graph_saver;

	TF_DISALLOW_COPY_AND_ASSIGN(ChainLossOp);
};

REGISTER_KERNEL_BUILDER(Name("ChainLoss")
		.Device(::tf::DEVICE_CPU),
		ChainLossOp);

REGISTER_KERNEL_BUILDER(Name("ChainLoss")
		.Device(::tf::DEVICE_GPU)
		.HostMemory("indexs")
		.HostMemory("in_labels")
		.HostMemory("weights")
		.HostMemory("statesinfo")
		.HostMemory("num_states")
		.HostMemory("objf"),
		ChainLossOp);

} // namespace



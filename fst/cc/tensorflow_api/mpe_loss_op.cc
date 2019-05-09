
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "loss.h"

#include <sys/time.h>

using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;

namespace tf = tensorflow;
/* (input)
 * inputs         : here it's nnet forward and no soft max and logit. nnet_outmust be 3 dimensional tensor,
 *                  which has dimension (t, n, p), where t is the time index, n is the minibatch index,
 *                  and p is class numbers. it map (rows, batch_size, cols)
 * sequence_length: The number of time steps for each sequence in the batch. which has dimension (n)
 * labels         : here it's acoustic align, max(labels) < p. which has dimension (n, t)
 * 
 * indexs(lattice): fst cur_state and next_state. indexs must be 3 dimensional tensor,
 *                  which has dimension (n, arcs_nums, 2), where n is the minibatch index,
 *                  states_nums is lattice state number, 2 is lattice dim save [cur_state, next_state]
 * pdf_values     : pdf_values is 2 dimensional tensor, which has dimension (n, arcs_nums),
 *                  practical application pdf = pdf_values[n] - 1
 * lm_ws          : lm_ws is 2 dimensional tensor, which has dimension (n, arcs_nums)
 * am_ws          : am_ws is 2 dimensional tensor, which has dimension (n, arcs_nums)
 * statesinfo     : statesinfo is 3 dimensional tensor, which has dimension (n, states_num, 2)
 * num_states     : num_states is states number, which has dimension (n)
 *
 * (attr)
 * silence_phones         : Colon-separated list of integer id's of silence phones, e.g. [1, 2, 3, ...]
 * pdf_to_phone           : pdf_id map phone. [pdf, phone]
 * one_silence_class      : If true, the newer behavior reduces insertions.
 * criterion              : Use state-level accuracies or phone accuracies.
 * old_acoustic_scale     : Add in the scores in the input lattices with this scale, rather than discarding them.
 * acoustic_scale         : Scaling factor for acoustic likelihoods
 *
 * (output)
 * loss                   : it accuracy frame rate . which has dimension (n)
 * gradient               : it shape same as nnet_out
 *
 * */

REGISTER_OP("MPELoss")
	.Input("inputs: float")          
	.Input("sequence_length: int32")
	.Input("labels: int32")
	.Input("indexs: int32")
	.Input("pdf_values: int32")
	.Input("lm_ws: float")
	.Input("am_ws: float")
	.Input("statesinfo: int32")
	.Input("num_states: int32")
	.Attr("silence_phones: list(int) = []")
	.Attr("pdf_to_phone: tensor = { dtype: DT_INT32 }")
	.Attr("one_silence_class: bool = true")
	.Attr("criterion: string = 'smbr'")
	.Attr("old_acoustic_scale: float = 0.0")
	.Attr("acoustic_scale: float = 1.0")
	.Output("loss: float")
	.Output("gradient: float")
	.SetShapeFn([](InferenceContext* c)
	{
		ShapeHandle inputs;         // nnet forward output
		ShapeHandle sequence_length;// every sequence length,it's vector
		ShapeHandle labels;         // align info
   		ShapeHandle indexs;         // next inputs it's lattice info
		ShapeHandle pdf_values;
		ShapeHandle lm_ws;
		ShapeHandle am_ws;
		ShapeHandle statesinfo;
		ShapeHandle num_states;

		// check shape
		TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &inputs));
		TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &sequence_length));
		TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &labels));
		TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 3, &indexs));
		TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 2, &pdf_values));
		TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 2, &lm_ws));
		TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 2, &am_ws));
		TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 3, &statesinfo));
		TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 1, &num_states));

		// Get batch size from inputs and sequence_length, and update inputs
		// with the merged batch_size since it is returned.
		DimensionHandle batch_size;
		TF_RETURN_IF_ERROR(
				c->Merge(c->Dim(inputs, 1), c->Dim(sequence_length, 0), &batch_size));
		TF_RETURN_IF_ERROR(c->ReplaceDim(inputs, 1, batch_size, &inputs));



		c->set_output(0, c->Vector(batch_size));
		c->set_output(1, inputs);

		return tf::Status::OK();
	});
/*
	.Doc(R"doc(
	Calculates the MMI Loss (log probability) for each batch entry.  Also calculates
	the gradient. 
   	inputs:  3-D, shape: (max_time x batch_size x num_classes), the logits.
	sequence_length:  A vector containing sequence lengths (batch).
   	indexs:  The indexs of a Tensor<int32, 3>.
	  indexs(i, :) == [b, instate, tostate] means lattice arc instate and tostate,
	pdf_values:
	lm_ws:
	am_ws:
	statesinfo:
	)doc");
*/

//using namespace tensorflow;

namespace distinguish_loss
{
class MPELossOp: public tf::OpKernel
{
public:
	explicit MPELossOp(tf::OpKernelConstruction* ctx) : tf::OpKernel(ctx) 
	{
		OP_REQUIRES_OK(ctx, ctx->GetAttr("silence_phones", &_silence_phones));
		OP_REQUIRES_OK(ctx, ctx->GetAttr("pdf_to_phone", &_pdf_to_phone));
//		const tf::TensorProto* proto;
//		OP_REQUIRES_OK(ctx, ctx->GetAttr("pdf_to_phone", &proto));
//		OP_REQUIRES_OK(ctx, ctx->device()->MakeTensorFromProto(
//					*proto, tf::AllocatorAttributes(), &_pdf_to_phone));
		OP_REQUIRES(ctx, _pdf_to_phone.dtype() == tf::DT_INT32,
				tf::errors::InvalidArgument("_pdf_to_phone must be int32, got ",
					DataTypeString(_pdf_to_phone.dtype())));

		OP_REQUIRES_OK(ctx, ctx->GetAttr("one_silence_class", &_one_silence_class));
		OP_REQUIRES_OK(ctx, ctx->GetAttr("criterion", &_criterion));
		OP_REQUIRES_OK(ctx, ctx->GetAttr("old_acoustic_scale", &_old_acoustic_scale));
		OP_REQUIRES_OK(ctx, ctx->GetAttr("acoustic_scale", &_acoustic_scale));
	}

	void Compute(tf::OpKernelContext* ctx) override 
	{
#ifdef DEBUG_SPEED
		struct timeval start;
		struct timeval end;
		gettimeofday(&start, NULL);
#endif
		const tf::Tensor* inputs;
		const tf::Tensor* sequence_length; // frames
		const tf::Tensor* labels;
		const tf::Tensor* indexs;
		const tf::Tensor* pdf_values;
		const tf::Tensor* lm_ws;
		const tf::Tensor* am_ws;
		const tf::Tensor* statesinfo;
		const tf::Tensor* num_states;
		OP_REQUIRES_OK(ctx, ctx->input("inputs", &inputs));
		OP_REQUIRES_OK(ctx, ctx->input("sequence_length", &sequence_length));
		OP_REQUIRES_OK(ctx, ctx->input("labels", &labels));
		OP_REQUIRES_OK(ctx, ctx->input("indexs", &indexs));
		OP_REQUIRES_OK(ctx, ctx->input("pdf_values", &pdf_values));
		OP_REQUIRES_OK(ctx, ctx->input("lm_ws", &lm_ws));
		OP_REQUIRES_OK(ctx, ctx->input("am_ws", &am_ws));
		OP_REQUIRES_OK(ctx, ctx->input("statesinfo", &statesinfo));
		OP_REQUIRES_OK(ctx, ctx->input("num_states", &num_states));

		OP_REQUIRES(ctx, inputs->shape().dims() == 3,
				tf::errors::InvalidArgument("inputs is not a 3-Tensor"));

		OP_REQUIRES(ctx, tf::TensorShapeUtils::IsVector(sequence_length->shape()),
				tf::errors::InvalidArgument("sequence_length is not a vector"));

		OP_REQUIRES(ctx, indexs->shape().dims() == 3,
				tf::errors::InvalidArgument("indexs is not a 3-Tensor"));
	
		OP_REQUIRES(ctx, tf::TensorShapeUtils::IsMatrix(pdf_values->shape()),
				tf::errors::InvalidArgument("pdf_values is not a matrix"));

		OP_REQUIRES(ctx, tf::TensorShapeUtils::IsMatrix(lm_ws->shape()),
				tf::errors::InvalidArgument("lm_ws is not a matrix"));

		OP_REQUIRES(ctx, tf::TensorShapeUtils::IsMatrix(am_ws->shape()),
				tf::errors::InvalidArgument("am_ws is not a matrix"));
		
		OP_REQUIRES(ctx, statesinfo->shape().dims() == 3,
				tf::errors::InvalidArgument("statesinfo is not 3-Tensor"));

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

		// check sequence length equal batch size
		OP_REQUIRES(
				ctx, batch_size == sequence_length->dim_size(0),
				tf::errors::InvalidArgument("len(sequence_length) != batch_size.  "
					"len(sequence_length):  ", sequence_length->dim_size(0),
					" batch_size: ", batch_size));
		auto seq_len_t = sequence_length->vec<tf::int32>();

		// malloc loss space
		tf::Tensor* loss = nullptr;
		OP_REQUIRES_OK(ctx, ctx->allocate_output("loss", sequence_length->shape(), &loss));
		auto loss_t = loss->vec<float>();

		// malloc gradient space
		tf::Tensor* gradient;
		OP_REQUIRES_OK(ctx,
				ctx->allocate_output("gradient", inputs_shape, &gradient));
		
		auto inputs_t = inputs->tensor<float, 3>();
		auto gradient_t = gradient->tensor<float, 3>();
		// gradient set zero
		// the setZero is so slow,so I decide zet zero in hubo::MPELoss
		//gradient_t.setZero();

		auto indexs_t = indexs->tensor<int, 3>();
		auto pdf_values_t = pdf_values->matrix<int>();
		auto lm_ws_t = lm_ws->matrix<float>();
		auto am_ws_t = am_ws->matrix<float>();
		auto statesinfo_t = statesinfo->tensor<int, 3>();
		auto num_states_t = num_states->vec<int>();
		auto labels_t = labels->matrix<int>();

		auto sequence_length_t = sequence_length->vec<int>();

		auto pdf_to_phone_t = _pdf_to_phone.matrix<int>();
		const tf::TensorShape& pdf_to_phone_shape = _pdf_to_phone.shape();
		const tf::int32 pdf_id_num = pdf_to_phone_shape.dim_size(0);
		
		OP_REQUIRES(ctx, pdf_id_num == num_classes_raw,
				tf::errors::InvalidArgument("pdf id == num_classes_raw"));
#ifdef DEBUG_SPEED
		gettimeofday(&end, NULL);
		std::cout << "DEBUG_SPEED : " << __FILE__ " : mpe_loss_op process data time:"
			<< (end.tv_sec - start.tv_sec)+(end.tv_usec-start.tv_usec)*1.0/1e6<< std::endl;
#endif

		bool ret_mpe = hubo::MPELoss(indexs_t.data(), pdf_values_t.data(),
				(float *)lm_ws_t.data(), (float *)am_ws_t.data(),
				statesinfo_t.data(), num_states_t.data(),
				max_num_arcs, max_num_states,
				inputs_t.data(),
				max_time, batch_size, num_classes_raw,
				labels_t.data(),
				sequence_length_t.data(),
				_silence_phones.data(),
				_silence_phones.size(),
				pdf_to_phone_t.data(),
				pdf_id_num,
				_old_acoustic_scale,
				_acoustic_scale, gradient_t.data(),
			   	loss_t.data(),
				_one_silence_class,
			   	_criterion);

#ifdef DEBUG_SPEED
		gettimeofday(&end, NULL);
		std::cout << "DEBUG_SPEED : " << __FILE__ << " : mpe_loss_op calculate mpe time:"
			<< (end.tv_sec - start.tv_sec)+(end.tv_usec-start.tv_usec)*1.0/1e6<< std::endl;
#endif
		//return ret_mmi;
	}
private:
	std::vector<tf::int32> _silence_phones;
	tf::Tensor _pdf_to_phone;
	bool _one_silence_class;
	std::string _criterion;
	float _old_acoustic_scale;
	float _acoustic_scale;
	TF_DISALLOW_COPY_AND_ASSIGN(MPELossOp);
};

REGISTER_KERNEL_BUILDER(Name("MPELoss").Device(::tensorflow::DEVICE_CPU), MPELossOp);

} // namespace


#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "loss.h"

using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;

namespace tf = tensorflow;

REGISTER_OP("MMILoss")
	.Input("inputs: float")
	.Input("sequence_length: int32")
	.Input("labels: int32")
	.Input("indexs: int32")
	.Input("pdf_values: int32")
	.Input("lm_ws: float")
	.Input("am_ws: float")
	.Input("statesinfo: int32")
	.Input("num_states: int32")
	.Attr("old_acoustic_scale: float = 0.0")
	.Attr("acoustic_scale: float = 1.0")
	.Attr("drop_frames: bool = true")
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
class MMILossOp: public tf::OpKernel
{
public:
	explicit MMILossOp(tf::OpKernelConstruction* ctx) : tf::OpKernel(ctx) 
	{
		OP_REQUIRES_OK(ctx, ctx->GetAttr("old_acoustic_scale", &_old_acoustic_scale));
		OP_REQUIRES_OK(ctx, ctx->GetAttr("acoustic_scale", &_acoustic_scale));
		OP_REQUIRES_OK(ctx, ctx->GetAttr("drop_frames", &_drop_frames));
	}

	void Compute(tf::OpKernelContext* ctx) override 
	{
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
		gradient_t.setZero();

		auto indexs_t = indexs->tensor<int, 3>();
		auto pdf_values_t = pdf_values->matrix<int>();
		auto lm_ws_t = lm_ws->matrix<float>();
		auto am_ws_t = lm_ws->matrix<float>();
		auto statesinfo_t = statesinfo->tensor<int, 3>();
		auto num_states_t = num_states->vec<int>();
		auto labels_t = labels->matrix<int>();

		auto sequence_length_t = sequence_length->vec<int>();


		bool ret_mmi = hubo::MMILoss(indexs_t.data(), pdf_values_t.data(),
				(float *)lm_ws_t.data(), (float *)am_ws_t.data(),
				statesinfo_t.data(), num_states_t.data(),
				max_num_arcs, max_num_states,
				inputs_t.data(),
				max_time, batch_size, num_classes_raw,
				labels_t.data(),
				sequence_length_t.data(),
				_old_acoustic_scale,
				_acoustic_scale, gradient_t.data(), loss_t.data(),
				_drop_frames);

		//return ret_mmi;
	}
private:
	float _old_acoustic_scale;
	float _acoustic_scale;
	bool _drop_frames;
	TF_DISALLOW_COPY_AND_ASSIGN(MMILossOp);
};

REGISTER_KERNEL_BUILDER(Name("MMILoss").Device(::tensorflow::DEVICE_CPU), MMILossOp);

} // namespace

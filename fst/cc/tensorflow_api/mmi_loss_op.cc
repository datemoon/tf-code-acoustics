
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/framework/allocator.h"


namespace tf = tensorflow;
//using namespace tensorflow;

namespace distinguish_loss
{
class MMILossOp: public tf::OpKernel
{
public:
	explicit MMILossOp(tf::OpKernelConstruction* ctx) : tf::OpKernel(ctx) 
	{
		OP_REQUIRES_OK(ctx, ctx->GetAttr("blank_label", &blank_label_));
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

		OP_REQUIRES(ctx, tf::inputs->shape().dims() == 3,
				tf::errors::InvalidArgument("inputs is not a 3-Tensor"));

		OP_REQUIRES(ctx, tf::TensorShapeUtils::IsVector(sequence_length->shape()),
				tf::errors::InvalidArgument("sequence_length is not a vector"));

		OP_REQUIRES(ctx, tf::TensorShapeUtils::IsMatrix(indexs->shape()),
				tf::errors::InvalidArgument("indexs is not a matrix"));
	
		OP_REQUIRES(ctx, tf::TensorShapeUtils::IsMatrix(pdf_values->shape()),
				tf::errors::InvalidArgument("pdf_values is not a matrix"));

		OP_REQUIRES(ctx, tf::TensorShapeUtils::IsMatrix(lm_ws->shape()),
				tf::errors::InvalidArgument("lm_ws is not a matrix"));

		OP_REQUIRES(ctx, tf::TensorShapeUtils::IsMatrix(am_ws->shape()),
				tf::errors::InvalidArgument("am_ws is not a matrix"));
		
		OP_REQUIRES(ctx, tf::TensorShapeUtils::IsMatrix(statesinfo->shape()),
				tf::errors::InvalidArgument("statesinfo is not a matrix"));

		const tf::TensorShape& inputs_shape = inputs->shape();
		const tf::int64 max_time = inputs_shape.dim_size(0);
		const tf::int64 batch_size = inputs_shape.dim_size(1);
		const tf::int64 num_classes_raw = inputs_shape.dim_size(2);

		// check num_classes_raw less then std::numeric_limits<int>::max()
		OP_REQUIRES(
				ctx, FastBoundsCheck(num_classes_raw, std::numeric_limits<int>::max()),
				tf::errors::InvalidArgument("num_classes cannot exceed max int"));
		const int num_classes = static_cast<const int>(num_classes_raw);

		// check sequence length equal batch size
		OP_REQUIRES(
				ctx, batch_size == seq_len->dim_size(0),
				errors::InvalidArgument("len(sequence_length) != batch_size.  "
					"len(sequence_length):  ", sequence_length->dim_size(0),
					" batch_size: ", batch_size));
		auto seq_len_t = seq_len->vec<int32>();

		// malloc loss space
		tf::Tensor* loss = nullptr;
		OP_REQUIRES_OK(ctx, ctx->allocate_output("loss", sequence_length->shape(), &loss));
		auto loss_t = loss->vec<float>();

		// malloc gradient space
		tf::Tensor* gradient;
		OP_REQUIRES_OK(ctx,
				ctx->allocate_output("gradient", inputs_shape, &gradient));
		auto gradient_t = gradient->tensor<float, 3>();
		auto inputs_t = inputs->tensor<float, 3>();

		// gradient set zero
		gradient_t.setZero();



	}
private:
	TF_DISALLOW_COPY_AND_ASSIGN(MMILossOp);
};

}

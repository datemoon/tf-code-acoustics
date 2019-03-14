
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("MMI_loss")
	.Input("inputs: float")
	.Input("sequence_length: int32")
	.Input("labels: int32")
	.Input("indexs: int32")
	.Input("pdf_values: int32")
	.Input("lm_ws: float")
	.Input("am_ws: float")
	.Input("statesinfo: float")
	.Input("num_states: int32")
	.Output("loss: float")
	.Output("gradient: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c)
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

		return Status::OK();
	})
	.Doc(R"doc(
	Calculates the MMI Loss (log probability) for each batch entry.  Also calculates
	the gradient. 
   	inputs:  3-D, shape: `(max_time x batch_size x num_classes)`, the logits.
	sequence_length:  A vector containing sequence lengths (batch).
   	indexs:  The indexs of a `Tensor<int32, 3>`.
	  `indexs(i, :) == [b, instate, tostate]` means lattice arc instate and tostate,
	pdf_values:
	lm_ws:
	am_ws:
	statesinfo:
	)doc");;



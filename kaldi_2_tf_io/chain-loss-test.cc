
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "base/timer.h"
#include "nnet3/nnet-utils.h"
#include "nnet3/nnet-chain-training.h"
#include "cudamatrix/cu-allocator.h"

#include "fst-convert-openfst.h"
#include "tf-2-kaldi-api.h"
#include "batch_input.h"

int main(int argc, char *argv[])
{
	using namespace kaldi;
	using namespace kaldi::nnet3;

   	typedef kaldi::int32 int32;
	typedef kaldi::int64 int64;

	const char *usage = "";
	ParseOptions po(usage);
	NnetSimpleComputationOptions opts;
	opts.acoustic_scale = 1.0; // by default do no scaling.

	int32 online_ivector_period = 0;
	//bool apply_exp = false, use_priors = false;
	std::string use_gpu = "no";

	Vector<BaseFloat> priors;
	po.Read(argc, argv);

	std::string nnet_rxfilename = po.GetArg(1),
		den_fst_rxfilename = po.GetArg(2),
		examples_rspecifier = po.GetArg(3),
		matrix_wspecifier = po.GetArg(4),
		tf_matrix_wspecifier = po.GetArg(5),
		tf_batch_matrix_wspecifier = po.GetArg(6);
	
	//CuDevice::Instantiate().SelectGpuId(use_gpu);
	// den fst
	fst::StdVectorFst den_fst;
	ReadFstKaldi(den_fst_rxfilename, &den_fst);

#define TEST_TF
#ifdef TEST_TF
		int32 *den_indexs = NULL;
		int32 *den_in_labels = NULL;
		int32 *den_out_labels = NULL;
		BaseFloat *den_weights = NULL;
		int32 *den_stateinfo = NULL;
		int32 den_start_state = 0;
		int32 den_num_states = fst::ConvertKaldiLatticeToSparseLattice(den_fst, &den_indexs, &den_in_labels, &den_out_labels,
				&den_weights, &den_stateinfo, &den_start_state);
		bool delete_laststatesuperfinal = true;
		
		hubo::DenominatorGraphSaver den_graph_saver;
		den_graph_saver.Init(den_indexs, den_in_labels, den_out_labels,
				den_weights, den_stateinfo,
				den_num_states, 3766,
				delete_laststatesuperfinal, den_start_state);

#endif
	//fst::RmEpsilon(&den_fst);
	fst::PrintStandardFst(den_fst);
	chain::DenominatorGraph den_graph(den_fst, 3766);

	Nnet raw_nnet;
	ReadKaldiObject(nnet_rxfilename, &raw_nnet);

	Nnet &nnet = raw_nnet;
	SetBatchnormTestMode(true, &nnet);
	SetDropoutTestMode(true, &nnet);

	CollapseModel(CollapseModelConfig(), &nnet);
	CachingOptimizingCompiler compiler(nnet, opts.optimize_config);

	BaseFloatMatrixWriter matrix_writer(matrix_wspecifier);
	BaseFloatMatrixWriter tf_matrix_writer(tf_matrix_wspecifier);
	BaseFloatMatrixWriter tf_batch_matrix_writer(tf_batch_matrix_wspecifier);
	// 
	SequentialNnetChainExampleReader example_reader(examples_rspecifier);
	chain::ChainTrainingOptions chain_config;
	bool use_xent = false;
	for(; !example_reader.Done(); example_reader.Next())
	{
		const Matrix<BaseFloat> *online_ivectors = NULL;
		const Vector<BaseFloat> *ivector = NULL;
			  
		NnetChainExample &chain_eg = example_reader.Value();

		CuMatrix<BaseFloat> features_gpu(chain_eg.inputs[0].features.NumRows(),
				chain_eg.inputs[0].features.NumCols(),
				kUndefined); 
		features_gpu.CopyFromGeneralMat(chain_eg.inputs[0].features);

		Matrix<BaseFloat> features(features_gpu);
		DecodableNnetSimple nnet_computer(
				opts, nnet, priors,
				features, &compiler,
				ivector, online_ivectors,
				online_ivector_period);

		int32 out_frames = chain_eg.outputs[0].supervision.frames_per_sequence;
		//Matrix<BaseFloat> nnet_output(nnet_computer.NumFrames(),
		Matrix<BaseFloat> nnet_output(out_frames,
				nnet_computer.OutputDim(), 
				kSetZero, kStrideEqualNumCols);
		//for (int32 t = 0; t < nnet_computer.NumFrames(); t++) 
		for (int32 t = 0; t < out_frames; t++) 
		{
			SubVector<BaseFloat> row(nnet_output, t);
			nnet_computer.GetOutputForFrame(t, &row);
		}
		
		CuMatrix<BaseFloat> nnet_output_gpu_tmp(nnet_output.NumRows(),nnet_output.NumCols(), 
				kUndefined, kStrideEqualNumCols);
		nnet_output_gpu_tmp.CopyFromMat(nnet_output);
		const CuMatrixBase<BaseFloat> &nnet_output_gpu = nnet_output_gpu_tmp;
		CuMatrix<BaseFloat> nnet_output_deriv(nnet_output.NumRows(),
				nnet_output.NumCols(),
				kUndefined);
		CuMatrix<BaseFloat> xent_deriv;

		BaseFloat tot_objf, tot_l2_term, tot_weight;
		ComputeChainObjfAndDeriv(chain_config, den_graph, chain_eg.outputs[0].supervision,
				nnet_output_gpu, 
				&tot_objf, &tot_l2_term, &tot_weight,
				&nnet_output_deriv,
				(use_xent ? &xent_deriv : NULL));

#ifdef TEST_TF
		int32 *indexs = NULL;
		int32 *in_labels = NULL;
		int32 *out_labels = NULL;
		BaseFloat *weights = NULL;
		int32 *stateinfo = NULL;
		int32 start_state = 0;
		BaseFloat *deriv_weights = chain_eg.outputs[0].deriv_weights.Data();
		int32 num_states = fst::ConvertKaldiLatticeToSparseLattice(chain_eg.outputs[0].supervision.fst, &indexs, &in_labels, &out_labels,
				&weights, &stateinfo, &start_state);
		int32 num_arc = stateinfo[2*(num_states-1)+0] + stateinfo[2*(num_states-1)+1];
		
		//Matrix<BaseFloat> nnet_output_tf(nnet_output.NumRows(), nnet_output.NumCols(), kSetZero, kStrideEqualNumCols);
		//

		CuMatrix<BaseFloat> nnet_output_deriv_tf(nnet_output.NumRows(),
				nnet_output.NumCols(), 
				kSetZero, kStrideEqualNumCols);
		BaseFloat  objf[3];

		hubo::ChainLossDen(indexs, in_labels, out_labels, weights, stateinfo, &num_states,
				num_arc, num_states,
				deriv_weights,
				chain_eg.outputs[0].supervision.weight, chain_eg.outputs[0].supervision.num_sequences,
				chain_eg.outputs[0].supervision.frames_per_sequence, chain_eg.outputs[0].supervision.label_dim,
				nnet_output_gpu_tmp.Data(),
				nnet_output_gpu_tmp.NumRows(), 1, nnet_output_gpu_tmp.NumCols(),
				den_graph_saver,
				nnet_output_deriv_tf.Data(),
				objf,
				chain_config.l2_regularize, chain_config.leaky_hmm_coefficient, chain_config.xent_regularize);

/*
		hubo::ChainLoss(indexs, in_labels, out_labels, weights, stateinfo, &num_states,
				num_arc, num_states, 
				chain_eg.outputs[0].supervision.weight, chain_eg.outputs[0].supervision.num_sequences,
				chain_eg.outputs[0].supervision.frames_per_sequence, chain_eg.outputs[0].supervision.label_dim, 
				nnet_output_gpu_tmp.Data(), 
				nnet_output_gpu_tmp.NumRows(), 1, nnet_output_gpu_tmp.NumCols(),
				den_indexs, den_in_labels, den_out_labels, den_weights, den_stateinfo, den_start_state,
				den_num_states,
				nnet_output_deriv_tf.Data(),
				objf,
				chain_config.l2_regularize, chain_config.leaky_hmm_coefficient, chain_config.xent_regularize);
*/
		Matrix<BaseFloat> tf_output_deriv(nnet_output_deriv_tf);
		tf_matrix_writer.Write("aaa", tf_output_deriv);

		// batch test
		{
			int32 max_num_arcs = num_arc;
			int32 max_num_states = num_states;
			int32 batch_size = 3;
			indexs = BatchIn(indexs, max_num_arcs * 2, batch_size);
			in_labels = BatchIn(in_labels, max_num_arcs, batch_size);
			out_labels = BatchIn(out_labels, max_num_arcs, batch_size);
			weights = BatchIn(weights, max_num_arcs, batch_size);
			stateinfo = BatchIn(stateinfo, max_num_states * 2 , batch_size);
			
			Vector<BaseFloat> deriv_weights_t;
			deriv_weights_t.Resize(out_frames*batch_size, kUndefined);
			for(int32 n = 0; n < batch_size; n++)
			{
				for (int32 t = 0; t < out_frames; t++)
				{
					deriv_weights_t(t * batch_size + n) = deriv_weights[t];
				}
			}
			deriv_weights = deriv_weights_t.Data();
			int32 *batch_num_states = new int32[1];
			batch_num_states[0] = num_states;
			batch_num_states = BatchIn(batch_num_states, 1, batch_size);
			
			Matrix<BaseFloat> nnet_output_batch(out_frames * batch_size,
					nnet_computer.OutputDim(), 
					kSetZero, kStrideEqualNumCols);
			//for (int32 t = 0; t < nnet_computer.NumFrames(); t++) 
			for (int32 t = 0; t < out_frames; t++) 
			{
				for(int32 i =0 ; i < batch_size; i++)
				{
					SubVector<BaseFloat> row(nnet_output_batch, t*batch_size+i);
					nnet_computer.GetOutputForFrame(t, &row);
				}
			}
			CuMatrix<BaseFloat> nnet_output_gpu_batch_tmp(nnet_output_batch.NumRows(),nnet_output_batch.NumCols(), 
					kUndefined, kStrideEqualNumCols);
			nnet_output_gpu_batch_tmp.CopyFromMat(nnet_output_batch);
			//const CuMatrixBase<BaseFloat> &nnet_output_gpu_batch = nnet_output_gpu_batch_tmp;
			CuMatrix<BaseFloat> nnet_output_deriv_batch_tf(nnet_output_batch.NumRows(),
					nnet_output_batch.NumCols(), 
					kSetZero, kStrideEqualNumCols);
		
			BaseFloat  objf[3];
			hubo::ChainLossDen(indexs, in_labels, out_labels, weights, stateinfo, batch_num_states,
					num_arc, num_states,
					deriv_weights,
					chain_eg.outputs[0].supervision.weight, chain_eg.outputs[0].supervision.num_sequences,
					chain_eg.outputs[0].supervision.frames_per_sequence, chain_eg.outputs[0].supervision.label_dim, 
					nnet_output_gpu_batch_tmp.Data(), 
					nnet_output_gpu_batch_tmp.NumRows()/batch_size, batch_size, nnet_output_gpu_batch_tmp.NumCols(),
//					den_indexs, den_in_labels, den_out_labels, den_weights, den_stateinfo, den_start_state,
//					den_num_states,
					den_graph_saver,
					nnet_output_deriv_batch_tf.Data(),
					objf,
					chain_config.l2_regularize, chain_config.leaky_hmm_coefficient, chain_config.xent_regularize);

			Matrix<BaseFloat> tf_batch_output_deriv(nnet_output_deriv_batch_tf);
			tf_batch_matrix_writer.Write("aaa", tf_batch_output_deriv);


		}
		CuDevice::Instantiate().PrintProfile();
#endif
		Matrix<BaseFloat> output_deriv(nnet_output_deriv);
		matrix_writer.Write("aaa", output_deriv);

		return 0;

	}

	return 0;
}

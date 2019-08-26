
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "base/timer.h"
#include "nnet3/nnet-utils.h"
#include "nnet3/nnet-chain-training.h"
#include "cudamatrix/cu-allocator.h"



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
	std::string use_gpu = "yes";

	Vector<BaseFloat> priors;
	po.Read(argc, argv);

	std::string nnet_rxfilename = po.GetArg(1),
		den_fst_rxfilename = po.GetArg(2),
		examples_rspecifier = po.GetArg(3),
		matrix_wspecifier = po.GetArg(4);
	
	// den fst
	fst::StdVectorFst den_fst;
	ReadFstKaldi(den_fst_rxfilename, &den_fst);

	chain::DenominatorGraph den_graph(den_fst, 3766);

	Nnet raw_nnet;
	ReadKaldiObject(nnet_rxfilename, &raw_nnet);

	Nnet &nnet = raw_nnet;
	SetBatchnormTestMode(true, &nnet);
	SetDropoutTestMode(true, &nnet);

	CollapseModel(CollapseModelConfig(), &nnet);
	CachingOptimizingCompiler compiler(nnet, opts.optimize_config);

	BaseFloatMatrixWriter matrix_writer(matrix_wspecifier);
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

		Matrix<BaseFloat> nnet_output(nnet_computer.NumFrames(),
				nnet_computer.OutputDim());
		for (int32 t = 0; t < nnet_computer.NumFrames(); t++) 
		{
			SubVector<BaseFloat> row(nnet_output, t);
			nnet_computer.GetOutputForFrame(t, &row);
		}
		
		CuMatrix<BaseFloat> nnet_output_gpu_tmp(nnet_output);
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

		Matrix<BaseFloat> output_deriv(nnet_output_deriv);
		matrix_writer.Write("aaa", output_deriv);

	}

	return 0;
}

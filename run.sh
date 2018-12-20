#!/bin/bash  -x


export CUDA_DEVICE_ORDER=PCI_BUS_ID

outdir=data-ce-whole-cnn-blstm-cv-12-19
mkdir -p $outdir
#nnetconf=conf/nnet.conf
#nnetconf=conf/blstm_nnet.conf
nnetconf=conf/cnn_blstm_nnet.conf
conf=conf/config.ini-whole-cnn-blstm
checkpoint_dir=$outdir/checkpoints
ps_hosts="127.0.0.1:2286"
worker_hosts="127.0.0.1:2245,127.0.0.1:2246,127.0.0.1:2247,127.0.0.1:2248"
#worker_hosts="127.0.0.1:2245"
rm -f $outdir/ps.*.log.new $outdir/worker.*.log.new

CUDA_VISIBLE_DEVICES=false python3 new-train.py \
	--checkpoint_dir=$checkpoint_dir \
	--log_file=$outdir/full_data.log \
	--nnet_conf=$nnetconf \
	--config=$conf \
   	--num_threads=1 \
	--ps-hosts=$ps_hosts \
	--worker_hosts=$worker_hosts \
	--job-name=ps \
	--task-index=0  1>> $outdir/ps.${ps_hosts}.log.new 2>&1 &

sleep 1
startid=0
endid=3
for i in `seq $startid $endid`
do
	declare -i task_index=$i-$startid
	gpu=$i
	CUDA_VISIBLE_DEVICES="$gpu" python3 new-train.py \
		--checkpoint_dir=$checkpoint_dir \
		--log_file=$outdir/full_data.log.$task_index \
		--nnet_conf=$nnetconf \
		--config=$conf \
		--num_threads=1 \
		--ps-hosts=$ps_hosts \
		--worker_hosts=$worker_hosts \
		--job-name=worker \
		--task-index=$task_index 1>>  $outdir/worker.${task_index}.log.new 2>&1 &
done


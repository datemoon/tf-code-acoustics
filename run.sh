#!/bin/bash


export CUDA_DEVICE_ORDER=PCI_BUS_ID

outdir=data-ctc-cv-new-12-11
mkdir -p $outdir
checkpoint_dir=$outdir/checkpoints
ps_hosts="127.0.0.1:2286"
worker_hosts="127.0.0.1:2245,127.0.0.1:2246,127.0.0.1:2247,127.0.0.1:2248"
rm -f $outdir/ps.*.log.new $outdir/worker.*.log.new

CUDA_VISIBLE_DEVICES=false python3 new-train.py \
	--checkpoint_dir=$checkpoint_dir \
	--log_file=$outdir/full_data.log \
	--config=conf/config.ini-ctc-new --num_threads=1 \
	--ps-hosts=$ps_hosts \
	--worker_hosts=$worker_hosts \
	--job-name=ps \
	--task-index=0  1>> $outdir/ps.${ps_hosts}.log.new 2>&1 &

sleep 1
startid=4
endid=7
for i in `seq $startid $endid`
do
	declare -i task_index=$i-$startid
	gpu=$i
	CUDA_VISIBLE_DEVICES="$gpu" python3 new-train.py \
		--checkpoint_dir=$checkpoint_dir \
		--log_file=$outdir/full_data.log.$task_index \
		--config=conf/config.ini-ctc-new --num_threads=1 \
		--ps-hosts=$ps_hosts \
		--worker_hosts=$worker_hosts \
		--job-name=worker \
		--task-index=$task_index 1>>  $outdir/worker.${task_index}.log.new 2>&1 &
done


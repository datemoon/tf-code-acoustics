#!/bin/bash

if [ $# != 4 ]
then
	echo $0 bin feat_pre split  outdir
	exit 0
fi

bin=$1
feat_name=$2

split=`echo $3-1|bc`
outdir=$4
mkdir -p $outdir

for i in `seq 0 $split`
do
	nohup $bin --compress=false scp:$feat_name$i \
		ark,scp:$outdir/feats.ark_${i},$outdir/feats.scp_${i} > $outdir/log.$i &
done


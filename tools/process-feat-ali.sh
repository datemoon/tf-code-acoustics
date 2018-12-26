#!/bin/bash


if [ $# != 3 ]
then
	echo $0 ali feat.scp split(0) outdir
	exit 1
fi
ali=$1
feat_scp=$2
alifile=${ali##*/}
feat_scpfile=${feat_scp##*/}
split=$3
outdir=$4
mkdir -p $outdir
awk '{printf("%d\n",NF);}' $ali > $outdir/ali.len
paste $outdir/ali.len $ali|sort -nk 1 |while read len aliinfo
do
	echo $aliinfo 
done > $outdir/sort_$alifile || exit 1

python map-feat-ali.py $feat_scp $outdir/sort_$alifile $outdir/sort_$feat_scpfile || exit 1

python split_feat_ali.py $outdir/sort_$feat_scpfile  \
	$outdir/sort_$alifile $split \
	$outdir/split || exit 1

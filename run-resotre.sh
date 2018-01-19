#!/bin/bash -x

export CUDA_DEVICE_ORDER=PCI_BUS_ID

if [ $# != 1 ]
then
	echo you must be input $0 "[ctc|ce|clean]"
	exit 0
fi

if [ $# == 1 ] && [ $1 == clean ]
then
	find -name "*.pyc" |xargs rm -f
	echo clean *.pyc
	exit 0
fi

if [ $1 == ce ]
then
	CUDA_VISIBLE_DEVICES="0" python ce_train_model.py --config=conf/config.ini --num_threads=1
elif [ $1 == ctc ]
then
	CUDA_VISIBLE_DEVICES="0,1,2,3" python ctc_train_model.py --config=conf/config.ini-ctc --num_threads=4
	#CUDA_VISIBLE_DEVICES="7" python ctc_train_model.py --config=conf/config.ini-ctc --num_threads=1
elif [ $1 == "ctc-phole" ]
then
	 CUDA_VISIBLE_DEVICES="2" python ctc_train_model.py --config=conf/config.ini-ctc-usephole --num_threads=1
else
	echo no this option
fi



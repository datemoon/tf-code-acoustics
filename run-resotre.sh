#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID

if [ $1 == clean ]
then
	find -name "*.pyc" |xargs rm -f
	exit 0
fi

CUDA_VISIBLE_DEVICES="1" python ce_train_model.py --config=conf/config.ini-2 --num_threads=1




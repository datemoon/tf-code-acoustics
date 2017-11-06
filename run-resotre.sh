#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID

CUDA_VISIBLE_DEVICES="1" python ce_train_model.py --train --config conf/config.ini_2 --num_threads 1


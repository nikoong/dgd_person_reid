#!/usr/bin/env bash
# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

source scripts/routines.sh


# Evaluate performance

result_dir=/home/nikoong/Algorithm_test/attribute/external/exp/results/baseline/DukeMTMC-reID_DUCK_add_attri_iter_30000_fc7_bn
echo $result_dir
python2 eval/metric_learning_moresimple.py ${result_dir}




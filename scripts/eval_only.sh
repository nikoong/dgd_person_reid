#!/usr/bin/env bash
# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

source scripts/routines.sh


exp=whole_body
iter="_jstl_iter_20000_fc7_bn"

trained_model=external/exp/snapshots/whole_body_jstl/jstl_iter_20000.caffemodel

# Extract features on all datasets
for dataset in cuhk03 cuhk01 prid viper 3dpes ilids; do
  extract_features ${exp} ${dataset} ${trained_model}
done

# Evaluate performance
for dataset in cuhk03 cuhk01 prid viper 3dpes ilids ilids; do
  result_dir=/home/nikoong/Algorithm_test/dgd_person_reid/external/exp/results/${exp}/${dataset}${iter}
  echo ${dataset}
  python2 eval/metric_learning.py ${result_dir}
  echo
done

#!/usr/bin/env bash
# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

source scripts/routines.sh

exp='jstl'
echo $exp

trained_model=/home/nikoong/Algorithm_test/dgd_person_reid/external/exp/snapshots/jstl/train_adjst_iter_30000.caffemodel


# Extract features on all datasets
for dataset in cuhk03 cuhk01 prid viper 3dpes ilids; do
  extract_features ${exp} ${dataset} ${trained_model}
done

# Evaluate performance
for dataset in cuhk03 cuhk01 prid viper 3dpes ilids; do
  result_dir=$(get_result_dir ${exp} ${dataset} ${trained_model})
  echo ${dataset}
  python2 eval/metric_learning.py ${result_dir}
  echo
done


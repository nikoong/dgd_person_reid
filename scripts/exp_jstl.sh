#!/usr/bin/env bash
# Experiments of joint single task learning (JSTL)

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

source scripts/routines.sh

exp='jstl'
log_name=test_once
pretrained_model=external/exp/snapshots/jstl/jstl_iter_26000.caffemodel
# Train JSTL model
#train_model ${exp} jstl
train_model ${exp} jstl ${pretrained_model} ${log_name}
trained_model=$(get_trained_model ${exp} jstl)
#trained_model=external/exp/snapshots/whole_body_jstl/jstl_iter_55000.caffemodel
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

#!/usr/bin/env bash
# Experiments of joint single task learning (JSTL)

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

source scripts/routines.sh

exp='triplet'
trained_model=external/exp/snapshots/triplet/jstl_iter_10000.caffemodel
#trained_model=external/exp/snapshots/triplet/fc9+norm+ran_num=1.4/jstl_iter_10000.caffemodel
blob=fc9

# Train JSTL model
train_model ${exp} jstl ${trained_model}
trained_model=$(get_trained_model ${exp} jstl)

echo ${trained_model}

# Extract features on all datasets
for dataset in cuhk03 cuhk01 prid viper 3dpes ilids; do
  extract_features ${exp} ${dataset} ${trained_model} ${blob}
done

# Evaluate performance
for dataset in cuhk03 cuhk01 prid viper 3dpes ilids; do
  result_dir=$(get_result_dir ${exp} ${dataset} ${trained_model})
  echo ${dataset}
  python2 eval/metric_learning.py ${result_dir}
  echo
done

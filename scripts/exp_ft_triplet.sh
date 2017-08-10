#!/usr/bin/env bash
# Experiments of fine-tuning on each dataset from JSTL+DGD.

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

source scripts/routines.sh

exp='triplet'
blob=fc9
pretrained_model=external/exp/snapshots/triplet/triplet_10000.caffemodel
log_name=ft_triplet

# Fine-tune on each dataset
for dataset in cuhk03 cuhk01 prid viper 3dpes ilids; do
  train_model ${exp} ${dataset} ${pretrained_model} ${log_name}
done

# Extract features on all datasets
for dataset in cuhk03 cuhk01 prid viper 3dpes ilids; do
  trained_model=$(get_trained_model ${exp} ${dataset})
  extract_features ${exp} ${dataset} ${trained_model} ${blob}
done

# Evaluate performance
for dataset in cuhk03 cuhk01 prid viper 3dpes ilids; do
  trained_model=$(get_trained_model ${exp} ${dataset})
  result_dir=$(get_result_dir ${exp} ${dataset} ${trained_model})
  echo ${dataset}
  python2 eval/metric_learning.py ${result_dir}
  echo
done


#!/usr/bin/env bash
# Experiments of fine-tuning on each dataset from JSTL+DGD.

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

source scripts/routines.sh

exp='soft_ft_triplet_ft'
blob=fc9
log_name=soft_ft_triplet_ft
:<<BLANK
# Fine-tune on each dataset
for dataset in 3dpes cuhk01; do
  pretrained_model=$(get_trained_model jstl_ft ${dataset})
  train_model ${exp} ${dataset} ${pretrained_model} ${log_name}
done


# Extract features on all datasets
for dataset in cuhk03 cuhk01 prid viper 3dpes ilids; do
  trained_model=$(get_trained_model ${exp} ${dataset})
  extract_features ${exp} ${dataset} ${trained_model} ${blob}
done
BLANK
# Evaluate performance
for dataset in cuhk03 cuhk01 prid viper 3dpes ilids; do
  trained_model=$(get_trained_model ${exp} ${dataset})
  result_dir=$(get_result_dir ${exp} ${dataset} ${trained_model})
  echo ${dataset}
  python2 eval/metric_learning.py ${result_dir}
  echo
done


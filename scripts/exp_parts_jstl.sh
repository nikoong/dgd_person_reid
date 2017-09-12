#!/usr/bin/env bash
# Experiments of joint single task learning (JSTL)

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

source scripts/routines.sh


part=whole_body 
log_name=whole_body_poly

exp=$part'_jstl'
#pretrained_model=Pretrained_models/whole_body_jstl_iter_55000.caffemodel


# Train JSTL model

#train_model ${exp} jstl ${pretrained_model} ${log_name}
train_model ${exp} jstl ${log_name}
trained_model=$(get_trained_model ${exp} jstl)
echo ${trained_model}


# Extract features on all datasets
for dataset in cuhk03 cuhk01 prid viper 3dpes ilids; do
  extract_parts_features ${exp} ${dataset} ${trained_model} ${part}
done

# Evaluate performance
for dataset in cuhk03 cuhk01 prid viper 3dpes ilids; do
  result_dir=$(get_result_dir ${exp} ${dataset} ${trained_model})
  echo ${dataset}
  python2 eval/metric_learning.py ${result_dir}
  echo
done

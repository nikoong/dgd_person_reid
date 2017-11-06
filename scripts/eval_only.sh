#!/usr/bin/env bash
# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

source scripts/routines.sh


exp=triplet_test
iter="iter_10000"
echo $exp


:<<EEEE
trained_model=external/exp/snapshots/whole_body_jstl/jstl_iter_20000.caffemodel
# Extract features on all datasets
for dataset in cuhk03 cuhk01 prid viper 3dpes ilids; do
  extract_features ${exp} ${dataset} ${trained_model}
done
EEEE

trained_model=/home/nikoong/Algorithm_test/dgd_person_reid/external/exp/snapshots/triplet/triplet_10000.caffemodel
# Extract features on all datasets
for dataset in cuhk03 cuhk01 prid viper 3dpes ilids; do
  extract_features ${exp} ${dataset} ${trained_model}
done



# Evaluate performance
for dataset in cuhk03 cuhk01 prid viper 3dpes ilids; do
  result_dir=/home/nikoong/Algorithm_test/dgd_person_reid/external/exp/results/${exp}/${dataset}${iter}
  echo ${dataset}
  python2 eval/metric_learning.py ${result_dir}
  echo
done


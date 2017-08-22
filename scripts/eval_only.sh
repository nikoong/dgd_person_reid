
# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

# Evaluate performance
for dataset in cuhk03 cuhk01 prid viper 3dpes ilids; do
  result_dir=/home/nikoong/Algorithm_test/dgd_person_reid/external/exp/results/up\&down_jstl/${dataset}"_jstl_iter_55000_fc7_bn"
  #echo ${result_dir}
  echo ${dataset}
  python2 eval/metric_learning.py ${result_dir}
  echo
done


cd $(dirname ${BASH_SOURCE[0]})/../
MODELS_DIR=models

get_trained_model() {
  local exp=$1
  local dataset=$2

  local solver=${MODELS_DIR}/${exp}/${dataset}_solver.prototxt
  local max_iter=$(grep 'max_iter' ${solver} | awk '{print $2}')
  local snapshot_prefix=$(grep 'snapshot_prefix' ${solver} | awk -F '"' '{print $2}')
  local model=${snapshot_prefix}_iter_${max_iter}.caffemodel
  echo ${snapshot_prefix}
}
get_trained_model jstl jstl

#!/usr/bin/env bash
cd $(dirname ${BASH_SOURCE[0]})/../



EXP_DIR=external/exp
RESULTS_DIR=$EXP_DIR/results
CAFFE_DIR=external/caffe



extract_features() {
  local prototxt=$1
  local test_probe_txt=$2
  local test_gallery_txt=$3
  local trained_model=$4
  local result_dir=$5
  if [[ $# -eq 6 ]]; then
    local blob=$6
  else
    local blob=fc7_bn
  fi
  local batch_size=1
  rm -rf ${result_dir}
  mkdir -p ${result_dir}

  # Extract test probe features.
  echo "Extracting ${test_probe_txt} set"
  local num_samples=$(wc -l ${test_probe_txt} | awk '{print $1}')
  local num_samples=$((num_samples + 1))
  local num_iters=$(((num_samples + ${batch_size} - 1) / ${batch_size}))
  local model=$(mktemp)
  echo "exfeat_template" ${test_probe_txt} 
  sed -e "s|\${batch_size}|${batch_size}|g; s|\${txt_path}|${test_probe_txt}|g" \
    ${prototxt} > ${model}
  ${CAFFE_DIR}/build/tools/extract_features \
    ${trained_model} ${model} ${blob},label \
    ${result_dir}/test_probe_features_lmdb,${result_dir}/test_probe_labels_lmdb \
    ${num_iters} lmdb GPU 0

  python2 tools/convert_lmdb_to_numpy.py \
    ${result_dir}/test_probe_features_lmdb ${result_dir}/test_probe_features.npy \
    --truncate ${num_samples}
  python2 tools/convert_lmdb_to_numpy.py \
    ${result_dir}/test_probe_labels_lmdb ${result_dir}/test_probe_labels.npy \
    --truncate ${num_samples}

  # Extract test gallery featuresand
  echo "Extracting ${test_gallery_txt} set"
  local num_samples=$(wc -l ${test_gallery_txt} | awk '{print $1}')
  local num_samples=$((num_samples + 1))
  local num_iters=$(((num_samples + ${batch_size} - 1) / ${batch_size}))
  local model=$(mktemp)
  echo "exfeat_template" ${test_gallery_txt}
  sed -e "s|\${batch_size}|${batch_size}|g; s|\${txt_path}|${test_gallery_txt}|g" \
    ${prototxt} > ${model}
  ${CAFFE_DIR}/build/tools/extract_features \
    ${trained_model} ${model} ${blob},label \
    ${result_dir}/test_gallery_features_lmdb,${result_dir}/test_gallery_labels_lmdb \
    ${num_iters} lmdb GPU 0
  python2 tools/convert_lmdb_to_numpy.py \
    ${result_dir}/test_gallery_features_lmdb ${result_dir}/test_gallery_features.npy \
    --truncate ${num_samples}
  python2 tools/convert_lmdb_to_numpy.py \
    ${result_dir}/test_gallery_labels_lmdb ${result_dir}/test_gallery_labels.npy \
    --truncate ${num_samples}
}



prototxt=/home/nikoong/Algorithm_test/dgd_person_reid/test_dir/exfeat_template.prototxt
test_gallery_txt=/home/nikoong/Algorithm_test/dgd_person_reid/external/exp/datasets/cuhk03/txt/test_gallery.txt
trained_model=/home/nikoong/Algorithm_test/dgd_person_reid/Pretrained_models/jstl_dgd.caffemodel
result_dir=test_dir/jstl_iter_20000_fc7_bn

for test_probe_txt in /home/nikoong/Algorithm_test/dgd_person_reid/external/exp/datasets/test10/txt/颜色.txt \
/home/nikoong/Algorithm_test/dgd_person_reid/external/exp/datasets/test10/txt/挎包.txt \
/home/nikoong/Algorithm_test/dgd_person_reid/external/exp/datasets/test10/txt/纹理.txt \
/home/nikoong/Algorithm_test/dgd_person_reid/external/exp/datasets/test10/txt/遮挡.txt \
/home/nikoong/Algorithm_test/dgd_person_reid/external/exp/datasets/test10/txt/旋转.txt \
/home/nikoong/Algorithm_test/dgd_person_reid/external/exp/datasets/test10/txt/背包.txt;do
  extract_features ${prototxt} ${test_probe_txt} ${test_gallery_txt} ${trained_model} ${result_dir}
  python2 eval/metric_learning_moresimple.py ${result_dir}
done






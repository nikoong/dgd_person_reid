#!/usr/bin/env bash
cd $(dirname ${BASH_SOURCE[0]})/../

CAFFE_DIR=external/caffe
EXP_DIR=external/exp
DATASETS_DIR=${EXP_DIR}/datasets
DB_DIR=${EXP_DIR}/db
RESULTS_DIR=${EXP_DIR}/results
SNAPSHOTS_DIR=${EXP_DIR}/snapshots
NIS_DIR=${EXP_DIR}/nis

layer=fc2

#edit
model=/home/nikoong/Algorithm_test/caffe-master/ours_model/vgg/VGG_sy/origin/cuhk03_trainval.prototxt
finetuned_model=/home/nikoong/Algorithm_test/output/VGG_sy/vgg_origin_batch100_iter_20000.caffemodel
output_npy=/home/nikoong/Algorithm_test/caffe-master/ours_model/nis_npy/vgg_sy_origin_fc2_norm.npy
output_lmdb=/home/nikoong/Algorithm_test/caffe-master/ours_model/nis_lmdb_dir/vgg_sy_origin_fc2_norm_lmdb




num_samples=$(wc -l /home/nikoong/Algorithm_test/New_dataset/train.txt | awk '{print $1}')
num_samples=$((num_samples + 1))
num_iters=$(((num_samples + 99) / 100))


mkdir -p $(dirname ${output_npy})

echo num_iters = ${num_iters}


python2 tools/compute_impact_score.py \
  ${model} ${finetuned_model} ${output_npy} \
  --num_iters ${num_iters} --layer ${layer} --normalize

python2 tools/save_individual_impact_score.py \
  ${output_npy} ${output_lmdb}

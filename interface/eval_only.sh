#!/usr/bin/env bash
# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

#source scripts/routines.sh

:<<EEEE


EEEE

trained_model=/home/nikoong/Algorithm_test/dgd_person_reid/external/exp/snapshots/dgd/jstl_iter_20000.caffemodel
result_dir=/home/nikoong/Algorithm_test/dgd_person_reid/interface/temp
blob=fc7_bn

  for subset in probe ; do   
    num_samples=$(wc -l /home/nikoong/Algorithm_test/dgd_person_reid/interface/txt/${subset}.txt | awk '{print $1}')
    num_samples=$((num_samples + 1))
    num_iters=$(((num_samples + 49) / 50))
    model=$(mktemp)
    
    sed -e "s/\${subset}/${subset}/g" \
      /home/nikoong/Algorithm_test/dgd_person_reid/interface/models/exfeat_template.prototxt > ${model}
    echo "exfeat_template" /home/nikoong/Algorithm_test/dgd_person_reid/interface/models/exfeat_template.prototxt/exfeat_template.prototxt
    /home/nikoong/Algorithm_test/caffe-master/build/tools/extract_features \
      ${trained_model} ${model} ${blob},label \
      ${result_dir}/${subset}_features_lmdb,${result_dir}/${subset}_labels_lmdb \
      ${num_iters} lmdb GPU 0
    

    python2 tools/convert_lmdb_to_numpy.py \
      ${result_dir}/${subset}_features_lmdb ${result_dir}/${subset}_features.npy \
      --truncate ${num_samples}
    python2 tools/convert_lmdb_to_numpy.py \
      ${result_dir}/${subset}_labels_lmdb ${result_dir}/${subset}_labels.npy \
      --truncate ${num_samples}
  done


# Evaluate performance
python2 /home/nikoong/Algorithm_test/dgd_person_reid/interface/metric_learning_simple.py \
/home/nikoong/Algorithm_test/dgd_person_reid/interface/temp \
/home/nikoong/Algorithm_test/dgd_person_reid/external/exp/datasets/cuhk03/cam_0/00636_00007.jpg \
/home/nikoong/Algorithm_test/dgd_person_reid/interface/txt/gallery.txt \


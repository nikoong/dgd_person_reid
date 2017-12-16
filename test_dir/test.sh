batch_size=50
test_probe_txt=a
sed -e "s/\${batch_size}/${batch_size}/g; s/\${txt_path}/${test_probe_txt}/g" a.prototxt


${CAFFE_DIR}/build/tools/extract_features \
  ${trained_model} ${prototxt} ${blob},label \
  ${result_dir}/test_probe_features_lmdb,${result_dir}/test_probe_labels_lmdb \
  ${num_iters} lmdb GPU 0
python2 tools/convert_lmdb_to_numpy.py \
  ${result_dir}/test_probe_features_lmdb ${result_dir}/test_probe_features.npy \
  --truncate ${num_samples}

# Extract test gallery featuresand
echo "Extracting ${test_gallery_txt} set"
local num_samples=$(wc -l ${test_gallery_txt} | awk '{print $1}')
local num_samples=$((num_samples + 1))
local num_iters=$(((num_samples + ${batch_size} - 1) / ${batch_size}))
local model=$(mktemp)
echo "exfeat_template" ${test_gallery_txt}
sed -e "s/\${batch_size}/${batch_size}/g; s/\${txt_path}/${test_gallery_txt}/g" \
  ${prototxt} > ${model}
${CAFFE_DIR}/build/tools/extract_features \
  ${trained_model} ${prototxt} ${blob},label \
  ${result_dir}/test_gallery_features_lmdb,${result_dir}/test_gallery_labels_lmdb \
  ${num_iters} lmdb GPU 0
python2 tools/convert_lmdb_to_numpy.py \
  ${result_dir}/test_gallery_labels_lmdb ${result_dir}/test_gallery_labels.npy \
  --truncate ${num_samples}


  # Evaluate performance
for dataset in ${datasets}; do
  result_dir=$(get_result_dir ${exp} ${dataset} ${test_weights})
  echo ${dataset}
  python2 eval/metric_learning_simple.py ${result_dir}
  echo
done
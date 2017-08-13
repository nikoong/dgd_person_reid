The deploy prototxts and pretrained caffemodels provided here can be used as generic person re-id feature extractors.

These models cannot be directly used for CUHK03, CUHK01, PRID, VIPeR, i-LIDS, 3DPeS, and Shinpuhkan datasets (as referenced in our paper), because the models are trained on these datasets. However, you can always use them for other datasets, or rerun our experiments with your own data split on the above seven datasets.

Note:

1. jstl_dgd_deploy.prototxt is for jstl_dgd.caffemodel. Has explicit BN layers and needs our modified caffe to run.

2. jstl_dgd_deploy_inference.prototxt is for jstl_dgd_inference.caffemodel. The BN layers are integrated into their bottom layers. All caffe versions compatible with the official one (release candidate 2) should be fine.

3. The mean values are (102, 102, 101) in BGR order.

import numpy as np
from argparse import ArgumentParser
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
import time

from utils import *



def _get_test_data(result_dir):
    PX = np.load(osp.join(result_dir, 'test_probe_features.npy'))
    PY = np.load(osp.join(result_dir, 'test_probe_labels.npy'))
    GX = np.load(osp.join(result_dir, 'test_gallery_features.npy'))
    GY = np.load(osp.join(result_dir, 'test_gallery_labels.npy'))
    # Reassign the labels to make them sequentially numbered from zero
    unique_labels = np.unique(np.r_[PY, GY])
    labels_map = {l: i for i, l in enumerate(unique_labels)}
    PY = np.asarray([labels_map[l] for l in PY])
    GY = np.asarray([labels_map[l] for l in GY])
    return PX, PY, GX, GY


def _eval_cmc(PX, PY, GX, GY):

    D = pairwise_distances(GX, PX, metric='mahalanobis', VI = np.eye(PX.shape[1]),n_jobs=-2)
    C = cmc(D, GY, PY)
    return C


def main(args):
    PX, PY, GX, GY = _get_test_data(args.result_dir)
    file_suffix = args.method
    localtime = time.asctime( time.localtime(time.time()) )
    print localtime
    C = _eval_cmc(PX, PY, GX, GY)
    localtime = time.asctime( time.localtime(time.time()) )
    print localtime
    for topk in [1, 5, 10, 20, 50]:
        print "{:8}{:8.1%}".format('top-' + str(topk), C[topk - 1])
    np.save(osp.join(args.result_dir, 'cmc_' + file_suffix), C)

    

if __name__ == '__main__':
    parser = ArgumentParser(
            description="Metric learning and evaluate performance")
    parser.add_argument('result_dir',
            help="Result directory. Containing extracted features and labels. "
                 "CMC curve will also be saved to this directory.")
    parser.add_argument('--method', choices=['euclidean', 'kissme'],
            default='euclidean')
    args = parser.parse_args()
    main(args)
# Crf postprocessing, crf function is applied

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
import numpy as np


def crf(img, prob):
    '''
        img.shape:  (height, width, channels), here channels=1
        prob.shape: (height, width, 1), here prob is probabilities
    '''
    img = np.swapaxes(img, 0, 1)    # img.shape: (width, height, channels)
    prob = np.swapaxes(prob, 0, 2)  # prob.shape: (1, width, height)

    n_labels = 2
    probs = np.tile(prob, (n_labels, 1, 1))  # probs.shape: (2, width, height)
    probs[0] = np.subtract(1, prob)  # class 0: background
    probs[1] = prob                  # class 1: pneumothorax

    d = dcrf.DenseCRF(img.shape[0] * img.shape[1], n_labels)

    # ----------------------------- Unary potential -----------------------------
    # 1) unary_from_labels
    #   U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    # 2) unary_from_softmax
    #   converting class probabilities to unary potential, the first parameter should be class
    U = unary_from_softmax(probs)  # U.shape: (2, width * height)
    # return a continuous array, whose memory is continuous
    U = np.ascontiguousarray(U)
    d.setUnaryEnergy(U)

    # --------------------------- Pairwise potentials ---------------------------
    # creating color-independent image features, here just position feature
    # potentially penalizing spatially adjacent small partitions, encouraging more spatial contiguous partitions
    feats = create_pairwise_gaussian(sdims=(50, 50), shape=img.shape[:2])
    d.addPairwiseEnergy(feats, compat=1,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # creating color-related image features
    # the segmentation results from CNN are very rough, which can be improved using local color features
    # chdim=0 if image channels are the first dimension
    feats = create_pairwise_bilateral(sdims=(1, 1), schan=(1,), img=img, chdim=-1)
    d.addPairwiseEnergy(feats, compat=2,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    # ----------------------------------------------------------------------------

    Q = d.inference(5)
    # res.shape:
    #   np.argmax => (width * height, )
    #   reshape   => (width, height)
    res = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1]))

    res = np.swapaxes(res, 0, 1)        # res.shape: (height, width)
    res = np.expand_dims(res, axis=-1)  # res.shape: (height, width, 1)

    return res


def crf_rgb(img, probs):
    '''
        img.shape: (height, width, 3)
        prob.shape: (height, width, 1)
    '''
    labels = probs.astype(np.int32)  # binary values (0 or 1)

    n_labels = 2
    d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

    # Unary potential
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)

    # pairwise potentials
    # creating color-independent image features
    feats = create_pairwise_gaussian(sdims=(5, 5), shape=img.shape[:2])
    d.addPairwiseEnergy(feats, compat=3, kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # creating color-related image features
    feats = create_pairwise_bilateral(sdims=(15, 15), schan=(20, 20, 20), img=img, chdim=2)
    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(5)
    res = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1]))

    return res

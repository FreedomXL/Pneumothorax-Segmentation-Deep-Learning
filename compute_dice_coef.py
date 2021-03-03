# Computing dice coefficient of predicted masks using numpy

import SimpleITK as sitk
import numpy as np
from crf import crf


def dice_coef(y_true, y_pred):

    smooth = 1.
    y_true_flatten = y_true.flatten()
    y_pred_flatten = y_pred.flatten()

    # replacing keras.backend with numpy
    intersection = np.sum(y_true_flatten * y_pred_flatten)
    union = np.sum(y_true_flatten) + np.sum(y_pred_flatten)
    return (2 * intersection + smooth) / (union + smooth)


def read_labels_masks(data_path, pred_mask_path, idx):

    # reading ground truth labels
    true_labels = sitk.ReadImage(data_path + '/segmentation/' + str(idx) + '.nii')
    true_labels = sitk.GetArrayFromImage(true_labels)

    # reading predicted masks (probabilities)
    pred_masks = sitk.ReadImage(pred_mask_path + '/pred_threshold_' + str(idx) + '.nii')
    pred_masks = sitk.GetArrayFromImage(pred_masks)

    return true_labels, pred_masks


def labels_masks_with_crf(data_path, pred_label_path, idx):

    # reading ground truth labels and predicted masks
    true_labels, pred_masks = read_labels_masks(data_path, pred_label_path, idx)

    # reading original images
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(data_path + '/volume/' + str(idx))
    reader.SetFileNames(dicom_names)
    images = reader.Execute()
    images = sitk.GetArrayFromImage(images)  # images.shape: (slices, 512, 512)

    # HU values are restricted to lung window settings [-1100, 300]
    # namely: window width: 1400, window level: -400
    images[images < -1100] = -1100
    images[images > 300] = 300

    # normalization
    max_value, min_value = images.max(), images.min()
    images = images / (max_value - min_value)

    # crf postprocessing
    images = np.expand_dims(images, axis=-1)  # images.shape: (slices, 512, 512, 1)
    pred_masks = np.expand_dims(pred_masks, axis=-1)  # pred_masks.shape: (slices, 512, 512, 1)
    for i in range(pred_masks.shape[0]):
        pred_masks[i] = crf(images[i], pred_masks[i])

    # pred_masks.shape: (slices, 512, 512, 1) => (slices, 512, 512)
    pred_masks = pred_masks.squeeze()

    return true_labels, pred_masks


if __name__ == '__main__':

    data_path = '/home/user/datasets/pneumothorax/'  # dataset path
    pred_label_path = 'pred'                 # path of predicted masks

    true_labels, pred_masks = read_labels_masks(data_path, pred_label_path, 7)
    # applying crf postprocessing
    # true_labels, pred_masks = labels_masks_with_crf(data_path, pred_label_path, 7)
    print('dice_coef:', dice_coef(true_labels, pred_masks))

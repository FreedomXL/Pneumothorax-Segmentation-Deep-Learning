
import SimpleITK as sitk
import numpy as np
import os
from keras.optimizers import Adam
from generate_data.HDF5DatasetGenerator import HDF5DatasetGenerator
from loss import *
from Models.UNet import UNet
from Models.dilated_UNet import dilated_UNet
from Models.ResNet34_UNet import ResNet34_UNet
from Models.ResNet50_UNet import ResNet50_UNet
from Models.MFP_UNet import MFP_UNet
from Models.Attention_UNet import Attention_UNet
from Models.MultiResUNet import MultiResUNet
from Models.UNet2Plus import UNet2Plus
from Models.UNet3Plus import UNet3Plus
from Models.PSPNet import PSPNet


data_path = '/home/user/datasets/pneumothorax/'  # dataset path

def evaluate(model_path, idx):
    '''
        Evaluate performance of single CT slice

        :model_path: path of evaluated model
        :idx: index of evaluated patient
    '''
    model = UNet(input_shape=(512, 512, 1))
    model.compile(optimizer=Adam(lr=1e-4), loss=focal_tversky_loss, metrics=[dice_coef])
    model.load_weights(model_path)

    # reading original images
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(data_path + 'volume/' + str(idx) + '/')
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

    # reading ground truth labels
    labels = sitk.ReadImage(data_path + 'segmentation/' + str(idx) + '.nii')
    labels = sitk.GetArrayFromImage(labels)  # eval_labels.shape: (slices, 512, 512)

    # (slices, 512, 512) => (slices, 512, 512, 1)
    images = np.expand_dims(images, axis=-1)
    labels = np.expand_dims(labels, axis=-1)

    results = model.evaluate(images, labels, batch_size=1)
    for i, metric in enumerate(model.metrics_names):
        print(metric, ':', results[i])


def evaluate_generator(model_path):
    '''
        Evaluate performance of CT slices using generator

        :model_path: path of evaluated model
    '''
    model = UNet(input_shape=(512, 512, 1))
    model.compile(optimizer=Adam(lr=1e-4), loss=focal_tversky_loss, metrics=[dice_coef])
    model.load_weights(model_path)

    image_path = 'generate_data/test.h5'  # h5 file of evaluated patients
    TOTAL_IMAGE = 1814                    # total images in h5 file
    BATCH_SIZE = 2

    reader = HDF5DatasetGenerator(db_path=image_path, batch_size=BATCH_SIZE)
    generator = reader.generator()

    results = model.evaluate_generator(generator, steps=TOTAL_IMAGE//BATCH_SIZE, verbose=1)
    for i, metric in enumerate(model.metrics_names):
        print(metric, ':', results[i])

    reader.close()


def predict(model_path, idx):
    '''
        Predict segmentation mask of single CT slice

        :model_path: path of evaluated model
        :idx: index of evaluated patient
    '''
    pred_dir = 'pred/'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

    model = UNet((512, 512, 1))
    model.compile(optimizer=Adam(lr=1e-4), loss=focal_tversky_loss, metrics=[dice_coef])
    model.load_weights(model_path)

    # reading original images
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(data_path + 'volume/' + str(idx) + '/')
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

    pred_inputs = np.expand_dims(images, axis=-1)  # pred_inputs.shape: (slices, 512, 512, 1)
    pred_masks = model.predict(pred_inputs, batch_size=1, verbose=1)  # pred_masks.shape: (slices, 512, 512, 1)

    # values in pred_masks are probabilities in range [0, 1]
    # converting to binary mask using a threshold of 0.5
    pred_masks[pred_masks >= 0.5] = 1
    pred_masks[pred_masks < 0.5] = 0
    pred_masks = pred_masks.astype(int)

    out = sitk.GetImageFromArray(pred_masks)
    sitk.WriteImage(out, pred_dir + 'pred_threshold_' + str(idx) + '.nii.gz')


if __name__ == '__main__':

    model_path = 'weights.11-0.22.hdf5'
    predict(model_path, 7)

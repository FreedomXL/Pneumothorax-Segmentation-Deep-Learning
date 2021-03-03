# Generating training and validation h5 file

import SimpleITK as sitk
import os
import numpy as np
from generate_data.HDF5DatasetWriter import HDF5DatasetWriter
from keras.preprocessing.image import ImageDataGenerator


IMG_HEIGHT = 512
IMG_WIDTH = 512

def generate_train_val(data_path, output_path, pattern='train'):

    full_slices = []
    full_labels = []

    volumes = os.listdir(data_path + pattern + '_volume/')
    volumes.sort(key=lambda x: int(x))

    for i in volumes:
        # original CT slices are stored in DICOM format
        # ground truth labels are stored in nii format
        img_path = data_path + 'train_volume/' + str(i) + '/'
        label_path = data_path + 'segmentation/' + str(i) + '.nii'

        # reading original CT slices using SimpleITK
        # images.shape: (slices, 512, 512)
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(img_path)
        reader.SetFileNames(dicom_names)
        images = reader.Execute()
        images = sitk.GetArrayFromImage(images)

        # reading ground truth labels
        # labels.shape: (slices, 512, 512)
        labels = sitk.ReadImage(label_path)
        labels = sitk.GetArrayFromImage(labels)

        # ----------------------- preprocessing -----------------------
        # SimpleITK reads the HU value
        # HU values are restricted to lung window settings [-1100, 300]
        # namely: window width: 1400, window level: -400
        images[images < -1100] = -1100
        images[images > 300] = 300

        # normalization
        max_value, min_value = images.max(), images.min()
        images = images / (max_value - min_value)
        # -------------------------------------------------------------

        full_slices.append(images)
        full_labels.append(labels)

    full_slices = np.vstack(full_slices)                # full_slices.shape: (total_slices, 512, 512)
    full_slices = np.expand_dims(full_slices, axis=-1)  # full_slices.shape: (total_slices, 512, 512, 1)
    full_labels = np.vstack(full_labels)                # full_labels.shape: (total_slices, 512, 512)
    full_labels = np.expand_dims(full_labels, axis=-1)  # full_labels.shape: (total_slices, 512, 512, 1)

    # -------------------- data augmentation --------------------
    # horizontal flipping
    # seed = 1
    # slice_datagen = ImageDataGenerator(horizontal_flip=True)
    # label_datagen = ImageDataGenerator(horizontal_flip=True)
    #
    # slice_datagen.fit(full_slices, augment=True, seed=seed)
    # label_datagen.fit(full_labels, augment=True, seed=seed)
    # slice_generator = slice_datagen.flow(full_slices, seed=seed)
    # label_generator = label_datagen.flow(full_labels, seed=seed)
    #
    # train_generator = zip(slice_generator, label_generator)
    # x, y = [], []
    # i = 0
    # for x_batch, y_batch in train_generator:
    #     i += 1
    #     x.append(x_batch)
    #     y.append(y_batch)
    #     if i >= 1:  # here the dataset is only doubled
    #         break
    # x = np.vstack(x)
    # y = np.vstack(y)
    # ----------------------------------------------------

    dataset = HDF5DatasetWriter(img_shape=(full_slices.shape[0], IMG_HEIGHT, IMG_WIDTH, 1),
                                label_shape=(full_slices.shape[0], IMG_HEIGHT, IMG_WIDTH, 1),
                                output_path=output_path)
    print('Build HDF5DatasetWriter finished.')

    dataset.add(full_slices, full_labels)
    print('Add original data finished...')
    # dataset.add(x, y)
    # print('Add augmented data finished...')
    dataset.close()


if __name__ == '__main__':

    data_path = '/home/user/datasets/pneumothorax/'  # dataset path
    # creating training h5 file
    generate_train_val(data_path=data_path, output_path='train.h5', pattern='train')
    # creating validation h5 file
    generate_train_val(data_path=data_path, output_path='val.h5', pattern='val')

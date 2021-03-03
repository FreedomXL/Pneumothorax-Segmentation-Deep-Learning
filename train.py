
import numpy as np
import os
from generate_data.HDF5DatasetGenerator import HDF5DatasetGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
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


train_path = 'generate_data/train.h5'  # h5 file for training
val_path = 'generate_data/val.h5'      # h5 file for validation
save_path = 'save/'                    # path for saving models and logs
TOTAL_TRAIN = 12555                    # total train images
TOTAL_VAL = 2928                       # total validation images
BATCH_SIZE = 2


def train():

    train_reader = HDF5DatasetGenerator(db_path=train_path, batch_size=BATCH_SIZE)
    train_iter = train_reader.generator()

    val_reader = HDF5DatasetGenerator(db_path=val_path, batch_size=BATCH_SIZE)
    val_iter = val_reader.generator()

    model = UNet(input_shape=(512, 512, 1))
    model.compile(optimizer=Adam(lr=1e-4), loss=focal_tversky_loss, metrics=[dice_coef])

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        os.mkdir(save_path + '/model')
        os.mkdir(save_path + '/model/logs')

    model_checkpoint = ModelCheckpoint(save_path + '/model/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                       monitor='val_loss', verbose=1, save_best_only=True)
    tensorboard = TensorBoard(log_dir=save_path + '/model/logs')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, mode='auto')
    callbacks = [model_checkpoint, tensorboard, reduce_lr]

    model.fit_generator(train_iter,
                        steps_per_epoch=TOTAL_TRAIN//BATCH_SIZE,
                        epochs=20,
                        validation_data=val_iter,
                        validation_steps=TOTAL_VAL//BATCH_SIZE,
                        verbose=1,
                        callbacks=callbacks)

    train_reader.close()
    val_reader.close()

    model.save(save_path + '/model/model.h5')
    print('Finished training ......')


if __name__ == '__main__':
    train()

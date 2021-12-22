from tensorflow.keras.losses import categorical_crossentropy
import tensorflow as tf
import LoadData
import TrainingModels
from EEG_Tensorflow_models.Models.TCNet_fusion import TCNet_fusion
from callbacks_of_Felix import LossHistory, lr_schedule, EarlyStoppingAtMinLoss, CustomLearningRateScheduler
import numpy as np


if __name__ == '__main__':
    X_train, y_train, X_valid, y_valid, sfreq = LoadData.load_dataset(dataset_name="BNCI2014001", subject_id=1, low_cut_hz=None, high_cut_hz=None,
                 trial_start_offset_seconds=-0.5,
                 trial_stop_offset_seconds=0, Channels=None, Classes=None, split=True)
    print(y_train.shape)
    print(y_valid.shape)
    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)
    np.save("X_valid.npy", X_valid)
    np.save("y_valid.npy", y_valid)
    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")
    X_valid = np.load("X_valid.npy")
    y_valid = np.load("y_valid.npy")
    model = TCNet_fusion(nb_classes=4,
                         Chans=22,
                         Samples=1125,
                         layers=2,
                         kernel_s=4,
                         filt=12,
                         dropout=0.3,
                         activation='elu',
                         F1=24,
                         D=2,
                         kernLength=32,
                         )

    lossHistory = LossHistory()
    earlyStoppingAtMinLoss = EarlyStoppingAtMinLoss(patience=100)
    customLearningRateScheduler = CustomLearningRateScheduler()

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    # train model
    model.fit(
        X_train, y_train,
        batch_size=64, epochs=100,
        validation_data=(X_valid, y_valid),
        #validation_split=0.1, shuffle=True,
        callbacks=[lossHistory, customLearningRateScheduler]
    )

    lossHistory.loss_plot('epoch')




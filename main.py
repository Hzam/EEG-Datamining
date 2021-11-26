from keras.utils.vis_utils import plot_model
from tensorflow.keras.losses import categorical_crossentropy
from models_of_Felix import ACRNN_4D
from callbacks_of_Felix import LossHistory, EarlyStoppingAtMinLoss, CustomLearningRateScheduler
from datasets import MNIST, KaggleEDFMASD, BCICompitionIV
from EEG_Tensorflow_models.Models import TCNet_fusion, EEGTCNET

def BCIC_IV_ACRNN_4D():
    # prepare dataset
    # train_images, train_labels, test_images, test_labels = MNIST().loadData()
    # Xtrain, Xtest, ytrain, ytest = KaggleEDFMASD().loadData(path='../../EEG Data/Disposed/withoutFeatureExtration/subject_1.pkl', format='3D')
    Xtrain, Xtest, ytrain, ytest = BCICompitionIV().loadData(path='../../data/data.mat', format='4D')
    # callbacks module
    lossHistory = LossHistory()
    earlyStoppingAtMinLoss = EarlyStoppingAtMinLoss(patience=10)
    customLearningRateScheduler = CustomLearningRateScheduler()

    # build model
    ACRNN_object = ACRNN_4D(input_shape=Xtrain.shape[1:], class_num=ytrain.shape[1])
    model = ACRNN_object.build_model_withAttention()
    model.compile(
        optimizer='adam',
        loss=categorical_crossentropy,
        metrics=['accuracy']
    )

    # train model
    model.fit(
        Xtrain, ytrain,
        batch_size=32, epochs=80,
        validation_data=(Xtest, ytest),
        # validation_split=0.1, shuffle=True,
        callbacks=[lossHistory, customLearningRateScheduler]
    )

    # evaluate and visualization
    score = model.evaluate(Xtest, ytest, verbose=1)
    print('test loss:', score[0])
    print('test accuracy:', score[1])
    lossHistory.loss_plot('epoch')


def BCIC_IV_EGGTCNET():
    Xtrain, Xtest, ytrain, ytest = BCICompitionIV().loadData(path='../../data/data.mat', format='3D')
    F1 = 8
    KE = 32
    KT = 4
    L = 2
    FT = 12
    pe = 0.2
    pt = 0.3
    classes = 4
    channels = 22
    model = EEGTCNET.EEGTCNet(nb_classes = 4,Chans=22, Samples=1125, layers=L, kernel_s=KT,filt=FT,
                      dropout=pt, activation='elu', F1=F1, D=2, kernLength=KE, dropout_eeg=pe)
    lossHistory = LossHistory()
    earlyStoppingAtMinLoss = EarlyStoppingAtMinLoss(patience=100)
    customLearningRateScheduler = CustomLearningRateScheduler()

    model.compile(
        optimizer='adam',
        loss=categorical_crossentropy,
        metrics=['accuracy']
    )
    # train model
    model.fit(
        Xtrain, ytrain,
        batch_size=64, epochs=100,
        validation_data=(Xtest, ytest),
        # validation_split=0.1, shuffle=True,
        callbacks=[lossHistory, customLearningRateScheduler]
    )
    lossHistory.loss_plot('epoch')
def BCIC_IV_EGGTCNETwithFusion():
    Xtrain, Xtest, ytrain, ytest = BCICompitionIV().loadData(path='../../data/data.mat', format='3D')
    model = TCNet_fusion(nb_classes=4,
                         Chans=22,
                         Samples=1125,
                         layers=2,
                         kernel_s=4,
                         filt=12,
                         dropout=0.3,
                         activation='elu',
                         F1=18,
                         D=3,
                         kernLength=20,
                         )
    plot_model(model=model, to_file='./EGGTCNETwithFusion.png', show_shapes=True)
    lossHistory = LossHistory()
    earlyStoppingAtMinLoss = EarlyStoppingAtMinLoss(patience=100)
    customLearningRateScheduler = CustomLearningRateScheduler()

    model.compile(
        optimizer='adam',
        loss=categorical_crossentropy,
        metrics=['accuracy']
    )
    # train model
    model.fit(
        Xtrain, ytrain,
        batch_size=64, epochs=100,
        validation_data=(Xtest, ytest),
        # validation_split=0.1, shuffle=True,
        callbacks=[lossHistory, customLearningRateScheduler]
    )
    lossHistory.loss_plot('epoch')

if __name__ == '__main__':
    BCIC_IV_EGGTCNETwithFusion()

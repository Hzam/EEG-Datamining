from tensorflow.keras.losses import categorical_crossentropy
from models_of_Felix import ACRNN
from callbacks_of_Felix import LossHistory, EarlyStoppingAtMinLoss, CustomLearningRateScheduler
from datasets import MNIST, KaggleEDFMASD, data_20211114_ZhangHao

if __name__ == '__main__':
    # prepare dataset
    # train_images, train_labels, test_images, test_labels = MNIST().loadData()
    # Xtrain, Xtest, ytrain, ytest = KaggleEDFMASD().loadData(path='../../EEG Data/Disposed/withoutFeatureExtration/subject_1.pkl', format='3D')
    Xtrain, Xtest, ytrain, ytest = data_20211114_ZhangHao().loadData(classification=4)
    # callbacks module
    lossHistory = LossHistory()
    earlyStoppingAtMinLoss = EarlyStoppingAtMinLoss(patience=10)
    customLearningRateScheduler = CustomLearningRateScheduler()

    # build model
    ACRNN_object = ACRNN(input_shape=Xtrain.shape[1:], class_num=ytrain.shape[1])
    model = ACRNN_object.build_model()
    model.compile(
        optimizer='adam',
        loss=categorical_crossentropy,
        metrics=['accuracy']
    )

    # train model
    model.fit(
        Xtrain, ytrain,
        batch_size=32, epochs=20,
        validation_data=(Xtest, ytest),
        #validation_split=0.1, shuffle=True,
        callbacks=[lossHistory, customLearningRateScheduler]
    )

    # evaluate and visualization
    score = model.evaluate(Xtest, ytest, verbose=1)
    print('test loss:', score[0])
    print('test accuracy:', score[1])
    lossHistory.loss_plot('epoch')
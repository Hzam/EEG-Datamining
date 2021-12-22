from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,Callback


class ThresholdCallback(Callback):
    def __init__(self, threshold):
        super(ThresholdCallback, self).__init__()
        self.threshold = threshold
        #self.log_name  = log_name
    
    def on_epoch_end(self, epoch, logs=None): 
        val_loss = logs['val_loss']
        if val_loss <= self.threshold:
            self.model.stop_training = True

def get_callbacks(callbacks_names,call_args):
    callbacks = dict()
    for i,j in enumerate(callbacks_names):#range(len(callbacks_names)):
        if callbacks_names[j]=='early_stopping':
            callb = EarlyStopping(monitor=call_args[i]['monitor'], patience=call_args[i]['patience'], min_delta=call_args[i]['min_delta'],
                                  mode=call_args[i]['mode'],verbose = call_args[i]['verbose'],restore_best_weights=call_args[i]['restore_best_weights'])
        elif callbacks_names[j]=='checkpoint':
            callb = ModelCheckpoint(filepath=call_args[i]['filepath'],save_format=call_args[i]['save_format'], monitor=call_args[i]['monitor'],
                                    verbose=call_args[i]['verbose'],save_weights_only=call_args[i]['save_weights_only'],save_best_only=call_args[i]['save_best_only'])
        elif callbacks_names[j]=='Threshold':
            callb = ThresholdCallback(threshold=call_args[i]['threshold'])
        callbacks[j]=callb
    return callbacks

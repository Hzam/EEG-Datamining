from braindecode.datasets.moabb import MOABBDataset
from braindecode.datautil.preprocess import (exponential_moving_standardize, preprocess, Preprocessor, scale)
from braindecode.datautil.windowers import create_windows_from_events
import numpy as np
import tensorflow as tf


def name_to_numclasses(class_names):
    classes = []
    for i in class_names:
        if i=='left hand':
            classes.append(0)
        elif i=='right hand':
            classes.append(1)
        elif i=='feet':
            classes.append(2)
        elif i=='tongue':
            classes.append(3)
    return classes

def get_classes(X,y, class_names):
    classes = name_to_numclasses(class_names)
    X_c = []
    y_c = []
    for i in classes:
        X_c.append(X[y==i,:,:,:])
        y_c.append(y[y==i])
    X_c = np.concatenate(X_c,axis=0)
    y_c = np.concatenate(y_c,axis=0)
    return X_c, y_c


def get_epochs(dset):
    y = []
    X = []
    for i in range(len(dset)):
        y.append(dset[i][1])
        X.append(np.expand_dims(dset[i][0],axis=[0,3]))
    
    y = np.asarray(y)
    X = np.concatenate(X,axis=0)
    return X,y



def load_dataset(dataset_name="BNCI2014001", subject_id=1, low_cut_hz = 4., high_cut_hz = 38., trial_start_offset_seconds = -0.5,
                 trial_stop_offset_seconds=0,Channels=None,Classes = None,split=True):

    dataset = MOABBDataset(dataset_name=dataset_name, subject_ids=[subject_id])

    # Parameters for exponential moving standardization
    factor_new = 1e-3
    init_block_size = 1000

    if Channels == None:
        preprocessors = [
            Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
            Preprocessor(scale, factor=1e6, apply_on_array=True),  # Convert from V to uV
            Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
            Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                        factor_new=factor_new, init_block_size=init_block_size)
        ]
    else:
        preprocessors = [
            Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
            Preprocessor('pick_channels',ch_names=Channels),
            Preprocessor(scale, factor=1e6, apply_on_array=True),  # Convert from V to uV
            Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
            Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                        factor_new=factor_new, init_block_size=init_block_size)
        ]

    # Transform the data
    preprocess(dataset, preprocessors)

    
    # Extract sampling frequency, check that they are same in all datasets
    sfreq = dataset.datasets[0].raw.info['sfreq']
    assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
    # Calculate the trial start offset in samples.
    trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

    # Create windows using braindecode function for this. It needs parameters to define how
    # trials should be used.
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=int(trial_stop_offset_seconds*sfreq),
        preload=True,
    )

    splitted = windows_dataset.split('session')
    if split:
        sess1 = 'session_T'
        sess2 = 'session_E'
    else:
        sess1 = 'session_0'
        sess2 = 'session_0'
    
    train_set = splitted[sess1]
    valid_set = splitted[sess2]

    X_train,y_train = get_epochs(train_set)
    X_valid,y_valid = get_epochs(valid_set)

    #
    y_train = y_train.reshape(len(y_train), 1)
    y_valid = y_valid.reshape(len(y_valid), 1)
    y_train = tf.keras.utils.to_categorical(y_train, 4).astype('int8')
    y_valid = tf.keras.utils.to_categorical(y_valid, 4).astype('int8')

    if Classes is not None:
        X_train,y_train = get_classes(X_train,y_train, Classes)
        X_valid,y_valid = get_classes(X_valid,y_valid, Classes)
        #
        y_train = y_train.reshape(len(y_train), 1)
        y_valid = y_valid.reshape(len(y_valid), 1)
        y_train = tf.keras.utils.to_categorical(y_train, 4).astype('int8')
        y_valid = tf.keras.utils.to_categorical(y_valid, 4).astype('int8')
    return X_train,y_train,X_valid,y_valid,sfreq

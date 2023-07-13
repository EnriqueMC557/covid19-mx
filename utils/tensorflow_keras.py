import os
import gc
import random

import numpy as np
import tensorflow as tf


def set_seed(RANDOM_SEED=557):
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    # tf.keras.utils.set_random_seed(RANDOM_SEED)


def reset_session():
    tf.keras.backend.clear_session()
    gc.collect()
    # set_seed()


def validate_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print('WARNING: None GPUs detected')


def set_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print('GPU Success!')
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        print('WARNING: None GPUs detected')


def reset_weights(model):
    for layer in model.layers:
        if layer.trainable_weights:
            weights = layer.get_weights()
            new_weights = [np.random.uniform(low=-0.05, high=0.05, size=w.shape) for w in weights]
            layer.set_weights(new_weights)

# build the model
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Conv1D, GlobalAveragePooling1D, Dropout, Dense, BatchNormalization, ReLU
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, auc, precision_recall_curve
from tensorflow.keras.layers import Input, Permute
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

##### Self.Written Packages #####
from utils.ALSTM import AttentionLSTM, AttentionLSTMCell
#################################

# written by kl3606
def build_lstm_fcn(input_shape, num_classes, num_cells):
    inputs = tf.keras.Input(shape=input_shape) 
    # expected inputs' shape: [batch_size, 1(univariate), T(original number of time step)] 
    # the inputs is after dimension shuffle

    # CONV Block
    x1= Permute((2, 1))(inputs) # input's shape change back to [batch _size, T, D = 1(univariate)]
    x1 = Conv1D(128, kernel_size=8, padding='same', kernel_initializer='he_normal')(inputs)
    x1 = BatchNormalization(momentum=0.99, epsilon=0.001)(x1)
    x1 = ReLU()(x1)

    x1 = Conv1D(256, kernel_size=5, padding='same', kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization(momentum=0.99, epsilon=0.001)(x1)
    x1 = ReLU()(x1)

    x1 = Conv1D(128, kernel_size=3, padding='same', kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization(momentum=0.99, epsilon=0.001)(x1)
    x1 = ReLU()(x1)

    x1 = GlobalAveragePooling1D()(x1)

    # LSTM Block
    # expected inputs' shape: [batch_size, 1(univariate), T(original number of time step)] 
    # the inputs is after dimension shuffle
    x2 = LSTM(num_cells)(inputs)
    x2 = Dropout(0.8)(x2)

    # Concatenate CONV and LSTM
    x = tf.keras.layers.concatenate([x1, x2]) 
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs,name = "LSTM")
    
    return model

def build_alstm_fcn(input_shape, num_classes, num_cells):
    inputs = tf.keras.Input(shape=input_shape)
    # expected inputs' shape: [batch_size, 1(univariate), T(original number of time step)] 
    # the inputs is after dimension shuffle

    # CONV Block
    x1= Permute((2, 1))(inputs) # input's shape change back to [batch _size, T, D = 1(univariate)]
    x1 = Conv1D(128, kernel_size=8, padding='same', kernel_initializer='he_normal')(inputs)
    x1 = BatchNormalization(momentum=0.99, epsilon=0.001)(x1)
    x1 = ReLU()(x1)

    x1 = Conv1D(256, kernel_size=5, padding='same', kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization(momentum=0.99, epsilon=0.001)(x1)
    x1 = ReLU()(x1)

    x1 = Conv1D(128, kernel_size=3, padding='same', kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization(momentum=0.99, epsilon=0.001)(x1)
    x1 = ReLU()(x1)

    x1 = GlobalAveragePooling1D()(x1)

    # LSTM Block
    # expected inputs' shape: [batch_size, 1(univariate), T(original number of time step)] 
    # the inputs is after dimension shuffle
    x2 = AttentionLSTM(num_cells)(inputs)
    x2 = Dropout(0.8)(x2)

    # Concatenate CONV and LSTM
    x = tf.keras.layers.concatenate([x1, x2])
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs,name = "ALSTM")
    
    return model

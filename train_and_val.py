# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization, Attention, Dropout, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from keras.models import Model
from keras.layers import Input, GRU, Dense, Bidirectional, Add, TimeDistributed, Flatten

from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.optimizers import Adam
from time import time

from collections import deque
import joblib
import glob
import os

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Activation, GlobalAveragePooling1D, Reshape, multiply
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, RobustScaler
from tensorflow.keras.utils import to_categorical
import pydotplus
from keras.utils import plot_model

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from tqdm.auto import tqdm


train_data='dataset3'
test_data='test1'
# Create sequences for non-long datasets
def create_sequences(X, y, seq_length):
    Xs, ys = [], []
    for j in range(int(len(X)/seq_length)):
        i=j*seq_length
        Xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length-1])
    return np.array(Xs), np.array(ys)

def load_data(seq_length):
    global scaler, encoder, X_train, y_train, X_val, y_val
# Load all data files and concatenate them into one DataFrame
    all_files = glob.glob(f"{train_data}/*.txt")
    li = []

    for filename in tqdm(all_files):
        # Preliminary check to decide the separator
        # print(filename)
        with open(filename, 'r') as temp_f:
            lines = temp_f.readlines()
            separator = "\t" if len(lines) > seq_length+1 else " "
        
        # Read the file with the determined separator
        df = pd.read_csv(filename, index_col=None, header=None, sep=separator)
        
        # Modify the dataset for long sequences
        new_dfs = []
        for i in range(len(df) - seq_length-1): 
            temp_df = pd.DataFrame(df.iloc[i:i + seq_length, 2:].values, columns=[2, 3, 4])  # Correct column labels for data
            label = df.iloc[i + seq_length, 0]  # The label is the first column of the sequence row
            temp_df['gesture'] = label
            new_dfs.append(temp_df)
        df = pd.concat(new_dfs, axis=0, ignore_index=True)
        # print(df)

        li.append(df)

    data = pd.concat(li, axis=0, ignore_index=True)

    # Scale the values
    scaler = RobustScaler()
    data.iloc[:, :3] = scaler.fit_transform(data.iloc[:, :3])

    # Label encoding
    encoder = LabelEncoder()
    data.iloc[:, 3] = encoder.fit_transform(data.iloc[:, 3])

    # Split features and labels
    X = data.iloc[:, :3].values
    y = data.iloc[:, 3].values

    # One-hot encode labels
    y = to_categorical(y)

    X, y = create_sequences(X, y, seq_length)

    # Split the data into training, validation, and testing
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
    print('X: ', X.shape)
    print('y: ', y.shape)
    print('X_train: ', X_train.shape)
    print('y_train: ', y_train.shape)
    print('X_val: ', X_val.shape)
    print('y_val: ', y_val.shape)
#=======================================================================
def senet(input_shape=(30, 3), num_classes=7):
    model_name= "SENet"
    ip = Input(shape=input_shape)
    
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(ip)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(256, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = GlobalAveragePooling1D()(y)
    
    out = Dense(num_classes, activation='softmax')(y)
    model = Model(ip, out)
    
    return model, model_name

def generate_lstm_model():
    model_name = "LSTM"
    input_shape = (30, 3)  # Adjust as needed

    # Input layer
    inputs = Input(shape=input_shape)

    # First LSTM layer
    lstm_out_1 = LSTM(60, return_sequences=True)(inputs)
    lstm_out_1 = BatchNormalization()(lstm_out_1)

    # Second LSTM layer
    lstm_out_2 = LSTM(60, return_sequences=True)(lstm_out_1)
    lstm_out_2 = BatchNormalization()(lstm_out_2)

    # Flatten and output layer
    flattened = Flatten()(lstm_out_2)
    output = Dense(y.shape[1], activation='softmax')(flattened)  # Adjust num_classes

    model = Model(inputs=inputs, outputs=output)
    return model, model_name
    
def generate_res_bi_lstm_model():
    model_name="residual bidirectional LSTM"
    input_shape = (30, 3)  # Adjust as needed

    # Input layer
    inputs = Input(shape=input_shape)

    # First Bi-LSTM layer with residual connection
    lstm_out = Bidirectional(LSTM(60, return_sequences=True))(inputs)
    lstm_out = BatchNormalization()(lstm_out)
    flattened_lstm_1 = TimeDistributed(Dense(3))(lstm_out)
    residual_1 = Dense(60, activation='relu')(inputs)
    residual_1 = TimeDistributed(Dense(3))(residual_1)
    added_1 = Add()([flattened_lstm_1, residual_1])

    # Second Bi-LSTM layer with residual connection
    lstm_out_2 = Bidirectional(LSTM(60, return_sequences=True))(added_1)
    lstm_out_2 = BatchNormalization()(lstm_out_2)
    flattened_lstm_2 = TimeDistributed(Dense(3))(lstm_out_2)
    residual_2 = Dense(60, activation='relu')(added_1)
    residual_2 = TimeDistributed(Dense(3))(residual_2)
    added_2 = Add()([flattened_lstm_2, residual_2])

    # Flatten and output layer
    flattened = Flatten()(added_2)
    output = Dense(y.shape[1], activation='softmax')(flattened)  # Adjust num_classes

    model = Model(inputs=inputs, outputs=output)
    return model, model_name
    

# single layer residual bidirectional GRU
def generate_model():
    model_name="single layer residual bidirectional GRU"
    input_shape = (30, 3)  # 30 time steps, 3 features

    # Input layer
    inputs = Input(shape=input_shape)
    
    # Bi-GRU layer
    gru_out = Bidirectional(GRU(60, return_sequences=True))(inputs)

    # Flatten the output for the main path
    flattened_gru = TimeDistributed(Dense(3))(gru_out)  # Adjust the Dense units to match the residual path

    # Residual connection
    residual = Dense(64, activation='relu')(inputs)  # Adjust to align with the GRU output dimension
    residual = TimeDistributed(Dense(3))(residual)   # Make it match the time-distributed nature of the GRU output

    # Adding the residual
    added = Add()([flattened_gru, residual])

    # Flatten before final Dense layers
    flattened = Flatten()(added)


    # Output layer
    output = Dense(y.shape[1], activation='softmax')(flattened)
    # Build the model
    model = Model(inputs=inputs, outputs=output)
    return model, model_name



#squeeze excitation residual bidirectional GRU with batch normalization
def generate_model5():
    model_name="squeeze excitation residual bidirectional GRU with batch normalization"
    input_shape = (30, 3)  # 30 time steps, 3 features

    # Input layer
    inputs = Input(shape=input_shape)

    # First Bi-GRU layer
    gru_out = Bidirectional(GRU(60, return_sequences=True))(inputs)
    gru_out = BatchNormalization()(gru_out)
    gru_out = squeeze_excite_block(gru_out)  # Squeeze-and-Excite block
    flattened_gru_1 = TimeDistributed(Dense(3))(gru_out)
    residual_1 = Dense(60, activation='relu')(inputs)
    residual_1 = TimeDistributed(Dense(3))(residual_1)
    added_1 = Add()([flattened_gru_1, residual_1])

    # Second Bi-GRU layer
    gru_out_2 = Bidirectional(GRU(60, return_sequences=True))(added_1)
    gru_out_2 = BatchNormalization()(gru_out_2)
    gru_out_2 = squeeze_excite_block(gru_out_2)  # Squeeze-and-Excite block
    flattened_gru_2 = TimeDistributed(Dense(3))(gru_out_2)
    residual_2 = Dense(60, activation='relu')(added_1)
    residual_2 = TimeDistributed(Dense(3))(residual_2)
    added_2 = Add()([flattened_gru_2, residual_2])

    # Flatten before final Dense layers
    flattened = Flatten()(added_2)

    # Output layer
    output = Dense(y.shape[1], activation='softmax')(flattened)

    # Build the model
    model = Model(inputs=inputs, outputs=output)
    return model, model_name

#residual bidirectional GRU with batch normalization
def generate_model7():
    model_name="residual bidirectional GRU with batch normalization"
    input_shape = (30, 3)  # 30 time steps, 3 features

    # Input layer
    inputs = Input(shape=input_shape)

    # First Bi-GRU layer
    gru_out = Bidirectional(GRU(60, return_sequences=True))(inputs)
    gru_out = BatchNormalization()(gru_out)
    flattened_gru_1 = TimeDistributed(Dense(3))(gru_out)
    residual_1 = Dense(60, activation='relu')(inputs)
    residual_1 = TimeDistributed(Dense(3))(residual_1)
    added_1 = Add()([flattened_gru_1, residual_1])

    # Second Bi-GRU layer
    gru_out_2 = Bidirectional(GRU(60, return_sequences=True))(added_1)
    gru_out_2 = BatchNormalization()(gru_out_2)
    flattened_gru_2 = TimeDistributed(Dense(4))(gru_out_2)
    residual_2 = Dense(60, activation='relu')(added_1)
    residual_2 = TimeDistributed(Dense(4))(residual_2)
    added_2 = Add()([flattened_gru_2, residual_2])

    # Flatten before final Dense layers
    flattened = Flatten()(added_2)

    # Output layer
    output = Dense(y.shape[1], activation='softmax')(flattened)

    # Build the model
    model = Model(inputs=inputs, outputs=output)
    return model, model_name

#SE Bi-GRU
def generate_model8():
    model_name="SE-Bi-GRU"
    input_shape = (30, 3)  # 30 time steps, 3 features

    # Input layer
    inputs = Input(shape=input_shape)

    # First Bi-GRU layer
    gru_out = Bidirectional(GRU(60, return_sequences=True))(inputs)
    gru_out = BatchNormalization()(gru_out)
    gru_out = squeeze_excite_block(gru_out)  # Squeeze-and-Excite block
    gru_out = TimeDistributed(Dense(3))(gru_out)

    # Second Bi-GRU layer
    gru_out_2 = Bidirectional(GRU(60, return_sequences=True))(gru_out)
    gru_out_2 = BatchNormalization()(gru_out_2)
    gru_out_2 = squeeze_excite_block(gru_out_2)  # Squeeze-and-Excite block
    gru_out_2 = TimeDistributed(Dense(3))(gru_out_2)

    # Flatten before final Dense layers
    flattened = Flatten()(gru_out_2)

    # Output layer
    output = Dense(y.shape[1], activation='softmax')(flattened)

    # Build the model
    model = Model(inputs=inputs, outputs=output)
    return model, model_name


#just SE-GRU
def generate_model10():
    model_name="SE-GRU"
    input_shape = (30, 3)  # 30 time steps, 3 features

    # Input layer
    ip = Input(shape=input_shape)
    
    y = GRU(128, return_sequences=True)(ip)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = GRU(256, return_sequences=True)(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = GRU(6)(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    
    out = Dense(y.shape[1], activation='softmax')(y)
    model = Model(ip, out)
    
    return model, model_name
    
def generate_gru_only_model():
    model_name="GRU"
    input_shape = (30, 3)  # 30 time steps, 3 features

    inputs = Input(shape=input_shape)
    gru_out = GRU(60, return_sequences=True)(inputs)
    gru_out = GRU(60, return_sequences=False)(gru_out)

    flattened = Flatten()(gru_out)
    output = Dense(y.shape[1], activation='softmax')(flattened)

    model = Model(inputs=inputs, outputs=output)
    return model, model_name

def generate_residual_gru_model():
    model_name="res-GRU"
    input_shape = (30, 3)  # 30 time steps, 3 features

    inputs = Input(shape=input_shape)
    gru_out = GRU(60, return_sequences=True)(inputs)
    gru_out = TimeDistributed(Dense(3))(gru_out)

    residual = TimeDistributed(Dense(60, activation='relu'))(inputs)
    residual = TimeDistributed(Dense(3))(residual)
    added = Add()([gru_out, residual])

    gru_out_2 = GRU(60, return_sequences=False)(added)
    flattened = Flatten()(gru_out_2)
    output = Dense(y.shape[1], activation='softmax')(flattened)

    model = Model(inputs=inputs, outputs=output)
    return model, model_name


def generate_bi_gru_only_model():
    model_name="bi-GRU"
    input_shape = (30, 3)  # 30 time steps, 3 features

    inputs = Input(shape=input_shape)
    gru_out = Bidirectional(GRU(60, return_sequences=True))(inputs)
    gru_out = Bidirectional(GRU(60, return_sequences=False))(gru_out)

    flattened = Flatten()(gru_out)
    output = Dense(y.shape[1], activation='softmax')(flattened)

    model = Model(inputs=inputs, outputs=output)
    return model, model_name


def squeeze_excite_block(input):
    ''' Create a squeeze-excite block '''
    filters = input.shape[-1]
    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 20, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se

# Custom Transformer block
def transformer_block(input, embed_dim, num_heads, ff_dim, rate=0.1):
    # Ensure the input dimensionality is compatible with the MultiHeadAttention layer
    input_processed = Dense(embed_dim)(input)
    
    attention_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(input_processed, input_processed)
    attention_output = Dropout(rate)(attention_output)
    out1 = LayerNormalization(epsilon=1e-6)(input_processed + attention_output)

    ffn_output = Dense(ff_dim, activation="relu")(out1)
    ffn_output = Dense(embed_dim)(ffn_output)
    ffn_output = Dropout(rate)(ffn_output)
    return LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = Add()([x, inputs])
    
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return Add()([x, res])

def generate_transformer_model():
    model_name = "Transformer"
    input_shape = (30, 3)  # Adjust as needed
    embed_dim = 64
    num_heads = 3
    ff_dim = 128
    dropout_rate = 0.1

    inputs = Input(shape=input_shape)
    x = transformer_block(inputs, embed_dim, num_heads, ff_dim, rate=dropout_rate)
    x = transformer_block(x, embed_dim, num_heads, ff_dim, rate=dropout_rate)

    x = Flatten()(x)
    output = Dense(y.shape[1], activation='softmax')(x)  # Adjust num_classes

    model = Model(inputs=inputs, outputs=output)
    return model, model_name
def generate_transformer_res_bi_gru_model(input_shape=(30,3), num_classes=6):
    model_name="transformer-res-bi-gru"
    inputs = Input(shape=input_shape)

    # Residual Bi-GRU layer
    gru_out = Bidirectional(GRU(30, return_sequences=True))(inputs)
    gru_out = BatchNormalization()(gru_out)
    gru_out_processed = TimeDistributed(Dense(30))(gru_out)
    residual = TimeDistributed(Dense(30))(inputs)
    gru_out = Add()([gru_out_processed, residual])

    # Transformer layer
    transformer_out = transformer_block(gru_out, embed_dim=15, num_heads=2, ff_dim=30)

    # Flatten and Dense layers for classification
    flattened = Flatten()(transformer_out)
    output = Dense(num_classes, activation='softmax')(flattened)

    model = Model(inputs=inputs, outputs=output)
    return model, model_name

def generate_MULTISTAGE_SE_transformer_bi_gru_model(input_shape=(30, 3), num_classes=6, num_stages=3):
    model_name = f"transformer-multi-stage-SE-res-bi-gru-{num_stages}"
    inputs = Input(shape=input_shape)

    # Initial transformation if necessary
    x = inputs

    # Multiple stages of Bi-GRU, SE, and Transformer
    for i in range(num_stages):
        # Bi-GRU layer
        gru_out = Bidirectional(GRU(15, return_sequences=True))(x)
        gru_out = BatchNormalization()(gru_out)

        # Squeeze-and-Excite block
        se_out = squeeze_excite_block(gru_out)

        # Transformer layer
        transformer_out = transformer_block(se_out, embed_dim=30, num_heads=2, ff_dim=60)

        # Link to the next stage
        x = LayerNormalization(name=f'layer_norm_{i+1}')(transformer_out)

    # Flatten and Dense layers for classification
    flattened = Flatten(name='flatten')(x)
    output = Dense(num_classes, activation='softmax', name='output')(flattened)

    model = Model(inputs=inputs, outputs=output)
    return model, model_name

def generate_transformer_SE_model(input_shape=(30,3), num_classes=6):
    model_name="transformer-SE"
    inputs = Input(shape=input_shape)

    # Conv1D layers
    x = Conv1D(60, kernel_size=3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = squeeze_excite_block(x)

    # Transformer layer
    transformer_out = transformer_block(x, embed_dim=30, num_heads=2, ff_dim=60)

    # Flatten and Dense layers for classification
    flattened = Flatten()(transformer_out)
    output = Dense(num_classes, activation='softmax')(flattened)

    model = Model(inputs=inputs, outputs=output)
    return model, model_name

def generate_transformer_bi_gru_cnn_model2(input_shape=(30,3), num_classes=6):
    model_name="transformer-SE-res-bi-gru2"
    inputs = Input(shape=input_shape)

    # Conv1D layers
    x = Conv1D(60, kernel_size=3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = squeeze_excite_block(x)

    # Residual Bi-GRU layer
    gru_out = Bidirectional(GRU(15, return_sequences=True))(x)
    gru_out = BatchNormalization()(gru_out)
    gru_out_processed = TimeDistributed(Dense(15))(gru_out)
    residual = TimeDistributed(Dense(15))(x)
    gru_out = Add()([gru_out_processed, residual])

    # Transformer layer
    transformer_out = transformer_block(gru_out, embed_dim=30, num_heads=2, ff_dim=60)

    # Flatten and Dense layers for classification
    flattened = Flatten()(transformer_out)
    output = Dense(num_classes, activation='softmax')(flattened)

    model = Model(inputs=inputs, outputs=output)
    return model, model_name
    

def generate_transformer_bi_gru_cnn_model4(input_shape=(30,3), num_classes=7):
    model_name="transformer-SE-res-bi-gru4"
    inputs = Input(shape=input_shape)

    # Conv1D layers
    x = Conv1D(60, kernel_size=3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = squeeze_excite_block(x)

    # Transformer layer
    transformer_out = transformer_block(x, embed_dim=30, num_heads=2, ff_dim=60)

    # Residual Bi-GRU layer
    gru_out = Bidirectional(GRU(15, return_sequences=True))(transformer_out)
    gru_out = BatchNormalization()(gru_out)
    gru_out_processed = TimeDistributed(Dense(15))(gru_out)
    residual = TimeDistributed(Dense(15))(transformer_out)
    gru_out = Add()([gru_out_processed, residual])

    se_out = squeeze_excite_block(gru_out)

    # Flatten and Dense layers for classification
    flattened = Flatten()(se_out)
    output = Dense(num_classes, activation='softmax')(flattened)

    model = Model(inputs=inputs, outputs=output)
    return model, model_name

def generate_transformer_bi_gru_cnn_model4plus(input_shape=(30,3), num_classes=7):
    model_name="transformer-SE-res-bi-gru4plus"
    inputs = Input(shape=input_shape)

    # Conv1D layers
    x = Conv1D(128, kernel_size=3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = squeeze_excite_block(x)

    # Transformer layer
    transformer_out = transformer_block(x, embed_dim=64, num_heads=3, ff_dim=128)

    # Residual Bi-GRU layer
    gru_out = Bidirectional(GRU(32, return_sequences=True))(transformer_out)
    gru_out = BatchNormalization()(gru_out)
    gru_out_processed = TimeDistributed(Dense(32))(gru_out)
    residual = TimeDistributed(Dense(32))(transformer_out)
    gru_out = Add()([gru_out_processed, residual])

    se_out = squeeze_excite_block(gru_out)

    # Flatten and Dense layers for classification
    flattened = Flatten()(se_out)
    output = Dense(num_classes, activation='softmax')(flattened)

    model = Model(inputs=inputs, outputs=output)
    return model, model_name


def generate_transformer_bi_gru_cnn_model4plus2(input_shape=(30,3), num_classes=6):
    model_name="transformer-SE-res-bi-gru4plus2"
    inputs = Input(shape=input_shape)

    # Conv1D layers
    x = Conv1D(128, kernel_size=3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = squeeze_excite_block(x)

    # Transformer layer
    transformer_out = transformer_block(x, embed_dim=64, num_heads=3, ff_dim=128)

    # Residual Bi-GRU layer
    gru_out = Bidirectional(GRU(32, return_sequences=True))(transformer_out)
    
    gru_out = Conv1D(128, kernel_size=3, padding='same')(gru_out)
    gru_out = BatchNormalization()(gru_out)
    gru_out = Activation('relu')(gru_out)
    
    gru_out_processed = TimeDistributed(Dense(32))(gru_out)
    residual = TimeDistributed(Dense(32))(transformer_out)
    gru_out = Add()([gru_out_processed, residual])


    se_out = squeeze_excite_block(gru_out)

    # Flatten and Dense layers for classification
    flattened = Flatten()(se_out)
    output = Dense(num_classes, activation='softmax')(flattened)

    model = Model(inputs=inputs, outputs=output)
    return model, model_name

def generate_transformer_bi_gru_cnn_model4plus3(input_shape=(30,3), num_classes=7):
    model_name="transformer-SE-res-bi-gru4plus3"
    inputs = Input(shape=input_shape)

    # Conv1D layers
    x = Conv1D(512, kernel_size=3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = squeeze_excite_block(x)

    # Transformer layer
    transformer_out = transformer_block(x, embed_dim=256, num_heads=2, ff_dim=512)

    # Residual Bi-GRU layer
    gru_out = Bidirectional(GRU(512, return_sequences=True))(transformer_out)
    gru_out = BatchNormalization()(gru_out)
    gru_out_processed = TimeDistributed(Dense(512))(gru_out)
    residual = TimeDistributed(Dense(512))(transformer_out)
    gru_out = Add()([gru_out_processed, residual])

    se_out = squeeze_excite_block(gru_out)

    # Flatten and Dense layers for classification
    flattened = Flatten()(se_out)
    output = Dense(num_classes, activation='softmax')(flattened)

    model = Model(inputs=inputs, outputs=output)
    return model, model_name


def generate_transformer_bi_gru_cnn_model4plus_attention(input_shape=(30, 3), num_classes=7):
    model_name = "generate_transformer_bi_gru_cnn_model4plus_attention"
    inputs = Input(shape=input_shape)

    # Conv1D layers
    x = Conv1D(128, kernel_size=3, padding='same')(inputs)
    x = LayerNormalization()(x)
    x = Activation('relu')(x)
    x = squeeze_excite_block(x)
    x = Dropout(0.3)(x)

    # Transformer layers
    for _ in range(2):  # Stack multiple transformer layers
        x = transformer_block(x, embed_dim=64, num_heads=3, ff_dim=128)

    # Residual Bi-GRU layer
    gru_out = Bidirectional(GRU(32, return_sequences=True))(x)
    gru_out = LayerNormalization()(gru_out)
    gru_out_processed = TimeDistributed(Dense(32))(gru_out)
    residual = TimeDistributed(Dense(32))(x)
    gru_out = Add()([gru_out_processed, residual])

    se_out = squeeze_excite_block(gru_out)
    se_out = Dropout(0.3)(se_out)

    # Multi-Head Self-Attention
    attention_out = Attention()([se_out, se_out])

    # Flatten and Dense layers for classification
    flattened = Flatten()(attention_out)
    flattened = Dropout(0.3)(flattened)
    output = Dense(num_classes, activation='softmax')(flattened)

    model = Model(inputs=inputs, outputs=output)
    return model, model_name

def generate_transformer_bi_gru_cnn_model4plus_attention2(input_shape=(30, 3), num_classes=7):
    model_name = "generate_transformer_bi_gru_cnn_model4plus_attention2"
    inputs = Input(shape=input_shape)

    # Conv1D layers
    x = Conv1D(80, kernel_size=2, padding='same')(inputs)
    x = LayerNormalization()(x)
    x = Activation('relu')(x)
    x = squeeze_excite_block(x)
    x = Dropout(0.3)(x)

    # Transformer layers
    for _ in range(2):  # Stack multiple transformer layers
        x = transformer_block(x, embed_dim=20, num_heads=3, ff_dim=80)

    # Residual Bi-GRU layer
    gru_out = Bidirectional(GRU(20, return_sequences=True))(x)
    gru_out = LayerNormalization()(gru_out)
    gru_out_processed = TimeDistributed(Dense(20))(gru_out)
    residual = TimeDistributed(Dense(20))(x)
    gru_out = Add()([gru_out_processed, residual])

    se_out = squeeze_excite_block(gru_out)
    se_out = Dropout(0.3)(se_out)

    # Multi-Head Self-Attention
    attention_out = Attention()([se_out, se_out])

    # Flatten and Dense layers for classification
    flattened = Flatten()(attention_out)
    flattened = Dropout(0.3)(flattened)
    output = Dense(num_classes, activation='softmax')(flattened)

    model = Model(inputs=inputs, outputs=output)
    return model, model_name

def generate_transformerenc_bi_gru_cnn_model4plus(input_shape=(30,3), num_classes=6):
    model_name="transformerenc-SE-res-bi-gru4plus"
    inputs = Input(shape=input_shape)

    # Conv1D layers
    x = Conv1D(128, kernel_size=3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = squeeze_excite_block(x)

    # Transformer layer
    transformer_out = transformer_encoder(x, head_size = 64, num_heads = 3, ff_dim = 64, dropout = 0.1)
    

    # Residual Bi-GRU layer
    gru_out = Bidirectional(GRU(32, return_sequences=True))(transformer_out)
    gru_out = BatchNormalization()(gru_out)
    gru_out_processed = TimeDistributed(Dense(32))(gru_out)
    residual = TimeDistributed(Dense(32))(transformer_out)
    gru_out = Add()([gru_out_processed, residual])

    se_out = squeeze_excite_block(gru_out)

    # Flatten and Dense layers for classification
    flattened = Flatten()(se_out)
    output = Dense(num_classes, activation='softmax')(flattened)

    model = Model(inputs=inputs, outputs=output)
    return model, model_name

def generate_transformerenc_bi_gru_cnn_model4plus_attention(input_shape=(30, 3), num_classes=6):
    model_name = "generate_transformerenc_bi_gru_cnn_model4plus_attention"
    inputs = Input(shape=input_shape)

    # Conv1D layers
    x = Conv1D(128, kernel_size=3, padding='same')(inputs)
    x = LayerNormalization()(x)
    x = Activation('relu')(x)
    x = squeeze_excite_block(x)
    x = Dropout(0.3)(x)

    # Transformer layers
    for _ in range(2):  # Stack multiple transformer layers
        x = transformer_out = transformer_encoder(x, head_size = 64, num_heads = 3, ff_dim = 64, dropout = 0.1)

    # Residual Bi-GRU layer
    gru_out = Bidirectional(GRU(32, return_sequences=True))(x)
    gru_out = LayerNormalization()(gru_out)
    gru_out_processed = TimeDistributed(Dense(32))(gru_out)
    residual = TimeDistributed(Dense(32))(x)
    gru_out = Add()([gru_out_processed, residual])

    se_out = squeeze_excite_block(gru_out)
    se_out = Dropout(0.3)(se_out)

    # Multi-Head Self-Attention
    attention_out = Attention()([se_out, se_out])

    # Flatten and Dense layers for classification
    flattened = Flatten()(attention_out)
    flattened = Dropout(0.3)(flattened)
    output = Dense(num_classes, activation='softmax')(flattened)

    model = Model(inputs=inputs, outputs=output)
    return model, model_name



def generate_transformer_bi_gru_cnn_model5(input_shape=(30,3), num_classes=6):
    model_name="transformer-SE-res-bi-gru5"
    inputs = Input(shape=input_shape)

    # Conv1D layers
    x = Conv1D(30, kernel_size=3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = squeeze_excite_block(x)

    # Transformer layer
    transformer_out = transformer_block(x, embed_dim=30, num_heads=2, ff_dim=60)

    # Residual Bi-GRU layer
    gru_out = Bidirectional(GRU(15, return_sequences=True))(transformer_out)
    gru_out = BatchNormalization()(gru_out)
    gru_out_processed = TimeDistributed(Dense(15))(gru_out)
    residual = TimeDistributed(Dense(15))(transformer_out)
    gru_out = Add()([gru_out_processed, residual])

    # Flatten and Dense layers for classification
    flattened = Flatten()(gru_out)
    output = Dense(num_classes, activation='softmax')(flattened)

    model = Model(inputs=inputs, outputs=output)
    return model, model_name

def generate_transformer_bi_gru_model2(input_shape=(30,3), num_classes=6):
    model_name="transformer-res-bi-gru2"
    inputs = Input(shape=input_shape)


    # Transformer layer
    transformer_out = transformer_block(inputs, embed_dim=30, num_heads=2, ff_dim=60)

    # Residual Bi-GRU layer
    gru_out = Bidirectional(GRU(15, return_sequences=True))(transformer_out)
    gru_out = BatchNormalization()(gru_out)
    gru_out_processed = TimeDistributed(Dense(15))(gru_out)
    residual = TimeDistributed(Dense(15))(transformer_out)
    gru_out = Add()([gru_out_processed, residual])

    # Flatten and Dense layers for classification
    flattened = Flatten()(gru_out)
    output = Dense(num_classes, activation='softmax')(flattened)

    model = Model(inputs=inputs, outputs=output)
    return model, model_name

def generate_transformer_bi_gru_cnn_model6(input_shape=(30,3), num_classes=7):
    model_name = "transformer-SE-res-bi-gru6"
    inputs = Input(shape=input_shape)

    # Conv1D layers with increased filters and kernel size
    x = Conv1D(256, kernel_size=5, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(128, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = squeeze_excite_block(x)

    # Transformer layer with increased embed_dim and num_heads
    transformer_out = transformer_block(x, embed_dim=128, num_heads=4, ff_dim=256)

    # Residual Bi-GRU layer with increased units
    gru_out = Bidirectional(GRU(64, return_sequences=True))(transformer_out)
    gru_out = BatchNormalization()(gru_out)
    gru_out_processed = TimeDistributed(Dense(64))(gru_out)
    residual = TimeDistributed(Dense(64))(transformer_out)
    gru_out = Add()([gru_out_processed, residual])

    # Additional SE block
    se_out = squeeze_excite_block(gru_out)

    # Global pooling instead of Flatten
    pooled = GlobalAveragePooling1D()(se_out)

    # Intermediate dense layer
    dense_out = Dense(64, activation='relu')(pooled)
    dense_out = Dropout(0.2)(dense_out)

    # Output layer
    output = Dense(num_classes, activation='softmax')(dense_out)

    model = Model(inputs=inputs, outputs=output)
    return model, model_name

from tensorflow.keras.layers import MultiHeadAttention

def generate_attention_gru_cnn_model7(input_shape=(30, 3), num_classes=7):
    model_name = "attention-gru-cnn-model7"
    inputs = Input(shape=input_shape)

    # Conv1D layers with fewer filters
    x = Conv1D(64, kernel_size=3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(64, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Attention layer (self-attention or lightweight transformer)
    attention_out = MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
    attention_out = BatchNormalization()(attention_out)
    attention_out = Activation('relu')(attention_out)

    # Bi-GRU layer for sequential modeling
    gru_out = Bidirectional(GRU(64, return_sequences=True))(attention_out)
    gru_out = BatchNormalization()(gru_out)

    # Global pooling to reduce the feature dimensions
    pooled = GlobalAveragePooling1D()(gru_out)

    # Intermediate dense layer
    dense_out = Dense(64, activation='relu')(pooled)
    dense_out = Dropout(0.2)(dense_out)  # Dropout to prevent overfitting

    # Output layer
    output = Dense(num_classes, activation='softmax')(dense_out)

    model = Model(inputs=inputs, outputs=output)
    return model, model_name


def f1score(y_true, y_pred):
    # Convert predictions to one-hot encoding
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.one_hot(y_pred, depth=tf.shape(y_true)[-1])
    
    # Calculate True Positives, False Positives, and False Negatives
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    # Calculate Precision and Recall
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    # Calculate F1 Score
    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def predict_gestures(data, seq_length=30):
    window = deque(maxlen=seq_length)
    predictions = []

    total_length = len(data)
    print("Progress: ", end="")
    for i in range(total_length):
        # Update window with new data
        window.append(data[i])
        if len(window) == seq_length:
            # Prepare the data for prediction
            window_data = np.array(window).reshape(1, seq_length, -1)
            window_data = scaler.transform(window_data[0]).reshape(1, seq_length, -1)
            
            # Make the prediction
            pred = model.predict(window_data, verbose=0)
            pred_label = encoder.inverse_transform([np.argmax(pred)])
            predictions.append(pred_label[0])
        
        # Update the progress bar
        progress = (i + 1) / total_length
        bar_length = 30  # Modify this to change the progress bar length
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(f'\rProgress: [{bar}] {int(progress * 100)}%', end='')
    print()  # Print new line at the end to avoid overwriting the last progress update
    return predictions

def plot_predictions(ground_truths, predictions, encoder, title=None, save_dir=None):
    times = np.arange(len(predictions)) * 0.1  # Assuming a frequency of 10Hz
    plt.figure(figsize=(6, 5))
    
    # Convert ground truths and predictions to numerical format for plotting
    # This assumes that your labels are categorical and can be found in encoder.classes_
    gt_numeric = [np.where(encoder.classes_ == gt)[0][0] for gt in ground_truths]  # Adjusted for alignment
    pred_numeric = [np.where(encoder.classes_ == pred)[0][0] for pred in predictions]
    
    # Plot the actual data
    plt.plot(times, gt_numeric, 'o', label='Ground Truth')  # Use numeric values for plotting
    plt.plot(times, pred_numeric, 'x', label='Predictions')

    # Set the y-ticks to show all labels
    plt.yticks(range(len(encoder.classes_)), encoder.classes_)
    
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Gesture')
    plt.title(title if title else 'Gesture Predictions')
    plt.grid(True)
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{title}.png" if title else "predictions.png")
        plt.savefig(save_path)
    plt.close()

def generate_confusion_matrix(y_true, y_pred, encoder, title=None, save_path=None):
    plt.figure()
    class_labels = encoder.classes_
    matrix = confusion_matrix(y_true, y_pred, labels=class_labels, normalize='true')
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    plt.title(title if title else 'confusion matrix')
    if save_path:
        plt.savefig(save_path)
    plt.close()


def ablation(model, model_name, batch, learning_rate, epochs, seq_length):
    load_data(seq_length)
    global scaler, encoder
    # Creating directory for saving results
    run_number = 1
    while os.path.exists(f"run{run_number}"):
        run_number += 1
    os.makedirs(f"run{run_number}")
    
    # checkpoint = ModelCheckpoint(f"run{run_number}/gesture_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
    # early_stopping = EarlyStopping(monitor='val_accuracy', patience=50, restore_best_weights=True, verbose=1)
    
    checkpoint = ModelCheckpoint(f"run{run_number}/gesture_model.h5", monitor='val_f1score', save_best_only=True, mode='max', verbose=1)
    early_stopping = EarlyStopping(monitor='val_f1score', patience=25, restore_best_weights=True, mode='max', verbose=1)
    
    
    # Model summary
    model.summary()
    
    # Plot the model
    plot_model(model, to_file=f"run{run_number}/model.png", show_shapes=True, show_layer_names=True)
    
    # Configurable parameters
    # learning_rate = 0.0002  # You can change this value
    batch_size = batch        # You can change this value
    # epochs=300
    
    # Compile the model with a specified learning rate
    # model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy', f1score])
    
    # Reduce learning rate when a metric has stopped improving
    # reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.7, patience=5, min_lr=1e-6, verbose=1)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_f1score', factor=0.7, patience=5, min_lr=1e-7, mode='max', verbose=1)
    
    
    # Training the model with specified paremeters
    start_time = time()
    # Training the model with specified batch size and callbacks including ReduceLROnPlateau
    history = model.fit(X_train, y_train, 
                        validation_data=(X_val, y_val), 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        callbacks=[checkpoint, early_stopping, reduce_lr])
    
    end_time = time()
    
    # Calculate and print training duration
    training_duration = end_time - start_time
    print(f"Training Duration: {training_duration:.2f} seconds")
    
    # Add an 'epoch' column to the history DataFrame
    history_df = pd.DataFrame(history.history)
    history_df['epoch'] = history_df.index+1  # Add 'epoch' column with values starting from 1
    
    # Reorder the columns to have 'epoch' as the first column
    history_df = history_df[['epoch'] + [col for col in history_df.columns if col != 'epoch']]
    # Save training history with 'epoch' as the first column
    history_df.to_csv(f"run{run_number}/training_history.csv", index=False)
    
    # Save the scaler and encoder
    joblib.dump(scaler, f"run{run_number}/scaler.pkl")
    joblib.dump(encoder, f"run{run_number}/encoder.pkl")
    
    
    # Setting the font to Times New Roman and bold for all text
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.size'] = 16
    
    
    # Compute and save the normalized confusion matrix
    y_pred = model.predict(X_val)
    # Define class labels
    class_labels = encoder.classes_
    matrix = confusion_matrix(y_val.argmax(axis=1), y_pred.argmax(axis=1), normalize='true')
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    
    # Adding axis labels
    plt.xlabel('Predicted Labels')  # Label for x-axis
    plt.ylabel('Actual Labels')    # Label for y-axis
    
    plt.savefig(f"run{run_number}/confusion_matrix.png")
    plt.close()
    
    # Access the last epoch details
    last_epoch_data = int(history_df.iloc[-1][0])
    
    # Save training parameters and duration
    training_parameters = {
        "Model Name": model_name,
        "Learning Rate": learning_rate,
        "Batch Size": batch_size,
        "Training Duration (seconds)": training_duration,
        "Epochs": last_epoch_data,
        "Dynamic Learning Rate": "monitor='val_accuracy', factor=0.7, patience=5, min_lr=1e-6",
        "seq_length": seq_length
    }
    
    # Write parameters to a text file
    with open(f"run{run_number}/training_parameters.txt", 'w') as file:
        for key, value in training_parameters.items():
            file.write(f"{key}: {value}\n")
    
    
    df = pd.read_csv(f"run{run_number}/training_history.csv")
    
    # Remove leading/trailing whitespaces from column names
    df.columns = df.columns.str.strip()
    
    # Setting the font to Times New Roman and bold for all text
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.size'] = 16
    
    metrics = ['loss', 'accuracy', 'f1score', 'val_loss', 'val_accuracy', 'val_f1score']
    
    for metric in metrics:
        # Adjusting the figure size to match the aspect ratio of the image
        plt.figure(figsize=(6, 5))  # You may need to tweak these numbers to match the exact ratio
    
        # Plot with red color without adding a label to avoid adding it to the legend
        plt.plot(df['epoch'], df[metric], 'r-')
    
        # Find min or max value and corresponding epoch, then update legend text accordingly
        legend_label = None  # Initialize the legend label to None
        if 'loss' in metric:
            min_loss_epoch = df['epoch'][df[metric].idxmin()]
            min_loss = df[metric].min()
            legend_label = f'Min={min_loss:.4f}  Epoch={min_loss_epoch}'
            plt.plot(min_loss_epoch, min_loss, 'bo', label=legend_label)
        else:
            max_value_epoch = df['epoch'][df[metric].idxmax()]
            max_value = df[metric].max()
            legend_label = f'Max={max_value:.4f}  Epoch={max_value_epoch}'
            plt.plot(max_value_epoch, max_value, 'bo', label=legend_label)
    
        # Set axis labels with bold font
        plt.xlabel('epoch', fontweight='bold')
        plt.ylabel(metric, fontweight='bold')
    
        # Make legend compact and with a white background and black border
        if legend_label:  # Only add the legend if there is a label for the blue points
            plt.legend(loc='upper right', bbox_to_anchor=(1, 0.85), frameon=True, facecolor='white', edgecolor='black')
    
        # Set axes background to white and remove gridlines
        ax = plt.gca()
        ax.set_facecolor('white')
        plt.grid(False)
    
        # Adding an outline to the plot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)
    
        # Replace '/' in metric name with '_', keep 'train' and 'val' in the file name
        filename = metric.replace('/', '_')
        plt.tight_layout()  # Adjust the layout of the figure to avoid partially cut images
        plt.savefig(f'run{run_number}/{filename}.png', facecolor='white', edgecolor='black')  # Ensuring white background and black edge
    
        plt.close()
    

    # Setting the font to Times New Roman and bold for all text
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.size'] = 16
    
    
    # Configurable parameters
    model_dir = 'run'+str(run_number)
    # seq_length = 30  # Length of sequence
    
    # Load the model, scaler, and encoder
    model = load_model(f'{model_dir}/gesture_model.h5', custom_objects={'f1score': f1score})
    scaler = joblib.load(f'{model_dir}/scaler.pkl')
    encoder = joblib.load(f'{model_dir}/encoder.pkl')
    
    
    # Prepare data and predictions
    all_files = glob.glob(f"{test_data}/*.txt")
    all_ground_truths = []
    all_predictions = []
    
    test_start=time()
    
    for filename in all_files:
        # Load data, skipping 'idle' and using first column as label
        dataset = np.loadtxt(filename, dtype=str)
        ground_truths = dataset[:, 0]  # Extracting the first column for ground truth labels
        long_data = dataset[:, 2:].astype(float)  # Skip 'idle' and frame number, convert to float
    
        predictions = predict_gestures(long_data,seq_length)
    
        # Trim the first seq_length-1 ground truths because they don't have corresponding predictions
        trimmed_ground_truths = ground_truths[seq_length-1:]
    
        all_ground_truths.extend(trimmed_ground_truths)
        all_predictions.extend(predictions)
    
        title = os.path.basename(filename).split(".")[0]
        plot_predictions(trimmed_ground_truths, predictions,encoder, title=title, save_dir=f'{model_dir}/test_newlong')
    
        # Assuming all ground truths and predictions are collected, plot overall confusion matrix
        generate_confusion_matrix(trimmed_ground_truths, predictions, encoder, title, save_path=f'{model_dir}/test_newlong/confusion_matrix_{title}.png')
    
    
        # Convert labels from strings to a consistent numerical format if necessary
        ground_truth_nums = [encoder.transform([label])[0] for label in trimmed_ground_truths]
        prediction_nums = [encoder.transform([label])[0] for label in predictions]
        
        # If your labels are already in the correct format, you can skip the above conversion
        accuracy = accuracy_score(ground_truth_nums, prediction_nums)
        f1 = f1_score(ground_truth_nums, prediction_nums, average='weighted')
        precision = precision_score(ground_truth_nums, prediction_nums, average='weighted')
        recall = recall_score(ground_truth_nums, prediction_nums, average='weighted')
        
        # Here we specify 'labels' to ensure it includes all classes during report generation
        # This also ensures that labels not present are accounted for in the report
        report = classification_report(ground_truth_nums, prediction_nums, labels=np.arange(len(encoder.classes_)), target_names=encoder.classes_, zero_division=0)
        
        # Assume 'title' is a variable holding the title for each iteration
        metrics_summary = f"Title: {title}\nAccuracy: {accuracy}\nF1 Score: {f1}\nPrecision: {precision}\nRecall: {recall}\n\nClassification Report:\n{report}\n\n---\n\n"
        
        # Save the metrics summary to a text file
        metrics_file = f'{model_dir}/evaluation_metrics.txt'  # Change this to your desired path
        with open(metrics_file, "a") as f:  # Note the "a" here instead of "w"
            f.write(metrics_summary)
        
        # Print out the metrics to the console as well
        print(metrics_summary)
    
    
    test_time=(time() - test_start)/len(all_predictions)
    
    title="overall"
    generate_confusion_matrix(all_ground_truths, all_predictions, encoder, title, save_path=f'{model_dir}/test_newlong/confusion_matrix_{title}.png')
    # Convert labels from strings to a consistent numerical format if necessary
    all_ground_truth_nums = [encoder.transform([label])[0] for label in all_ground_truths]
    all_prediction_nums = [encoder.transform([label])[0] for label in all_predictions]
    
    # If your labels are already in the correct format, you can skip the above conversion
    accuracy = accuracy_score(all_ground_truth_nums, all_prediction_nums)
    f1 = f1_score(all_ground_truth_nums, all_prediction_nums, average='weighted')
    precision = precision_score(all_ground_truth_nums, all_prediction_nums, average='weighted')
    recall = recall_score(all_ground_truth_nums, all_prediction_nums, average='weighted')
    
    # Here we specify 'labels' to ensure it includes all classes during report generation
    # This also ensures that labels not present are accounted for in the report
    report = classification_report(all_ground_truth_nums, all_prediction_nums, labels=np.arange(len(encoder.classes_)), target_names=encoder.classes_, zero_division=0)
    
    # Assume 'title' is a variable holding the title for each iteration
    metrics_summary = f"Title: {title}\naverage inference time: {test_time} \nAccuracy: {accuracy}\nF1 Score: {f1}\nPrecision: {precision}\nRecall: {recall}\n\nClassification Report:\n{report}\n\n---\n\n"
    
    # Save the metrics summary to a text file
    metrics_file = f'{model_dir}/overall_evaluation_metrics.txt'  # Change this to your desired path
    with open(metrics_file, "a") as f:  # Note the "a" here instead of "w"
        f.write(metrics_summary)
    
    # Print out the metrics to the console as well
    print(metrics_summary)


###############################################################################


# model, model_name=generate_transformer_bi_gru_cnn_model4()
# ablation(model, model_name)

# model, model_name=generate_transformer_bi_gru_cnn_model4plus()
# ablation(model, model_name, 2048)

# model, model_name=generate_transformer_bi_gru_cnn_model4plus()
# ablation(model, model_name, 1024)

# model, model_name=generate_transformer_bi_gru_cnn_model4plus()
# ablation(model, model_name, 512)

# model, model_name=generate_transformer_bi_gru_cnn_model4plus()
# ablation(model, model_name, 256)

epoch=300
learning_rate=0.0002


batch=2048

seq_length=30
model, model_name=generate_transformer_bi_gru_cnn_model5(input_shape=(seq_length,3), num_classes=7)
ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)
model, model_name=generate_transformer_bi_gru_cnn_model4plus(input_shape=(seq_length,3), num_classes=7)
ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)
model, model_name=senet(input_shape=(seq_length,3), num_classes=7)
ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)

seq_length=35
model, model_name=generate_transformer_bi_gru_cnn_model5(input_shape=(seq_length,3), num_classes=7)
ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)
model, model_name=generate_transformer_bi_gru_cnn_model4plus(input_shape=(seq_length,3), num_classes=7)
ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)
model, model_name=senet(input_shape=(seq_length,3), num_classes=7)
ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)


# seq_length=30
# model, model_name=generate_transformer_bi_gru_cnn_model4plus(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)
# model, model_name=generate_transformer_bi_gru_cnn_model4plus(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)
# model, model_name=generate_transformer_bi_gru_cnn_model4plus(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)

# seq_length=25
# model, model_name=generate_transformer_bi_gru_cnn_model4plus(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)
# model, model_name=generate_transformer_bi_gru_cnn_model4plus(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)
# model, model_name=generate_transformer_bi_gru_cnn_model4plus(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)

# seq_length=20
# model, model_name=generate_transformer_bi_gru_cnn_model4plus(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)
# model, model_name=generate_transformer_bi_gru_cnn_model4plus(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)
# model, model_name=generate_transformer_bi_gru_cnn_model4plus(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)

batch=1024

# seq_length=40
# model, model_name=generate_transformer_bi_gru_cnn_model4plus(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)
# model, model_name=generate_transformer_bi_gru_cnn_model4plus(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)
# model, model_name=generate_transformer_bi_gru_cnn_model4plus(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)

# seq_length=35
# model, model_name=generate_transformer_bi_gru_cnn_model4plus(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)
# model, model_name=generate_transformer_bi_gru_cnn_model4plus(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)
# model, model_name=generate_transformer_bi_gru_cnn_model4plus(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)

# seq_length=30
# model, model_name=generate_transformer_bi_gru_cnn_model4plus(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)
# model, model_name=generate_transformer_bi_gru_cnn_model4plus(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)
# model, model_name=generate_transformer_bi_gru_cnn_model4plus(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)

# seq_length=25
# model, model_name=generate_transformer_bi_gru_cnn_model4plus(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)
# model, model_name=generate_transformer_bi_gru_cnn_model4plus(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)
# model, model_name=generate_transformer_bi_gru_cnn_model4plus(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)

# seq_length=20
# model, model_name=generate_transformer_bi_gru_cnn_model4plus(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)
# model, model_name=generate_transformer_bi_gru_cnn_model4plus(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)
# model, model_name=generate_transformer_bi_gru_cnn_model4plus(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)


# batch=512

# seq_length=40
# model, model_name=generate_transformer_bi_gru_cnn_model6(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)
# model, model_name=generate_transformer_bi_gru_cnn_model6(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)
# model, model_name=generate_transformer_bi_gru_cnn_model6(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)

# seq_length=35
# model, model_name=generate_transformer_bi_gru_cnn_model6(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)
# model, model_name=generate_transformer_bi_gru_cnn_model6(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)
# model, model_name=generate_transformer_bi_gru_cnn_model6(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)

# seq_length=30
# model, model_name=generate_transformer_bi_gru_cnn_model6(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)
# model, model_name=generate_transformer_bi_gru_cnn_model6(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)
# model, model_name=generate_transformer_bi_gru_cnn_model6(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)


# seq_length=25
# model, model_name=generate_transformer_bi_gru_cnn_model4plus(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)
# model, model_name=generate_transformer_bi_gru_cnn_model4plus(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)
# model, model_name=generate_transformer_bi_gru_cnn_model4plus(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)

# seq_length=20
# model, model_name=generate_transformer_bi_gru_cnn_model4plus(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)
# model, model_name=generate_transformer_bi_gru_cnn_model4plus(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)
# model, model_name=generate_transformer_bi_gru_cnn_model4plus(input_shape=(seq_length,3), num_classes=7)
# ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)

# model, model_name=senet()
# ablation(model, model_name)

# model, model_name=generate_transformer_bi_gru_cnn_model4plus_attention()
# ablation(model, model_name)

# model, model_name=generate_transformer_bi_gru_cnn_model4plus_attention2()
# ablation(model, model_name)



batch=1023

seq_length=30
model, model_name=generate_transformer_bi_gru_cnn_model6(input_shape=(seq_length,3), num_classes=7)
ablation(model, model_name, batch=batch, learning_rate=learning_rate, epochs=epoch, seq_length=seq_length)


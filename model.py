from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import TimeDistributed
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Lambda
from keras.layers import Dropout
from keras.regularizers import l2
from keras.initializers import random_normal
from keras.utils.conv_utils import conv_output_length
from keras.layers import GaussianNoise,concatenate
from keras.layers import merge

'''
This file builds the models
'''

import numpy as np

from keras import backend as K
from keras.models import Model, Sequential
from keras.layers.recurrent import SimpleRNN
from keras.layers import Dense, Activation, Bidirectional, Reshape,Flatten, Lambda, Input,\
    Masking, Convolution1D, BatchNormalization, GRU, Conv1D, RepeatVector, Conv2D,UpSampling1D,MaxPooling1D
from keras.optimizers import SGD, adam
from keras.layers import ZeroPadding1D, Convolution1D, ZeroPadding2D, Convolution2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers import TimeDistributed, Dropout
from keras.layers.merge import add  # , # concatenate BAD FOR COREML
from keras.utils.conv_utils import conv_output_length
from keras.activations import relu

import tensorflow as tf

#QRNN
from qrnn import QRNN,QRNN_Bidirectional
from keras.constraints import maxnorm


def selu(x):
    # from Keras 2.0.6 - does not exist in 2.0.4
    """Scaled Exponential Linear Unit. (Klambauer et al., 2017)
    # Arguments
       x: A tensor or variable to compute the activation function for.
    # References
       - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * K.elu(x, alpha)

def clipped_relu(x):
    return relu(x, max_value=20)

# Define CTC loss
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args

    # hack for load_model
    import tensorflow as tf

    ''' from TF: Input requirements
    1. sequence_length(b) <= time for all b
    2. max(labels.indices(labels.indices[:, 1] == b, 2)) <= sequence_length(b) for all b.
    '''

    # print("CTC lambda inputs / shape")
    # print("y_pred:",y_pred.shape)  # (?, 778, 30)
    # print("labels:",labels.shape)  # (?, 80)
    # print("input_length:",input_length.shape)  # (?, 1)
    # print("label_length:",label_length.shape)  # (?, 1)


    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def ctc(y_true, y_pred):
    return y_pred

######################################
def graves(input_dim=26, rnn_size=512, output_dim=29, std=0.6):
    """ Implementation of Graves 2006 model

    Architecture:
        Gaussian Noise on input
        BiDirectional LSTM

    Reference:
        ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf
    """

    K.set_learning_phase(1)
    input_data = Input(name='the_input', shape=(None, input_dim))
    # x = BatchNormalization(axis=-1)(input_data)

    x = GaussianNoise(std)(input_data)
    x = Bidirectional(LSTM(rnn_size,
                      return_sequences=True,
                      implementation=0))(x)
    y_pred = TimeDistributed(Dense(output_dim, activation='softmax'))(x)

    # Input of labels and other CTC requirements
    labels = Input(name='the_labels', shape=[None,], dtype='int32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred,
                                                                       labels,
                                                                       input_length,
                                                                       label_length])


    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=[loss_out])

    return model


def brsmv1(input_dim=39, rnn_size=512, num_classes=29, input_std_noise=.0, residual=None, num_hiddens=256, num_layers=5,
           dropout=0.2 , input_dropout=False, weight_decay=1e-4, activation='tanh'):
    """ Implementation of brsmv1 model

    Reference:
        http://www.pee.ufrj.br/index.php/pt/producao-academica/dissertacoes-de-mestrado/2017/2016033174-end-to-end-speech-recognition-applied-to-brazilian-portuguese-using-deep-learning/file
    """

    K.set_learning_phase(1)
    input_data = Input(name='the_input', shape=(None, input_dim))
    o=input_data
    if input_std_noise is not None:
        o = GaussianNoise(input_std_noise)(o)
    if residual is not None:
        o = TimeDistributed(Dense(num_hiddens*2,
                                  kernel_regularizer=l2(weight_decay)))(o)
    if input_dropout:
        o = Dropout(dropout)(o)
    for i, _ in enumerate(range(num_layers)):
        new_o = Bidirectional(LSTM(num_hiddens,
                                   return_sequences=True,
                                   kernel_regularizer=l2(weight_decay),
                                   recurrent_regularizer=l2(weight_decay),
                                   dropout=dropout,
                                   recurrent_dropout=dropout,
                                   activation=activation))(o)
        if residual is not None:
            o = merge([new_o,  o], mode=residual)
        else:
            o = new_o
    o = TimeDistributed(Dense(num_classes, activation='softmax'))(o)
    # Input of labels and other CTC requirements
    labels = Input(name='the_labels', shape=[None,], dtype='int32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([o,
                                                                       labels,
                                                                       input_length,
                                                                       label_length])
    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=[loss_out])
    return model
    

def Gru_model(input_dim=39, rnn_size=512, num_classes=29, input_std_noise=.0, residual=None, num_hiddens=256, num_layers=5,dropout=0.2 , input_dropout=False, weight_decay=1e-4, activation='tanh'):
    """ Implementation of Gru teste model

    Reference:
    
    """

    K.set_learning_phase(1)
    input_data = Input(name='the_input', shape=(None, input_dim))
    o=input_data
    if input_std_noise is not None:
        o = GaussianNoise(input_std_noise)(o)
    if residual is not None:
        o = TimeDistributed(Dense(num_hiddens*2,
                                  kernel_regularizer=l2(weight_decay)))(o)
    if input_dropout:
        o = Dropout(dropout)(o)
    for i, _ in enumerate(range(num_layers)):
        new_o = Bidirectional(GRU(num_hiddens,
                                   return_sequences=True,
                                   kernel_regularizer=l2(weight_decay),
                                   recurrent_regularizer=l2(weight_decay),
                                   dropout=dropout,
                                   recurrent_dropout=dropout,
                                   activation=activation))(o)
        if residual is not None:
            o = merge([new_o,  o], mode=residual)
        else:
            o = new_o
    o = TimeDistributed(Dense(num_classes, activation='softmax'))(o)
    # Input of labels and other CTC requirements
    labels = Input(name='the_labels', shape=[None,], dtype='int32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([o,
                                                                       labels,
                                                                       input_length,
                                                                       label_length])
    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=[loss_out])
    return model
    
def qrnn_deepspeech(input_dim=39, rnn_size=512, num_classes=29, input_std_noise=.0, residual=None, num_hiddens=256, num_layers=5,
           dropout=0.2 , input_dropout=False, weight_decay=1e-4, activation='tanh'):
    """ Implementation of Qrnn DeepSpeech

    Reference: 

    """

    K.set_learning_phase(1)
    input_data = Input(name='the_input', shape=(None, input_dim))
    o=input_data
    if input_std_noise is not None:
        o = GaussianNoise(input_std_noise)(o)
    if residual is not None:
        o = TimeDistributed(Dense(num_hiddens*2,
                                  kernel_regularizer=l2(weight_decay)))(o)
    if input_dropout:
        o = Dropout(dropout)(o)
    o = QRNN_Bidirectional(QRNN(num_hiddens,
                                   return_sequences=True,
                                   activation=activation))(o)
    
    o = QRNN_Bidirectional(QRNN(num_hiddens,
                                   return_sequences=True,
                                   activation=activation))(o)
    
    o = QRNN_Bidirectional(QRNN(num_hiddens,
                                   return_sequences=True,
                                   
                                   activation=activation))(o)
    
    o = QRNN_Bidirectional(QRNN(num_hiddens,
                                   return_sequences=True,
                                   activation=activation))(o)
    
    o = QRNN_Bidirectional(QRNN(num_hiddens,
                                   return_sequences=True,
                                   activation=activation))(o)
    
        
    o = TimeDistributed(Dense(num_classes,activation='softmax'))(o)
    # Input of labels and other CTC requirements
    labels = Input(name='the_labels', shape=[None,], dtype='int32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([o,
                                                                       labels,
                                                                       input_length,
                                                                       label_length])
    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=[loss_out])

    return model



def CR2(input_dim=39, conv_size=512, num_classes=29, input_std_noise=.0, residual=None, num_hiddens=256, num_layers=5,
           dropout=0.2 , input_dropout=False, weight_decay=1e-4, activation='tanh'):
    """ Implementation of CR2

    Reference: https://arxiv.org/abs/1611.07174

    """

    K.set_learning_phase(1)
    input_data = Input(name='the_input', shape=(None, input_dim))
    o=input_data
    if input_std_noise is not None:
        o = GaussianNoise(input_std_noise)(o)
        
    if input_dropout:
        o = Dropout(dropout)(o)
    x=o
    for j in range(10):
        x = Conv1D(24, kernel_size = 3,padding='same')(x)
    for j in range(2):    
        x = Conv1D(8,kernel_size = 3,padding='same')(x)
    for j in range(2):    
        x = Conv1D(4,kernel_size = 3,padding='same')(x)
    for j in range(2):    
        x = Conv1D(2,kernel_size = 3,padding='same')(x)
    for _ in range(4) :
        print('shape x:',x.shape)
        x = Bidirectional(SimpleRNN(256,return_sequences=True))(x) #Bidirectional(
    o = TimeDistributed(Dense(256,activation='relu'))(x)        
    o = TimeDistributed(Dense(num_classes,activation='softmax'))(o)
    # Input of labels and other CTC requirements
    labels = Input(name='the_labels', shape=[None,], dtype='int32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([o,
                                                                       labels,
                                                                       input_length,
                                                                       label_length])
    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=[loss_out])

    return model

def DeepSpeech2(input_dim=39, conv_size=512, num_classes=29, input_std_noise=.0, residual=None, num_hiddens=256, num_layers=5,
           dropout=0.2 , input_dropout=False, weight_decay=1e-4, activation='tanh'):
    """ Implementation of CR2

    Reference: http://proceedings.mlr.press/v48/amodei16.html

    """

    K.set_learning_phase(1)
    input_data = Input(name='the_input', shape=(None, input_dim))
    o=input_data
    o =BatchNormalization(axis=-1, name='BN_1')(o)
    o = Conv1D(512, 5, strides=1, activation=clipped_relu, name='Conv1D_1')(o)
    o= Conv1D(512, 5, strides=1, activation=clipped_relu, name='Conv1D_2')(o)
    o= Conv1D(512, 5, strides=2, activation=clipped_relu, name='Conv1D_3')(o)
    
    # Batch Normalization
    o = BatchNormalization(axis=-1, name='BN_2')(o)
    
    # BiRNNs
    o= Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_1'), merge_mode='sum')(o)
    o= Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_2'), merge_mode='sum')(o)
    o = Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_3'), merge_mode='sum')(o)
    o =Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_4'), merge_mode='sum')(o)
    o =Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_5'), merge_mode='sum')(o)
    o =Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_6'), merge_mode='sum')(o)
    o = Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_7'), merge_mode='sum')(o)
    
    # Batch Normalization
    o = BatchNormalization(axis=-1, name='BN_3')(o)
    o = TimeDistributed(Dense(1024,activation=clipped_relu, name='FC1'))(o)        
    o = TimeDistributed(Dense(num_classes,activation='softmax'))(o)
    # Input of labels and other CTC requirements
    labels = Input(name='the_labels', shape=[None,], dtype='int32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([o,
                                                                       labels,
                                                                       input_length,
                                                                       label_length])
    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=[loss_out])

    return model

def DeepSpeech2_Simplified(input_dim=39, conv_size=512, num_classes=29, input_std_noise=.0, residual=None, num_hiddens=256, num_layers=5,
           dropout=0.2 , input_dropout=False, weight_decay=1e-4, activation='tanh'):
    """ Implementation of CR2

    Reference: http://proceedings.mlr.press/v48/amodei16.html

    """

    K.set_learning_phase(1)
    input_data = Input(name='the_input', shape=(None, input_dim))
    o=input_data
    o =BatchNormalization(axis=-1, name='BN_1')(o)
    o = Conv1D(512, 5, strides=1, activation=clipped_relu, name='Conv1D_1')(o)
    o= Conv1D(512, 5, strides=1, activation=clipped_relu, name='Conv1D_2')(o)
    o= Conv1D(512, 5, strides=2, activation=clipped_relu, name='Conv1D_3')(o)
    
    # Batch Normalization
    o = BatchNormalization(axis=-1, name='BN_2')(o)
    
    # BiRNNs
    o= Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_1'), merge_mode='sum')(o)
    o= Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_2'), merge_mode='sum')(o)
    o = Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_3'), merge_mode='sum')(o)
    # Batch Normalization
    o = BatchNormalization(axis=-1, name='BN_3')(o)
    o = TimeDistributed(Dense(1024,activation=clipped_relu, name='FC1'))(o)        
    o = TimeDistributed(Dense(num_classes,activation='softmax'))(o)
    # Input of labels and other CTC requirements
    labels = Input(name='the_labels', shape=[None,], dtype='int32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([o,
                                                                       labels,
                                                                       input_length,
                                                                       label_length])
    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=[loss_out])

    return model

def ConvDilated(input_dim=39, conv_size=512, num_classes=29, input_std_noise=.0, residual=None, num_hiddens=256, num_layers=5,
           dropout=0.2 , input_dropout=False, weight_decay=1e-4, activation='tanh'):
    """ Implementation of ConvDilated DeepSpeech

    Reference: 

    """

    K.set_learning_phase(1)
    input_data = Input(name='the_input', shape=(None, input_dim))
    o=input_data
    if input_std_noise is not None:
        o = GaussianNoise(input_std_noise)(o)
        
    if input_dropout:
        o = Dropout(dropout)(o)
    x=o
    for j in range(6):
        x = Conv1D(512, kernel_size = 7,padding='same')(x)
    for j in range(2):    
        x = Conv1D(256,kernel_size = 5,dilation_rate=3**j,padding='same')(x)
    for j in range(2):    
        x = Conv1D(128,kernel_size = 3,dilation_rate = 3**j,padding='same')(x)
    for j in range(2):    
        x = Conv1D(64,kernel_size = 2,padding='same')(x)
        
            
    """for j in range(2):
            x = Conv1D(int(conv_size//2)*2,name='dilated2-'+str(j), padding='causal',
                       kernel_size = 3, 
                       dilation_rate = 3**j)(x)"""
            
    '''for dilation_rate in range(3):
        for i in range(3):
            x = Conv1D(32*2**(i), 
                       kernel_size = 3, 
                       dilation_rate = dilation_rate+1)(x)'''
    #o = QRNN_Bidirectional(QRNN(num_hiddens, return_sequences=True, activation=activation))(x)
    
    o = TimeDistributed(Dense(512,activation='relu'))(x)
    o = TimeDistributed(Dense(256,activation='relu'))(o)
    

    o = TimeDistributed(Dense(num_classes,activation='softmax'))(o)
    # Input of labels and other CTC requirements
    labels = Input(name='the_labels', shape=[None,], dtype='int32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([o,
                                                                       labels,
                                                                       input_length,
                                                                       label_length])
    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=[loss_out])

    return model


def ConvDilated_HighWay(input_dim=39, conv_size=512, num_classes=29, input_std_noise=.0, residual=None, num_hiddens=256, num_layers=5,
           dropout=0.2 , input_dropout=False, weight_decay=1e-4, activation='tanh'):
    """ Implementation of ConvDilated+HighWay DeepSpeech
    * Increasing Receptive Field for consideration all context:
       - Larger Filters ( usually does not produce optimal results)
       - Adding Layers( good, but produce vanishing gradient problem):
          alternatives for fix vanishing gradient problem:
          + HighWay 
          + ResNets
          + DenseNets
       - Dilated Convolutions ( good, usually used with adding layers)
       
    Reference: 

    """

    K.set_learning_phase(1)
    input_data = Input(name='the_input', shape=(None, input_dim))
    o=input_data
    if input_std_noise is not None:
        o = GaussianNoise(input_std_noise)(o)
        
    if input_dropout:
        o = Dropout(dropout)(o)
    x=o
    for j in range(6):
        x = Conv1D(16, kernel_size = 3,padding='causal')(x)
    for j in range(2):
        x=Lambda(hc)(x)
        #x = Conv1D(8,kernel_size = 3,padding='causal',dilation_rate = 3**j)(x)
    for j in range(2):    
        x = Conv1D(4,kernel_size = 3,padding='causal',dilation_rate = 3**j)(x)
    for j in range(2):    
        x = Conv1D(2,kernel_size = 3,padding='causal')(x)
        
    o = TimeDistributed(Dense(256,activation='relu'))(x)        
    o = TimeDistributed(Dense(num_classes,activation='softmax'))(o)
    # Input of labels and other CTC requirements
    labels = Input(name='the_labels', shape=[None,], dtype='int32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([o,
                                                                       labels,
                                                                       input_length,
                                                                       label_length])
    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=[loss_out])

    return model
    

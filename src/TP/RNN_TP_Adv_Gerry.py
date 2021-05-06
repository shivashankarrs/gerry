'''Recurrent Neural Network Classifier with GRU unit

'''
import os
import pickle
import numpy as np
from sklearn.metrics import f1_score
from tensorflow.keras.layers import Input, Embedding
from tensorflow.keras.layers import Bidirectional, GRU
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
import tensorflow as tf
from math import exp
import keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.python.framework import ops
import pandas as pd 

import sys
sys.path.append('../')


import evaluator
from gerry_eval import bias_attribute_selection_simple, Gerrymandering_eval, bias_attribute_selection
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


tf.random.set_seed(1)
os.environ['PYTHONHASHSEED']=str(1)
np.random.seed(1)

max_len = 150  
import sys
sys.path.append("/home/sssub/intersectional_bias/Multilingual_Fairness_LREC-master/")
    
tp_output_dir = '/home/sssub/intersectional_bias/Multilingual_Fairness_LREC-master/publish/results/TP/adv/'
wt_dir = '/home/sssub/intersectional_bias/Multilingual_Fairness_LREC-master/data/tp/embeddings/'


gerrygroups = 62

tp_keymapping = {
    'gender':{0:"male", 1:"female"}, 
    'country':{0:"UK", 1:"US"}, 
    'supertopic': {0: 'Computer & Accessories',
                1: 'Fashion Accessories',
                2: 'Fitness & Nutrition',
                3: 'Tires',
                4: 'Hotels',
                5: 'Pets'}
}

def reverse_gradient(X, hp_lambda):
    """Flips the sign of the incoming gradient during training."""
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1

    grad_name = "GradientReversal%d" % reverse_gradient.num_calls

    @ops.RegisterGradient(grad_name)
    def _flip_gradients(grad):
        return [tf.negative(grad) * hp_lambda]
    #g = K.get_session().graph
    g = tf.compat.v1.Session().graph
    with g.gradient_override_map({'Identity': grad_name}):
        y = tf.identity(X)

    return y

class GradientReversal(Layer):
    """Layer that flips the sign of gradient during training."""

    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = True
        self.hp_lambda = hp_lambda

    @staticmethod
    def get_output_shape_for(input_shape):
        return input_shape

    def build(self, input_shape):
        self._trainable_weights = []

    def call(self, x, mask=None):
        return reverse_gradient(x, self.hp_lambda)

    def get_config(self):
        config = {}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def text_to_seq(text, tokenizer, MAX_SEQUENCE_LENGTH=max_len):
    return pad_sequences(tokenizer.texts_to_sequences(text), maxlen = MAX_SEQUENCE_LENGTH) 

def build_rnn(traindata, valdata, testdata, tokenizer, odir=tp_output_dir, LAMBDA_REVERSAL_STRENGTH = 1):
    '''Train, valid, test RNN
    
        lang: The language name
        odir: output directory of prediction results
    '''
    rnn_size = 200
    epochs = 100
    
    # load embedding weights
    weights = np.load(wt_dir+'tp.npy')

    # build model architecture
    text_input = Input(
        shape=(max_len,), dtype='int32', name='input'
    )
    embeds = Embedding(
        weights.shape[0], weights.shape[1],
        weights=[weights], input_length=max_len,
        trainable=True, name='embedding'
    )(text_input)

    bigru = Bidirectional(GRU(
        rnn_size, kernel_initializer="glorot_uniform"), name = 'adv'
    )(embeds)

    dp = Dropout(rate=.2, name='representation')(bigru)

    predicts = Dense(
        1, activation='sigmoid', name='hatepredict'
    )(dp) # binary prediction

    flip_layer_0 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_0 = flip_layer_0(dp)
    flip_out_0 = Dense(
        units=1, activation='sigmoid', name='flip_layer_0'
        )(flip_in_0)

    flip_layer_1 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_1 = flip_layer_1(dp)
    flip_out_1 = Dense(
        units=1, activation='sigmoid', name='flip_layer_1'
        )(flip_in_1)

    flip_layer_2 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_2 = flip_layer_2(dp)
    flip_out_2 = Dense(
        units=1, activation='sigmoid', name='flip_layer_2'
        )(flip_in_2)

    flip_layer_3 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_3 = flip_layer_3(dp)
    flip_out_3 = Dense(
        units=1, activation='sigmoid', name='flip_layer_3'
        )(flip_in_3)

    flip_layer_4 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_4 = flip_layer_4(dp)
    flip_out_4 = Dense(
        units=1, activation='sigmoid', name='flip_layer_4'
        )(flip_in_4)

    flip_layer_5 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_5 = flip_layer_5(dp)
    flip_out_5 = Dense(
        units=1, activation='sigmoid', name='flip_layer_5'
        )(flip_in_5)

    flip_layer_6 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_6 = flip_layer_6(dp)
    flip_out_6 = Dense(
        units=1, activation='sigmoid', name='flip_layer_6'
        )(flip_in_6)

    flip_layer_7 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_7 = flip_layer_7(dp)
    flip_out_7 = Dense(
        units=1, activation='sigmoid', name='flip_layer_7'
        )(flip_in_7)

    flip_layer_8 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_8 = flip_layer_8(dp)
    flip_out_8 = Dense(
        units=1, activation='sigmoid', name='flip_layer_8'
        )(flip_in_8)

    flip_layer_9 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_9 = flip_layer_9(dp)
    flip_out_9 = Dense(
        units=1, activation='sigmoid', name='flip_layer_9'
        )(flip_in_9)

    flip_layer_10 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_10 = flip_layer_10(dp)
    flip_out_10 = Dense(
        units=1, activation='sigmoid', name='flip_layer_10'
        )(flip_in_10)

    flip_layer_11 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_11 = flip_layer_11(dp)
    flip_out_11 = Dense(
        units=1, activation='sigmoid', name='flip_layer_11'
        )(flip_in_11)

    flip_layer_12 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_12 = flip_layer_12(dp)
    flip_out_12 = Dense(
        units=1, activation='sigmoid', name='flip_layer_12'
        )(flip_in_12)

    flip_layer_13 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_13 = flip_layer_13(dp)
    flip_out_13 = Dense(
        units=1, activation='sigmoid', name='flip_layer_13'
        )(flip_in_13)

    flip_layer_14 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_14 = flip_layer_14(dp)
    flip_out_14 = Dense(
        units=1, activation='sigmoid', name='flip_layer_14'
        )(flip_in_14)

    flip_layer_15 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_15 = flip_layer_15(dp)
    flip_out_15 = Dense(
        units=1, activation='sigmoid', name='flip_layer_15'
        )(flip_in_15)

    flip_layer_16 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_16 = flip_layer_16(dp)
    flip_out_16 = Dense(
        units=1, activation='sigmoid', name='flip_layer_16'
        )(flip_in_16)

    flip_layer_17 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_17 = flip_layer_17(dp)
    flip_out_17 = Dense(
        units=1, activation='sigmoid', name='flip_layer_17'
        )(flip_in_17)

    flip_layer_18 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_18 = flip_layer_18(dp)
    flip_out_18 = Dense(
        units=1, activation='sigmoid', name='flip_layer_18'
        )(flip_in_18)

    flip_layer_19 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_19 = flip_layer_19(dp)
    flip_out_19 = Dense(
        units=1, activation='sigmoid', name='flip_layer_19'
        )(flip_in_19)

    flip_layer_20 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_20 = flip_layer_20(dp)
    flip_out_20 = Dense(
        units=1, activation='sigmoid', name='flip_layer_20'
        )(flip_in_20)

    flip_layer_21 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_21 = flip_layer_21(dp)
    flip_out_21 = Dense(
        units=1, activation='sigmoid', name='flip_layer_21'
        )(flip_in_21)

    flip_layer_22 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_22 = flip_layer_22(dp)
    flip_out_22 = Dense(
        units=1, activation='sigmoid', name='flip_layer_22'
        )(flip_in_22)

    flip_layer_23 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_23 = flip_layer_23(dp)
    flip_out_23 = Dense(
        units=1, activation='sigmoid', name='flip_layer_23'
        )(flip_in_23)

    flip_layer_24 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_24 = flip_layer_24(dp)
    flip_out_24 = Dense(
        units=1, activation='sigmoid', name='flip_layer_24'
        )(flip_in_24)

    flip_layer_25 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_25 = flip_layer_25(dp)
    flip_out_25 = Dense(
        units=1, activation='sigmoid', name='flip_layer_25'
        )(flip_in_25)

    flip_layer_26 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_26 = flip_layer_26(dp)
    flip_out_26 = Dense(
        units=1, activation='sigmoid', name='flip_layer_26'
        )(flip_in_26)

    flip_layer_27 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_27 = flip_layer_27(dp)
    flip_out_27 = Dense(
        units=1, activation='sigmoid', name='flip_layer_27'
        )(flip_in_27)

    flip_layer_28 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_28 = flip_layer_28(dp)
    flip_out_28 = Dense(
        units=1, activation='sigmoid', name='flip_layer_28'
        )(flip_in_28)

    flip_layer_29 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_29 = flip_layer_29(dp)
    flip_out_29 = Dense(
        units=1, activation='sigmoid', name='flip_layer_29'
        )(flip_in_29)

    flip_layer_30 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_30 = flip_layer_30(dp)
    flip_out_30 = Dense(
        units=1, activation='sigmoid', name='flip_layer_30'
        )(flip_in_30)

    flip_layer_31 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_31 = flip_layer_31(dp)
    flip_out_31 = Dense(
        units=1, activation='sigmoid', name='flip_layer_31'
        )(flip_in_31)

    flip_layer_32 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_32 = flip_layer_32(dp)
    flip_out_32 = Dense(
        units=1, activation='sigmoid', name='flip_layer_32'
        )(flip_in_32)

    flip_layer_33 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_33 = flip_layer_33(dp)
    flip_out_33= Dense(
        units=1, activation='sigmoid', name='flip_layer_33'
        )(flip_in_33)

    flip_layer_34 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_34= flip_layer_34(dp)
    flip_out_34 = Dense(
        units=1, activation='sigmoid', name='flip_layer_34'
        )(flip_in_34)

    flip_layer_35 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_35 = flip_layer_35(dp)
    flip_out_35 = Dense(
        units=1, activation='sigmoid', name='flip_layer_35'
        )(flip_in_35)

    flip_layer_36 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_36 = flip_layer_36(dp)
    flip_out_36 = Dense(
        units=1, activation='sigmoid', name='flip_layer_36'
        )(flip_in_36)

    flip_layer_37 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_37 = flip_layer_37(dp)
    flip_out_37 = Dense(
        units=1, activation='sigmoid', name='flip_layer_37'
        )(flip_in_37)

    flip_layer_38 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_38 = flip_layer_38(dp)
    flip_out_38 = Dense(
        units=1, activation='sigmoid', name='flip_layer_38'
        )(flip_in_38)

    flip_layer_39 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_39 = flip_layer_39(dp)
    flip_out_39 = Dense(
        units=1, activation='sigmoid', name='flip_layer_39'
        )(flip_in_39)

    flip_layer_40 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_40 = flip_layer_40(dp)
    flip_out_40 = Dense(
        units=1, activation='sigmoid', name='flip_layer_40'
        )(flip_in_40)

    flip_layer_41 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_41 = flip_layer_41(dp)
    flip_out_41 = Dense(
        units=1, activation='sigmoid', name='flip_layer_41'
        )(flip_in_41)

    flip_layer_42 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_42 = flip_layer_42(dp)
    flip_out_42 = Dense(
        units=1, activation='sigmoid', name='flip_layer_42'
        )(flip_in_42)

    flip_layer_43 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_43 = flip_layer_43(dp)
    flip_out_43 = Dense(
        units=1, activation='sigmoid', name='flip_layer_43'
        )(flip_in_43)

    flip_layer_44 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_44 = flip_layer_44(dp)
    flip_out_44 = Dense(
        units=1, activation='sigmoid', name='flip_layer_44'
        )(flip_in_44)

    flip_layer_45 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_45 = flip_layer_45(dp)
    flip_out_45 = Dense(
        units=1, activation='sigmoid', name='flip_layer_45'
        )(flip_in_45)

    flip_layer_46 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_46 = flip_layer_46(dp)
    flip_out_46 = Dense(
        units=1, activation='sigmoid', name='flip_layer_46'
        )(flip_in_46)

    flip_layer_47 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_47 = flip_layer_47(dp)
    flip_out_47 = Dense(
        units=1, activation='sigmoid', name='flip_layer_47'
        )(flip_in_47)

    flip_layer_48 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_48 = flip_layer_48(dp)
    flip_out_48 = Dense(
        units=1, activation='sigmoid', name='flip_layer_48'
        )(flip_in_48)

    flip_layer_49 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_49 = flip_layer_49(dp)
    flip_out_49 = Dense(
        units=1, activation='sigmoid', name='flip_layer_49'
        )(flip_in_49)

    flip_layer_50 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_50 = flip_layer_50(dp)
    flip_out_50 = Dense(
        units=1, activation='sigmoid', name='flip_layer_50'
        )(flip_in_50)

    flip_layer_51 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_51 = flip_layer_51(dp)
    flip_out_51 = Dense(
        units=1, activation='sigmoid', name='flip_layer_51'
        )(flip_in_51)

    flip_layer_52 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_52 = flip_layer_52(dp)
    flip_out_52 = Dense(
        units=1, activation='sigmoid', name='flip_layer_52'
        )(flip_in_52)

    flip_layer_53 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_53 = flip_layer_53(dp)
    flip_out_53 = Dense(
        units=1, activation='sigmoid', name='flip_layer_53'
        )(flip_in_53)

    flip_layer_54 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_54 = flip_layer_54(dp)
    flip_out_54 = Dense(
        units=1, activation='sigmoid', name='flip_layer_54'
        )(flip_in_54)

    flip_layer_55 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_55 = flip_layer_55(dp)
    flip_out_55 = Dense(
        units=1, activation='sigmoid', name='flip_layer_55'
        )(flip_in_55)

    flip_layer_56 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_56 = flip_layer_56(dp)
    flip_out_56 = Dense(
        units=1, activation='sigmoid', name='flip_layer_56'
        )(flip_in_56)

    flip_layer_57 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_57 = flip_layer_57(dp)
    flip_out_57 = Dense(
        units=1, activation='sigmoid', name='flip_layer_57'
        )(flip_in_57)

    flip_layer_58 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_58 = flip_layer_58(dp)
    flip_out_58 = Dense(
        units=1, activation='sigmoid', name='flip_layer_58'
        )(flip_in_58)

    flip_layer_59 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_59 = flip_layer_59(dp)
    flip_out_59 = Dense(
        units=1, activation='sigmoid', name='flip_layer_59'
        )(flip_in_59)

    flip_layer_60 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_60 = flip_layer_60(dp)
    flip_out_60 = Dense(
        units=1, activation='sigmoid', name='flip_layer_60'
        )(flip_in_60)

    flip_layer_61 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_61 = flip_layer_61(dp)
    flip_out_61 = Dense(
        units=1, activation='sigmoid', name='flip_layer_61'
        )(flip_in_61)

    model = Model(inputs=text_input, outputs=[predicts, flip_out_0, flip_out_1, flip_out_2, flip_out_3, flip_out_4, flip_out_5, flip_out_6, 
                flip_out_7, flip_out_8, flip_out_9, flip_out_10, flip_out_11, flip_out_12, flip_out_13, flip_out_14, flip_out_15, flip_out_16,
                flip_out_17, flip_out_18, flip_out_19, flip_out_20, flip_out_21, flip_out_22, flip_out_23, flip_out_24, flip_out_25, flip_out_26,
                flip_out_27, flip_out_28, flip_out_29, flip_out_30, flip_out_31, flip_out_32, flip_out_33, flip_out_34, flip_out_35, flip_out_36,
                flip_out_37, flip_out_38, flip_out_39, flip_out_40, flip_out_41, flip_out_42, flip_out_43, flip_out_44, flip_out_45, flip_out_46,
                flip_out_47, flip_out_48, flip_out_49, flip_out_50, flip_out_51, flip_out_52, flip_out_53, flip_out_54, flip_out_55, flip_out_56,
                flip_out_57, flip_out_58, flip_out_59, flip_out_60, flip_out_61])#eth_out

    advmodel = Model(inputs=text_input, outputs=[flip_out_0, flip_out_1, flip_out_2, flip_out_3, flip_out_4, flip_out_5, flip_out_6, 
                flip_out_7, flip_out_8, flip_out_9, flip_out_10, flip_out_11, flip_out_12, flip_out_13, flip_out_14, flip_out_15, flip_out_16,
                flip_out_17, flip_out_18, flip_out_19, flip_out_20, flip_out_21, flip_out_22, flip_out_23, flip_out_24, flip_out_25, flip_out_26,
                flip_out_27, flip_out_28, flip_out_29, flip_out_30, flip_out_31, flip_out_32, flip_out_33, flip_out_34, flip_out_35, flip_out_36,
                flip_out_37, flip_out_38, flip_out_39, flip_out_40, flip_out_41, flip_out_42, flip_out_43, flip_out_44, flip_out_45, flip_out_46,
                flip_out_47, flip_out_48, flip_out_49, flip_out_50, flip_out_51, flip_out_52, flip_out_53, flip_out_54, flip_out_55, flip_out_56,
                flip_out_57, flip_out_58, flip_out_59, flip_out_60, flip_out_61])#eth_out
   
    layer_names = ['hatepredict']
    loss_weights = [1]
    for i in range(gerrygroups):
        layer_names.append('flip_layer_{}'.format(i))
        loss_weights.append(1)

    loss_dict = {}
    metrics_dict = {}
    adv_loss_dict = {}
    adv_metrics_dict = {}

    for l in layer_names: 
        if l!='hatepredict':
            adv_loss_dict[l] = 'binary_crossentropy'
            adv_metrics_dict[l] = 'accuracy'
        loss_dict[l] = 'binary_crossentropy'
        metrics_dict[l] = 'accuracy'
    
    model.compile(
        loss=loss_dict, optimizer='rmsprop',
        metrics=metrics_dict, loss_weights = loss_weights
    )

    advmodel.compile(
        loss=adv_loss_dict, optimizer='rmsprop',
        metrics=adv_metrics_dict, loss_weights = loss_weights[1:]
    )

    print(model.summary())
    validation_results = {}

    for iterv in range(epochs):
        print('--------------Epoch: {}--------------'.format(iterv))

        train_iter = evaluator.data_iter_TP_gerry_adv(traindata, batch_size=64, tokenizer=tok, MAX_SEQUENCE_LENGTH=max_len)

        # train model

        for (x_train, y_train, y_gerry_class_0, y_gerry_class_1, y_gerry_class_2, y_gerry_class_3, y_gerry_class_4, y_gerry_class_5, 
        y_gerry_class_6, y_gerry_class_7, y_gerry_class_8, y_gerry_class_9, y_gerry_class_10, y_gerry_class_11, y_gerry_class_12, y_gerry_class_13,
        y_gerry_class_14, y_gerry_class_15, y_gerry_class_16, y_gerry_class_17, y_gerry_class_18, y_gerry_class_19, y_gerry_class_20, y_gerry_class_21,
        y_gerry_class_22, y_gerry_class_23, y_gerry_class_24, y_gerry_class_25, y_gerry_class_26, y_gerry_class_27, y_gerry_class_28, y_gerry_class_29,
        y_gerry_class_30, y_gerry_class_31, y_gerry_class_32, y_gerry_class_33, y_gerry_class_34, y_gerry_class_35, y_gerry_class_36, y_gerry_class_37, 
        y_gerry_class_38, y_gerry_class_39, y_gerry_class_40, y_gerry_class_41, y_gerry_class_42, y_gerry_class_43, y_gerry_class_44, y_gerry_class_45, 
        y_gerry_class_46, y_gerry_class_47, y_gerry_class_48, y_gerry_class_49, y_gerry_class_50, y_gerry_class_51, y_gerry_class_52, y_gerry_class_53, 
        y_gerry_class_54, y_gerry_class_55, y_gerry_class_56, y_gerry_class_57, y_gerry_class_58, y_gerry_class_59, y_gerry_class_60, y_gerry_class_61
        ) in train_iter:

            if len(np.unique(y_train)) == 1:
                continue

            x_train = text_to_seq(x_train, tokenizer, max_len)
            
            tmp = model.train_on_batch(
                [x_train], [y_train, 
                    y_gerry_class_0,
                    y_gerry_class_1,
                    y_gerry_class_2,
                    y_gerry_class_3,
                    y_gerry_class_4,
                    y_gerry_class_5,
                    y_gerry_class_6,
                    y_gerry_class_7,
                    y_gerry_class_8,
                    y_gerry_class_9,
                    y_gerry_class_10,
                    y_gerry_class_11,
                    y_gerry_class_12,
                    y_gerry_class_13,
                    y_gerry_class_14,
                    y_gerry_class_15,
                    y_gerry_class_16,
                    y_gerry_class_17,
                    y_gerry_class_18,
                    y_gerry_class_19,
                    y_gerry_class_20,
                    y_gerry_class_21,
                    y_gerry_class_22,
                    y_gerry_class_23,
                    y_gerry_class_24,
                    y_gerry_class_25,
                    y_gerry_class_26,
                    y_gerry_class_27,
                    y_gerry_class_28,
                    y_gerry_class_29,
                    y_gerry_class_30,
                    y_gerry_class_31,
                    y_gerry_class_32,
                    y_gerry_class_33,
                    y_gerry_class_34,
                    y_gerry_class_35,
                    y_gerry_class_36,
                    y_gerry_class_37,
                    y_gerry_class_38,
                    y_gerry_class_39,
                    y_gerry_class_40,
                    y_gerry_class_41,
                    y_gerry_class_42,
                    y_gerry_class_43,
                    y_gerry_class_44,
                    y_gerry_class_45,
                    y_gerry_class_46,
                    y_gerry_class_47,
                    y_gerry_class_48,
                    y_gerry_class_49,
                    y_gerry_class_50,
                    y_gerry_class_51,
                    y_gerry_class_52,
                    y_gerry_class_53,
                    y_gerry_class_54,
                    y_gerry_class_55,
                    y_gerry_class_56,
                    y_gerry_class_57,
                    y_gerry_class_58,
                    y_gerry_class_59,
                    y_gerry_class_60,
                    y_gerry_class_61] #y_tr_ethnicity
                    )
            """advtmp = advmodel.train_on_batch(
                [x_train], [y_gerry_class_0,
                    y_gerry_class_1,
                    y_gerry_class_2,
                    y_gerry_class_3,
                    y_gerry_class_4,
                    y_gerry_class_5,
                    y_gerry_class_6,
                    y_gerry_class_7,
                    y_gerry_class_8,
                    y_gerry_class_9,
                    y_gerry_class_10,
                    y_gerry_class_11,
                    y_gerry_class_12,
                    y_gerry_class_13,
                    y_gerry_class_14,
                    y_gerry_class_15,
                    y_gerry_class_16,
                    y_gerry_class_17,
                    y_gerry_class_18,
                    y_gerry_class_19,
                    y_gerry_class_20,
                    y_gerry_class_21,
                    y_gerry_class_22,
                    y_gerry_class_23,
                    y_gerry_class_24,
                    y_gerry_class_25,
                    y_gerry_class_26,
                    y_gerry_class_27,
                    y_gerry_class_28,
                    y_gerry_class_29,
                    y_gerry_class_30,
                    y_gerry_class_31,
                    y_gerry_class_32,
                    y_gerry_class_33,
                    y_gerry_class_34,
                    y_gerry_class_35,
                    y_gerry_class_36,
                    y_gerry_class_37,
                    y_gerry_class_38,
                    y_gerry_class_39,
                    y_gerry_class_40,
                    y_gerry_class_41,
                    y_gerry_class_42,
                    y_gerry_class_43,
                    y_gerry_class_44,
                    y_gerry_class_45,
                    y_gerry_class_46,
                    y_gerry_class_47,
                    y_gerry_class_48,
                    y_gerry_class_49,
                    y_gerry_class_50,
                    y_gerry_class_51,
                    y_gerry_class_52,
                    y_gerry_class_53,
                    y_gerry_class_54,
                    y_gerry_class_55,
                    y_gerry_class_56,
                    y_gerry_class_57,
                    y_gerry_class_58,
                    y_gerry_class_59,
                    y_gerry_class_60,
                    y_gerry_class_61] #y_tr_ethnicity
                    )"""


        # valid model to find the best model
        print('---------------Validation------------')

        valid_iter = evaluator.data_iter_tp(valdata, batch_size=64, tokenizer=tok, MAX_SEQUENCE_LENGTH=max_len, if_shuffle=False)

        y_preds = []
        y_valids = []
        y_pred_prob = []
        valid_gender = []
        valid_country = []
        valid_super = []

        for x_valid, y_valid, y_va_gender, y_va_country, y_va_super in valid_iter:
            tmp_preds = model.predict([x_valid])
            for item_tmp in tmp_preds[0]:
                y_preds.append(int(round(item_tmp[0])))
                y_pred_prob.append(item_tmp[0])
            y_valids.extend(y_valid)
            valid_gender.extend(y_va_gender)
            valid_country.extend(y_va_country)
            valid_super.extend(y_va_super)

        valid_f1 = f1_score(
            y_true=y_valids, y_pred=y_preds, 
            average='macro',
        )
        print('Validating f1-macro score: ' + str(valid_f1))

        results = {'pred': y_preds, 'pred_prob': y_pred_prob, 'gender': valid_gender, 
                    'country': valid_country, 'supertopic': valid_super, 'label': y_valids}
    
        results_df = pd.DataFrame(results)
    
        gerry_score =   Gerrymandering_eval(results_df,
                                        tasks = ['gender', 'country', 'supertopic'],
                                        label = "label", 
                                        pred="pred",
                                        pred_prob = "pred_prob",
                                        all_labels = True,
                                        print_results = False
                                        )
    
        selected_attributes = bias_attribute_selection_simple(gerry_eval_results = gerry_score, 
                                                attributes = ['gender', 'country', 'supertopic'],
                                                subgroup_min_size = 0,
                                                df = results_df,
                                                key_mapping = tp_keymapping
                                                )

        avg_violation = np.mean([i[0] for i in selected_attributes['GAP_t11']])
        max_violation = np.max([i[0] for i in selected_attributes['GAP_t11']])
        validation_results[iterv] = (avg_violation, valid_f1)
        print(iterv, valid_f1, avg_violation, max_violation)
        
        print('--------------Test--------------------')
        y_preds = []
        y_probs = []
        test_gender = []
        test_country = []
        test_supertopic = []
        test_label = []

        test_iter = evaluator.data_iter_tp(testdata, batch_size=64, tokenizer=tok, MAX_SEQUENCE_LENGTH=max_len, if_shuffle=False)

        for x_test, y_test, gender_test, country_test, st_test in test_iter:
            tmp_preds = model.predict([x_test])
            for item_tmp in tmp_preds[0]:
                y_probs.append(item_tmp[0])
                y_preds.append(int(round(item_tmp[0])))
            test_gender.extend(gender_test)
            test_country.extend(country_test)
            test_supertopic.extend(st_test)
            test_label.extend(y_test)
        
        results = {'preds': y_preds, 'pred_prob': y_probs, 'gender': test_gender, 'country': test_country, 'supertopic': test_supertopic, 'label': test_label}
        results_df = pd.DataFrame(results)
        results_df.to_csv(odir+'TP_RNN_adv_Gerry_results_{}_{}.tsv'.format(iterv, LAMBDA_REVERSAL_STRENGTH), index=False)

    with open(odir+'validation_results_adv_TP_RNN_Gerry_{}.pkl'.format(LAMBDA_REVERSAL_STRENGTH), 'wb') as f:
        pickle.dump(validation_results, f)
    
if __name__ == '__main__':
    with open('/home/sssub/intersectional_bias/Multilingual_Fairness_LREC-master/tp-subset-data-emnlp-6topics.pkl', 'rb') as f:
        data = pickle.load(f)

    tok_dir = '/home/sssub/intersectional_bias/Multilingual_Fairness_LREC-master/data/tp/tokenizer/'
   
    import random
    fulllist = [i for i in range(len(data))]

    random.seed(1)
    train_list = random.sample(fulllist, int(0.8*len(data)))

    non_train_list = []

    for i in range(len(data)):
        if i not in train_list: non_train_list.append(i)

    random.seed(1)
    dev_list = random.sample(non_train_list, int(0.5*len(non_train_list)))

    test_list = []

    for i in non_train_list:
        if i not in dev_list: test_list.append(i)

    traindata = [data[i] for i in train_list]
    devdata = [data[i] for i in dev_list]
    testdata = [data[i] for i in test_list]

    if os.path.exists('trainlist_tp.pkl'):
        with open('trainlist_tp.pkl', 'rb') as f:
            _traindata = pickle.load(f)
        assert _traindata == train_list
    
    if os.path.exists('vallist_tp.pkl'):
        with open('vallist_tp.pkl', 'rb') as f:
            _valdata = pickle.load(f)
        assert _valdata == dev_list
    
    if os.path.exists('testlist_tp.pkl'):
        with open('testlist_tp.pkl', 'rb') as f:
            _testdata = pickle.load(f)
        assert _testdata == test_list
    
    print(len(traindata), len(devdata), len(testdata), len(set(train_list)), len(set(dev_list)), len(set(test_list)))
    print(set(train_list).intersection(set(dev_list)), set(test_list).intersection(set(dev_list)))

    with open(tok_dir+'tp.tkn', 'rb') as rfile:
        tok = pickle.load(rfile)

    for LAMBDA_REVERSAL_STRENGTH in [0.0001, 0.001, 0.01, 0.05, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 95, 100]:
        build_rnn(traindata, devdata, testdata, tokenizer = tok, odir=tp_output_dir, LAMBDA_REVERSAL_STRENGTH = LAMBDA_REVERSAL_STRENGTH)
    
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

tf.random.set_seed(1)
os.environ['PYTHONHASHSEED']=str(1)
np.random.seed(1)

gerrygroups = 80

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

def build_rnn(lang, odir, LAMBDA_REVERSAL_STRENGTH = 1):
    '''Train, valid, test RNN
    
        lang: The language name
        odir: output directory of prediction results
    '''
    doc_idx = 2
    rnn_size = 200
    max_len = 40 # sequence length
    epochs = 100

    encode_dir = '/home/sssub/intersectional_bias/Multilingual_Fairness_LREC-master/data/encode/'+lang+'/'
    indices_dir = '/home/sssub/intersectional_bias/Multilingual_Fairness_LREC-master/data/indices/'+lang+'/'
    wt_dir = '/home/sssub/intersectional_bias/Multilingual_Fairness_LREC-master/resources/weight/'
    res_dir = '/home/sssub/intersectional_bias/Multilingual_Fairness_LREC-master/resources/classifier/'
    
    # load embedding weights
    weights = np.load(wt_dir+lang+'.npy')

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

    flip_layer_62 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_62 = flip_layer_62(dp)
    flip_out_62 = Dense(
        units=1, activation='sigmoid', name='flip_layer_62'
        )(flip_in_62)

    flip_layer_63 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_63 = flip_layer_63(dp)
    flip_out_63 = Dense(
        units=1, activation='sigmoid', name='flip_layer_63'
        )(flip_in_63)

    flip_layer_64 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_64 = flip_layer_64(dp)
    flip_out_64 = Dense(
        units=1, activation='sigmoid', name='flip_layer_64'
        )(flip_in_64)

    flip_layer_65 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_65 = flip_layer_65(dp)
    flip_out_65 = Dense(
        units=1, activation='sigmoid', name='flip_layer_65'
        )(flip_in_65)

    flip_layer_66 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_66 = flip_layer_66(dp)
    flip_out_66 = Dense(
        units=1, activation='sigmoid', name='flip_layer_66'
        )(flip_in_66)

    flip_layer_67 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_67 = flip_layer_67(dp)
    flip_out_67 = Dense(
        units=1, activation='sigmoid', name='flip_layer_67'
        )(flip_in_67)

    flip_layer_68 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_68 = flip_layer_68(dp)
    flip_out_68 = Dense(
        units=1, activation='sigmoid', name='flip_layer_68'
        )(flip_in_68)

    flip_layer_69 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_69 = flip_layer_69(dp)
    flip_out_69 = Dense(
        units=1, activation='sigmoid', name='flip_layer_69'
        )(flip_in_69)

    flip_layer_70 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_70 = flip_layer_70(dp)
    flip_out_70 = Dense(
        units=1, activation='sigmoid', name='flip_layer_70'
        )(flip_in_70)

    flip_layer_71 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_71 = flip_layer_71(dp)
    flip_out_71 = Dense(
        units=1, activation='sigmoid', name='flip_layer_71'
        )(flip_in_71)

    flip_layer_72 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_72 = flip_layer_72(dp)
    flip_out_72 = Dense(
        units=1, activation='sigmoid', name='flip_layer_72'
        )(flip_in_72)

    flip_layer_73 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_73 = flip_layer_73(dp)
    flip_out_73 = Dense(
        units=1, activation='sigmoid', name='flip_layer_73'
        )(flip_in_73)

    flip_layer_74 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_74 = flip_layer_74(dp)
    flip_out_74 = Dense(
        units=1, activation='sigmoid', name='flip_layer_74'
        )(flip_in_74)

    flip_layer_75 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_75= flip_layer_75(dp)
    flip_out_75 = Dense(
        units=1, activation='sigmoid', name='flip_layer_75'
        )(flip_in_75)

    flip_layer_76 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_76 = flip_layer_76(dp)
    flip_out_76 = Dense(
        units=1, activation='sigmoid', name='flip_layer_76'
        )(flip_in_76)

    flip_layer_77 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_77 = flip_layer_77(dp)
    flip_out_77 = Dense(
        units=1, activation='sigmoid', name='flip_layer_77'
        )(flip_in_77)

    flip_layer_78 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_78 = flip_layer_78(dp)
    flip_out_78 = Dense(
        units=1, activation='sigmoid', name='flip_layer_78'
        )(flip_in_78)

    flip_layer_79 = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    flip_in_79 = flip_layer_79(dp)
    flip_out_79 = Dense(
        units=1, activation='sigmoid', name='flip_layer_79'
        )(flip_in_79)

    model = Model(inputs=text_input, outputs=[predicts, flip_out_0, flip_out_1, flip_out_2, flip_out_3, flip_out_4, flip_out_5, flip_out_6, 
                flip_out_7, flip_out_8, flip_out_9, flip_out_10, flip_out_11, flip_out_12, flip_out_13, flip_out_14, flip_out_15, flip_out_16,
                flip_out_17, flip_out_18, flip_out_19, flip_out_20, flip_out_21, flip_out_22, flip_out_23, flip_out_24, flip_out_25, flip_out_26,
                flip_out_27, flip_out_28, flip_out_29, flip_out_30, flip_out_31, flip_out_32, flip_out_33, flip_out_34, flip_out_35, flip_out_36,
                flip_out_37, flip_out_38, flip_out_39, flip_out_40, flip_out_41, flip_out_42, flip_out_43, flip_out_44, flip_out_45, flip_out_46,
                flip_out_47, flip_out_48, flip_out_49, flip_out_50, flip_out_51, flip_out_52, flip_out_53, flip_out_54, flip_out_55, flip_out_56,
                flip_out_57, flip_out_58, flip_out_59, flip_out_60, flip_out_61, flip_out_62, flip_out_63, flip_out_64, flip_out_65, flip_out_66,
                flip_out_67, flip_out_68, flip_out_69, flip_out_70, flip_out_71, flip_out_72, flip_out_73, flip_out_74, flip_out_75, flip_out_76,
                flip_out_77, flip_out_78, flip_out_79
                ])#eth_out
   
    advmodel = Model(inputs=text_input, outputs=[flip_out_0, flip_out_1, flip_out_2, flip_out_3, flip_out_4, flip_out_5, flip_out_6, 
                flip_out_7, flip_out_8, flip_out_9, flip_out_10, flip_out_11, flip_out_12, flip_out_13, flip_out_14, flip_out_15, flip_out_16,
                flip_out_17, flip_out_18, flip_out_19, flip_out_20, flip_out_21, flip_out_22, flip_out_23, flip_out_24, flip_out_25, flip_out_26,
                flip_out_27, flip_out_28, flip_out_29, flip_out_30, flip_out_31, flip_out_32, flip_out_33, flip_out_34, flip_out_35, flip_out_36,
                flip_out_37, flip_out_38, flip_out_39, flip_out_40, flip_out_41, flip_out_42, flip_out_43, flip_out_44, flip_out_45, flip_out_46,
                flip_out_47, flip_out_48, flip_out_49, flip_out_50, flip_out_51, flip_out_52, flip_out_53, flip_out_54, flip_out_55, flip_out_56,
                flip_out_57, flip_out_58, flip_out_59, flip_out_60, flip_out_61, flip_out_62, flip_out_63, flip_out_64, flip_out_65, flip_out_66,
                flip_out_67, flip_out_68, flip_out_69, flip_out_70, flip_out_71, flip_out_72, flip_out_73, flip_out_74, flip_out_75, flip_out_76,
                flip_out_77, flip_out_78, flip_out_79
                ])#eth_out
   
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

    best_valid_f1 = 0.0
    best_model = None
    
    best_hmean_v = 0
    best_hmean_epoch = 0
    best_bias_v = 1
    best_bias_epoch = 0
    validation_results = {}
    best_perf = 0
    best_perf_epoch = 0

    for e in range(epochs):
        print('--------------Epoch: {}--------------'.format(e))

        # load training and batch dataset
        train_iter = evaluator.data_iter_gerry_adv(
            indices_dir+'train.tsv', batch_size=64
        )

        output_demo = []
        # train model
        for (_, x_train, y_train, y_gerry_class_0, y_gerry_class_1, y_gerry_class_2, y_gerry_class_3, y_gerry_class_4, y_gerry_class_5, 
        y_gerry_class_6, y_gerry_class_7, y_gerry_class_8, y_gerry_class_9, y_gerry_class_10, y_gerry_class_11, y_gerry_class_12, y_gerry_class_13,
        y_gerry_class_14, y_gerry_class_15, y_gerry_class_16, y_gerry_class_17, y_gerry_class_18, y_gerry_class_19, y_gerry_class_20, y_gerry_class_21,
        y_gerry_class_22, y_gerry_class_23, y_gerry_class_24, y_gerry_class_25, y_gerry_class_26, y_gerry_class_27, y_gerry_class_28, y_gerry_class_29,
        y_gerry_class_30, y_gerry_class_31, y_gerry_class_32, y_gerry_class_33, y_gerry_class_34, y_gerry_class_35, y_gerry_class_36, y_gerry_class_37, 
        y_gerry_class_38, y_gerry_class_39, y_gerry_class_40, y_gerry_class_41, y_gerry_class_42, y_gerry_class_43, y_gerry_class_44, y_gerry_class_45, 
        y_gerry_class_46, y_gerry_class_47, y_gerry_class_48, y_gerry_class_49, y_gerry_class_50, y_gerry_class_51, y_gerry_class_52, y_gerry_class_53, 
        y_gerry_class_54, y_gerry_class_55, y_gerry_class_56, y_gerry_class_57, y_gerry_class_58, y_gerry_class_59, y_gerry_class_60, y_gerry_class_61, 
        y_gerry_class_62, y_gerry_class_63, y_gerry_class_64, y_gerry_class_65, y_gerry_class_66, y_gerry_class_67, y_gerry_class_68, y_gerry_class_69, 
        y_gerry_class_70, y_gerry_class_71, y_gerry_class_72, y_gerry_class_73, y_gerry_class_74, y_gerry_class_75, y_gerry_class_76, y_gerry_class_77, 
        y_gerry_class_78, y_gerry_class_79) in train_iter:

            if len(np.unique(y_train)) == 1:
                continue
            
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
                    y_gerry_class_61,
                    y_gerry_class_62,
                    y_gerry_class_63,
                    y_gerry_class_64,
                    y_gerry_class_65,
                    y_gerry_class_66,
                    y_gerry_class_67,
                    y_gerry_class_68,
                    y_gerry_class_69,
                    y_gerry_class_70,
                    y_gerry_class_71,
                    y_gerry_class_72,
                    y_gerry_class_73,
                    y_gerry_class_74,
                    y_gerry_class_75,
                    y_gerry_class_76,
                    y_gerry_class_77,
                    y_gerry_class_78,
                    y_gerry_class_79] #y_tr_ethnicity
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
                    y_gerry_class_61,
                    y_gerry_class_62,
                    y_gerry_class_63,
                    y_gerry_class_64,
                    y_gerry_class_65,
                    y_gerry_class_66,
                    y_gerry_class_67,
                    y_gerry_class_68,
                    y_gerry_class_69,
                    y_gerry_class_70,
                    y_gerry_class_71,
                    y_gerry_class_72,
                    y_gerry_class_73,
                    y_gerry_class_74,
                    y_gerry_class_75,
                    y_gerry_class_76,
                    y_gerry_class_77,
                    y_gerry_class_78,
                    y_gerry_class_79] #y_tr_ethnicity
                    )"""


        # valid model to find the best model
        print('---------------Validation------------')
        valid_iter = evaluator.data_iter_adv(
            indices_dir+'valid.tsv', batch_size=64,
            if_shuffle=False
        )
        y_preds = []
        y_valids = []
        y_pred_prob = []
        valid_gender = []
        valid_age = []
        valid_country = []
        valid_ethnicity = []

        for _, x_valid, y_valid, y_va_gender, y_va_age, y_va_country, y_va_ethnicity in valid_iter:
            tmp_preds = model.predict([x_valid])
            #print(tmp_preds[0].shape, tmp_preds[1].shape, len(tmp_preds))
            for item_tmp in tmp_preds[0]:
                y_preds.append(int(round(item_tmp[0])))
                y_pred_prob.append(item_tmp[0])
            y_valids.extend(y_valid)
            valid_gender.extend(y_va_gender)
            valid_age.extend(y_va_age)
            valid_country.extend(y_va_country)
            valid_ethnicity.extend(y_va_ethnicity)

        valid_f1 = f1_score(
            y_true=y_valids, y_pred=y_preds, 
            average='macro',
        )
        print('Validating f1-macro score: ' + str(valid_f1))

        results = {'pred': y_preds, 'pred_prob': y_pred_prob, 'gender': valid_gender, 'age': valid_age,
                    'country': valid_country, 'ethnicity': valid_ethnicity, 'label': y_valids}
    
        results_df = pd.DataFrame(results)
    
        gerry_score =   Gerrymandering_eval(results_df,
                                        tasks = ['gender', 'age', 'country', 'ethnicity'],
                                        label = "label", 
                                        pred="pred",
                                        pred_prob = "pred_prob",
                                        all_labels = True,
                                        print_results = False
                                        )
    
        selected_attributes = bias_attribute_selection_simple(gerry_eval_results = gerry_score, 
                                                attributes = ['gender', 'age', 'country', 'ethnicity'],
                                                subgroup_min_size = 0,
                                                df = results_df
                                                )
        avg_violation = np.mean([i[0] for i in selected_attributes['GAP_t11']])
        max_violation = np.max([i[0] for i in selected_attributes['GAP_t11']])
        from scipy.stats import hmean

        if hmean([valid_f1, 1-avg_violation]) > best_hmean_v:
            best_hmean_epoch = e
            best_hmean_v = hmean([valid_f1, 1-avg_violation])
        
        if valid_f1 > best_perf:
            best_perf_epoch = e
            best_perf = valid_f1

        if avg_violation < best_bias_v:
            best_bias_epoch = e
            best_bias_v = avg_violation

        print(e-best_bias_epoch, e, valid_f1, avg_violation, max_violation)
        if e-best_bias_epoch >= 5:
            print ("check this model: 5", best_bias_epoch, e-best_bias_epoch)
        if e-best_bias_epoch >= 10:
            print ("check this model: 10", best_bias_epoch, e-best_bias_epoch)
        #if best_valid_f1 < valid_f1:

        validation_results[e] = (avg_violation, valid_f1)
        best_valid_f1 = valid_f1
        best_model = model
        
        print('--------------Test--------------------')
        y_preds = []
        y_probs = []

  
        test_iter = evaluator.data_iter(
            indices_dir+'test.tsv', batch_size=64,
            if_shuffle=False
        )

        ctest = 0
        for _, x_test, y_test in test_iter:
            tmp_preds = best_model.predict([x_test])
            #print(tmp_preds[0].shape, tmp_preds[1].shape, len(tmp_preds))
            for item_tmp in tmp_preds[0]:
                y_probs.append(item_tmp[0])
                y_preds.append(int(round(item_tmp[0])))
            ctest+=1
       
        with open(odir+'hatespeech_adv_gerry_{}_{}.tsv'.format(e, LAMBDA_REVERSAL_STRENGTH), 'w') as wfile:
            with open(indices_dir+'test.tsv') as dfile:
                wfile.write(
                    dfile.readline().strip()+'\tpred\tpred_prob\n')
                for idx, line in enumerate(dfile):
                    wfile.write(line.strip()+'\t'+str(y_preds[idx])+'\t'+str(y_probs[idx])+'\n')

        # save the predicted results
        evaluator.eval(
            odir+'hatespeech_adv_gerry_{}_{}.tsv'.format(e, LAMBDA_REVERSAL_STRENGTH), 
            odir+'hatespeech_adv_gerry_{}_{}.score'.format(e, LAMBDA_REVERSAL_STRENGTH)
        )
    
    with open(odir+'validation_hatespeech_adv_gerry_{}.pkl'.format(LAMBDA_REVERSAL_STRENGTH), 'wb') as f:
        pickle.dump(validation_results, f)
        
    print("best epochs", best_perf_epoch, best_hmean_epoch, best_bias_epoch)

if __name__ == '__main__':
    #langs = [
    #    'English', 'Italian', 'Polish', 
    #    'Portuguese', 'Spanish'
    #]
    langs = ['English']
    
    odir = '/home/sssub/intersectional_bias/Multilingual_Fairness_LREC-master/publish/results/hatespeech/adv/'

    if not os.path.exists(odir):
        os.mkdir(odir)

    for LAMBDA_REVERSAL_STRENGTH in [0.0001, 0.001, 0.01, 0.05, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 95, 100]:
        for lang in langs:
            build_rnn(lang, odir, LAMBDA_REVERSAL_STRENGTH)


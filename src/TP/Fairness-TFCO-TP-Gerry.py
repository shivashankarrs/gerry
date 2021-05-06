import os
import sys
import tempfile
import urllib
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow_constrained_optimization as tfco
from sklearn.metrics import f1_score
from tensorflow.keras.layers import Input, Embedding
from tensorflow.keras.layers import Bidirectional, GRU
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

import sys
sys.path.append('../')

import evaluator, random, math, pickle
import pandas as pd 
from gerry_eval import bias_attribute_selection_simple, Gerrymandering_eval
from gerry_eval import *
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow_constrained_optimization.python.rates import basic_expression
from tensorflow_constrained_optimization.python.rates import binary_rates
from tensorflow_constrained_optimization.python.rates import defaults
from tensorflow_constrained_optimization.python.rates import deferred_tensor
from tensorflow_constrained_optimization.python.rates import expression
from tensorflow_constrained_optimization.python.rates import multiclass_rates
from tensorflow_constrained_optimization.python.rates import subsettable_context
from tensorflow_constrained_optimization.python.rates import term
from tensorflow_constrained_optimization.python.rates import loss

tf.random.set_seed(1)
os.environ['PYTHONHASHSEED']=str(1)
np.random.seed(1)

print("TensorFlow " + tf.__version__)

sys.path.append("/home/sssub/intersectional_bias/Multilingual_Fairness_LREC-master/outputs/")
sys.path.append("/home/sssub/intersectional_bias/Multilingual_Fairness_LREC-master/")

odir = '/home/sssub/intersectional_bias/Multilingual_Fairness_LREC-master/publish/results/TP/TFCO/'
wt_dir = '/home/sssub/intersectional_bias/Multilingual_Fairness_LREC-master/data/tp/embeddings/'
tok_dir = '/home/sssub/intersectional_bias/Multilingual_Fairness_LREC-master/data/tp/tokenizer/'
max_len = 150  
BATCH_SIZE = 128

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

def build_rnn():
    '''Train, valid, test RNN
    
        lang: The language name
        odir: output directory of prediction results
    '''
    rnn_size = 200
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
        rnn_size, kernel_initializer="glorot_uniform"), name = 'inlp'
    )(embeds)

    dp = Dropout(rate=.2)(bigru)
    predicts = Dense(
        1, name='predict'
    )(dp) # binary prediction
    model = Model(inputs=text_input, outputs=predicts)
    print(model.summary())
    return model

def _is_multiclass(context):
  """Returns True iff we're given a multiclass context."""
  if not isinstance(context, subsettable_context.SubsettableContext):
    raise TypeError("context must be a SubsettableContext object")
  raw_context = context.raw_context
  return raw_context.num_classes is not None

def custom_error_rate(context,
               penalty_loss=loss.SoftmaxLoss(),
               constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
    if _is_multiclass(context):
        return multiclass_rates.error_rate(
            context=context,
            penalty_loss=penalty_loss,
            constraint_loss=constraint_loss)

    print("binary classification problem")

    return binary_rates.error_rate(
        context=context,
        penalty_loss=penalty_loss,
        constraint_loss=constraint_loss)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def text_to_seq(text, tokenizer, MAX_SEQUENCE_LENGTH=max_len):
        return pad_sequences(tokenizer.texts_to_sequences(text), maxlen = MAX_SEQUENCE_LENGTH) 

def create_data(datap):
    
    data = {'x': [], 'y': [], 'gender': [], 'country': [], 'supertopic': []}
    
    for d in datap:
        data['x'].append(d['title'].strip() + ' ' + d['text'].strip())
        if d['rate'] == 2:
            data['y'].append(1)
        else:
            data['y'].append(d['rate'])

        if d['gender'] == 'M':
            data['gender'].append(0)
        else:
            data['gender'].append(1)
        
        if d['country'] == 'united_kingdom':
            data['country'].append(0)
        else:
            data['country'].append(1)
            
        data['supertopic'].append(topics_lookup[d['supertopic']])
    return data

def constrained_optimization(traindata, devdata, testdata, tok, nu=0.0001, weighted=True):

    model_constrained = build_rnn()

    input_tensor = tf.Variable(
        np.zeros((BATCH_SIZE, max_len), dtype="int32"),
        name="input")

    def predictions():
        return model_constrained(input_tensor)

    labels_tensor = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="labels")

    groups_tensor_0 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_0")

    groups_tensor_1 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_1")

    groups_tensor_2 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_2")

    groups_tensor_3 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_3")

    groups_tensor_4 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_4")

    groups_tensor_5 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_5")

    groups_tensor_6 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_6")

    groups_tensor_7 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_7")

    groups_tensor_8 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_8")

    groups_tensor_9 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_9")

    groups_tensor_10 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_10")

    groups_tensor_11 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_11")

    groups_tensor_12 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_12")

    groups_tensor_13 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_13")

    groups_tensor_14 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_14")

    groups_tensor_15 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_15")

    groups_tensor_16 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_16")

    groups_tensor_17 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_17")

    groups_tensor_18 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_18")

    groups_tensor_19 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_19")

    groups_tensor_20 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_20")

    groups_tensor_21 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_21")

    groups_tensor_22 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_22")

    groups_tensor_23 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_23")

    groups_tensor_24 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_24")

    groups_tensor_25 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_25")

    groups_tensor_26 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_26")

    groups_tensor_27 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_27")

    groups_tensor_28 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_28")

    groups_tensor_29 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_29")

    groups_tensor_30 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_30")

    groups_tensor_31 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_31")

    groups_tensor_32 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_32")

    groups_tensor_33 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_33")

    groups_tensor_34 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_34")

    groups_tensor_35 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_35")

    groups_tensor_36 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_36")

    groups_tensor_37 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_37")

    groups_tensor_38 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_38")

    groups_tensor_39 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_39")

    groups_tensor_40 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_40")

    groups_tensor_41 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_41")

    groups_tensor_42 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_42")

    groups_tensor_43 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_43")

    groups_tensor_44 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_44")

    groups_tensor_45 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_45")

    groups_tensor_46 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_46")

    groups_tensor_47 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_47")

    groups_tensor_48 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_48")

    groups_tensor_49 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_49")

    groups_tensor_50 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_50")

    groups_tensor_51 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_51")

    groups_tensor_52 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_52")

    groups_tensor_53 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_53")

    groups_tensor_54 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_54")

    groups_tensor_55 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_55")

    groups_tensor_56 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_56")

    groups_tensor_57 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_57")

    groups_tensor_58 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_58")

    groups_tensor_59 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_59")

    groups_tensor_60 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_60")

    groups_tensor_61 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_61")

    context = tfco.rate_context(predictions, lambda: labels_tensor)
    objective = custom_error_rate(context)

    context_subset_0 = context.subset(lambda:groups_tensor_0 >0)
    context_subset_1 = context.subset(lambda:groups_tensor_1 >0)
    context_subset_2 = context.subset(lambda:groups_tensor_2 >0)
    context_subset_3 = context.subset(lambda:groups_tensor_3 >0)
    context_subset_4 = context.subset(lambda:groups_tensor_4 >0)
    context_subset_5 = context.subset(lambda:groups_tensor_5 >0)
    context_subset_6 = context.subset(lambda:groups_tensor_6 >0)
    context_subset_7 = context.subset(lambda:groups_tensor_7 >0)
    context_subset_8 = context.subset(lambda:groups_tensor_8 >0)
    context_subset_9 = context.subset(lambda:groups_tensor_9 >0)
    context_subset_10 = context.subset(lambda:groups_tensor_10 >0)
    context_subset_11 = context.subset(lambda:groups_tensor_11 >0)
    context_subset_12 = context.subset(lambda:groups_tensor_12 >0)
    context_subset_13 = context.subset(lambda:groups_tensor_13 >0)
    context_subset_14 = context.subset(lambda:groups_tensor_14 >0)
    context_subset_15 = context.subset(lambda:groups_tensor_15 >0)
    context_subset_16 = context.subset(lambda:groups_tensor_16 >0)
    context_subset_17 = context.subset(lambda:groups_tensor_17 >0)
    context_subset_18 = context.subset(lambda:groups_tensor_18 >0)
    context_subset_19 = context.subset(lambda:groups_tensor_19 >0)
    context_subset_20 = context.subset(lambda:groups_tensor_20 >0)
    context_subset_21 = context.subset(lambda:groups_tensor_21 >0)
    context_subset_22 = context.subset(lambda:groups_tensor_22 >0)
    context_subset_23 = context.subset(lambda:groups_tensor_23 >0)
    context_subset_24 = context.subset(lambda:groups_tensor_24 >0)
    context_subset_25 = context.subset(lambda:groups_tensor_25 >0)
    context_subset_26 = context.subset(lambda:groups_tensor_26 >0)
    context_subset_27 = context.subset(lambda:groups_tensor_27 >0)
    context_subset_28 = context.subset(lambda:groups_tensor_28 >0)
    context_subset_29 = context.subset(lambda:groups_tensor_29 >0)
    context_subset_30 = context.subset(lambda:groups_tensor_30 >0)
    context_subset_31 = context.subset(lambda:groups_tensor_31 >0)
    context_subset_32 = context.subset(lambda:groups_tensor_32 >0)
    context_subset_33 = context.subset(lambda:groups_tensor_33 >0)
    context_subset_34 = context.subset(lambda:groups_tensor_34 >0)
    context_subset_35 = context.subset(lambda:groups_tensor_35 >0)
    context_subset_36 = context.subset(lambda:groups_tensor_36 >0)
    context_subset_37 = context.subset(lambda:groups_tensor_37 >0)
    context_subset_38 = context.subset(lambda:groups_tensor_38 >0)
    context_subset_39 = context.subset(lambda:groups_tensor_39 >0)
    context_subset_40 = context.subset(lambda:groups_tensor_40 >0)
    context_subset_41 = context.subset(lambda:groups_tensor_41 >0)
    context_subset_42 = context.subset(lambda:groups_tensor_42 >0)
    context_subset_43 = context.subset(lambda:groups_tensor_43 >0)
    context_subset_44 = context.subset(lambda:groups_tensor_44 >0)
    context_subset_45 = context.subset(lambda:groups_tensor_45 >0)
    context_subset_46 = context.subset(lambda:groups_tensor_46 >0)
    context_subset_47 = context.subset(lambda:groups_tensor_47 >0)
    context_subset_48 = context.subset(lambda:groups_tensor_48 >0)
    context_subset_49 = context.subset(lambda:groups_tensor_49 >0)
    context_subset_50 = context.subset(lambda:groups_tensor_50 >0)
    context_subset_51 = context.subset(lambda:groups_tensor_51 >0)
    context_subset_52 = context.subset(lambda:groups_tensor_52 >0)
    context_subset_53 = context.subset(lambda:groups_tensor_53 >0)
    context_subset_54 = context.subset(lambda:groups_tensor_54 >0)
    context_subset_55 = context.subset(lambda:groups_tensor_55 >0)
    context_subset_56 = context.subset(lambda:groups_tensor_56 >0)
    context_subset_57 = context.subset(lambda:groups_tensor_57 >0)
    context_subset_58 = context.subset(lambda:groups_tensor_58 >0)
    context_subset_59 = context.subset(lambda:groups_tensor_59 >0)
    context_subset_60 = context.subset(lambda:groups_tensor_60 >0)
    context_subset_61 = context.subset(lambda:groups_tensor_61 >0)

    inter_data = {}

    tasks = ['gender', 'country', 'supertopic']
    xtrain_tp =  pd.DataFrame(create_data(traindata))
    # gerry combinations
    attribute_distinct_labels = {attribute:list(xtrain_tp[~xtrain_tp[attribute].isnull()][attribute].unique()) for attribute in tasks}
    print (attribute_distinct_labels)
    gerry_combs = Gerrymandering_groups(
        attributes = tasks, 
        attribute_distinct_labels = attribute_distinct_labels
        )
    group_comb_itr = 0

    # iterate all gerry combs
    for task_comb, group_comb in tqdm(gerry_combs):
        group_indices = task_comb_data(xtrain_tp, task_comb, group_comb)
        group_key = "@".join([str(i)+str(j) for i,j in zip(task_comb, group_comb)])
        subdf = xtrain_tp[group_indices]
        _labelsize = 0
        for _ylabel in subdf['y']:
            if _ylabel == 1: _labelsize += 1
        inter_data[group_comb_itr] = len(subdf)/_labelsize
        group_comb_itr+=1

    constraints = []

    contexts_available = [context_subset_0, context_subset_1, context_subset_2, context_subset_3, context_subset_4, context_subset_5, context_subset_6, context_subset_7, context_subset_8, \
                        context_subset_9, context_subset_10, context_subset_11, context_subset_12, context_subset_13, context_subset_14, context_subset_15, context_subset_16, context_subset_17, context_subset_18, \
                        context_subset_19, context_subset_20, context_subset_21, context_subset_22, context_subset_23, context_subset_24, context_subset_25, context_subset_26,
                        context_subset_27, context_subset_28, context_subset_29, context_subset_30, context_subset_31, context_subset_32, context_subset_33, context_subset_34,
                        context_subset_35, context_subset_36, context_subset_37, context_subset_38, context_subset_39, context_subset_40, context_subset_41,  context_subset_42,
                        context_subset_43, context_subset_44, context_subset_45, context_subset_46, context_subset_47, context_subset_48, context_subset_49, context_subset_50,
                        context_subset_51, context_subset_52, context_subset_53, context_subset_54, context_subset_55, context_subset_56, context_subset_57, context_subset_58, context_subset_59,
                        context_subset_60, context_subset_61]

    for i in range(len(contexts_available)):
        if weighted:
            constraints.append(inter_data[i] * (tfco.true_positive_rate(context) -  tfco.true_positive_rate(contexts_available[i])) <= nu) ##CW
            constraints.append(inter_data[i] * (tfco.true_positive_rate(contexts_available[i]) -  tfco.true_positive_rate(context)) <= nu) ##CW
        else:
            constraints.append((tfco.true_positive_rate(context) -  tfco.true_positive_rate(contexts_available[i])) <= nu) ##NW
            constraints.append((tfco.true_positive_rate(contexts_available[i]) -  tfco.true_positive_rate(context)) <= nu) ##NW
        
    problem = tfco.RateMinimizationProblem(objective, constraints)

    # Set up a constrained optimizer.
    optimizer = tfco.LagrangianOptimizerV2(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        num_constraints=problem.num_constraints)

    var_list = (model_constrained.trainable_weights + problem.trainable_variables +
                optimizer.trainable_variables())

    NUM_ITERATIONS = 100 # Number of training iterations.
    objective_list = []
    violations_list = []
    iterv = 0
    validation_results = {}

    while iterv < NUM_ITERATIONS:
        train_iter = evaluator.data_iter_TP_gerry_adv(traindata, batch_size=BATCH_SIZE, tokenizer=tok, MAX_SEQUENCE_LENGTH=max_len)
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
            if x_train.shape[0] < BATCH_SIZE: continue
            x_train = text_to_seq(x_train, tokenizer=tok, MAX_SEQUENCE_LENGTH=max_len)
            
            input_tensor.assign(x_train)  
            labels_tensor.assign(y_train)
            groups_tensor_0.assign(np.array(y_gerry_class_0))
            groups_tensor_1.assign(np.array(y_gerry_class_1))
            groups_tensor_2.assign(np.array(y_gerry_class_2))
            groups_tensor_3.assign(np.array(y_gerry_class_3))
            groups_tensor_4.assign(np.array(y_gerry_class_4))
            groups_tensor_5.assign(np.array(y_gerry_class_5))
            groups_tensor_6.assign(np.array(y_gerry_class_6))
            groups_tensor_7.assign(np.array(y_gerry_class_7))
            groups_tensor_8.assign(np.array(y_gerry_class_8))
            groups_tensor_9.assign(np.array(y_gerry_class_9))
            groups_tensor_10.assign(np.array(y_gerry_class_10))
            groups_tensor_11.assign(np.array(y_gerry_class_11))
            groups_tensor_12.assign(np.array(y_gerry_class_12))
            groups_tensor_13.assign(np.array(y_gerry_class_13))
            groups_tensor_14.assign(np.array(y_gerry_class_14))
            groups_tensor_15.assign(np.array(y_gerry_class_15))
            groups_tensor_16.assign(np.array(y_gerry_class_16))
            groups_tensor_17.assign(np.array(y_gerry_class_17))
            groups_tensor_18.assign(np.array(y_gerry_class_18))
            groups_tensor_19.assign(np.array(y_gerry_class_19))
            groups_tensor_20.assign(np.array(y_gerry_class_20))
            groups_tensor_21.assign(np.array(y_gerry_class_21))
            groups_tensor_22.assign(np.array(y_gerry_class_22))
            groups_tensor_23.assign(np.array(y_gerry_class_23))
            groups_tensor_24.assign(np.array(y_gerry_class_24))
            groups_tensor_25.assign(np.array(y_gerry_class_25))
            groups_tensor_26.assign(np.array(y_gerry_class_26))
            groups_tensor_27.assign(np.array(y_gerry_class_27))
            groups_tensor_28.assign(np.array(y_gerry_class_28))
            groups_tensor_29.assign(np.array(y_gerry_class_29))
            groups_tensor_30.assign(np.array(y_gerry_class_30))
            groups_tensor_31.assign(np.array(y_gerry_class_31))
            groups_tensor_32.assign(np.array(y_gerry_class_32))
            groups_tensor_33.assign(np.array(y_gerry_class_33))
            groups_tensor_34.assign(np.array(y_gerry_class_34))
            groups_tensor_35.assign(np.array(y_gerry_class_35))
            groups_tensor_36.assign(np.array(y_gerry_class_36))
            groups_tensor_37.assign(np.array(y_gerry_class_37))
            groups_tensor_38.assign(np.array(y_gerry_class_38))
            groups_tensor_39.assign(np.array(y_gerry_class_39))
            groups_tensor_40.assign(np.array(y_gerry_class_40))
            groups_tensor_41.assign(np.array(y_gerry_class_41))
            groups_tensor_42.assign(np.array(y_gerry_class_42))
            groups_tensor_43.assign(np.array(y_gerry_class_43))
            groups_tensor_44.assign(np.array(y_gerry_class_44))
            groups_tensor_45.assign(np.array(y_gerry_class_45))
            groups_tensor_46.assign(np.array(y_gerry_class_46))
            groups_tensor_47.assign(np.array(y_gerry_class_47))
            groups_tensor_48.assign(np.array(y_gerry_class_48))
            groups_tensor_49.assign(np.array(y_gerry_class_49))
            groups_tensor_50.assign(np.array(y_gerry_class_50))
            groups_tensor_51.assign(np.array(y_gerry_class_51))
            groups_tensor_52.assign(np.array(y_gerry_class_52))
            groups_tensor_53.assign(np.array(y_gerry_class_53))
            groups_tensor_54.assign(np.array(y_gerry_class_54))
            groups_tensor_55.assign(np.array(y_gerry_class_55))
            groups_tensor_56.assign(np.array(y_gerry_class_56))
            groups_tensor_57.assign(np.array(y_gerry_class_57))
            groups_tensor_58.assign(np.array(y_gerry_class_58))
            groups_tensor_59.assign(np.array(y_gerry_class_59))
            groups_tensor_60.assign(np.array(y_gerry_class_60))
            groups_tensor_61.assign(np.array(y_gerry_class_61))

            optimizer.minimize(problem, var_list=var_list)
            objective = problem.objective()
            violations = problem.constraints()
        objective_list.append(objective)
        violations_list.append(violations)
  
        # valid model to find the best model
        print('---------------Validation------------')
        valid_iter = evaluator.data_iter_tp(devdata, batch_size=BATCH_SIZE, tokenizer=tok, MAX_SEQUENCE_LENGTH=max_len, if_shuffle=False)

        y_preds = []
        y_valids = []
        y_pred_prob = []
        valid_gender = []
        valid_country = []
        valid_super = []

        for x_valid, y_valid, y_va_gender, y_va_country, y_va_super in valid_iter:
            tmp_preds = model_constrained.predict([x_valid])
            for item_tmp in tmp_preds:
                y_preds.append(int(round(sigmoid(item_tmp[0]))))
                y_pred_prob.append(sigmoid(item_tmp[0]))
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
        iterv += 1
        
        print('--------------Test--------------------')

        y_preds = []
        y_probs = []
        test_gender = []
        test_country = []
        test_supertopic = []
        test_label = []

        test_iter = evaluator.data_iter_tp(testdata, batch_size=BATCH_SIZE, tokenizer=tok, MAX_SEQUENCE_LENGTH=max_len, if_shuffle=False)

        for x_test, y_test, gender_test, country_test, st_test in test_iter:
            tmp_preds = model_constrained.predict([x_test])
            for item_tmp in tmp_preds:
                y_probs.append(sigmoid(item_tmp[0]))
                y_preds.append(int(round(sigmoid(item_tmp[0]))))
            test_gender.extend(gender_test)
            test_country.extend(country_test)
            test_supertopic.extend(st_test)
            test_label.extend(y_test)
        
        results = {'preds': y_preds, 'pred_prob': y_probs, 'gender': test_gender, 'country': test_country, 'supertopic': test_supertopic, 'label': test_label}
        results_df = pd.DataFrame(results)
        results_df.to_csv(odir+'TP_RNN_TFCO_Gerry_cw_{}_results_{}_{}.tsv'.format(weighted, iterv, nu), index=False)

    with open(odir+'validation_results_TP_TFCO_Gerry_cw_{}_{}.pkl'.format(weighted, nu), 'wb') as f:
        pickle.dump(validation_results, f)
        
    best_index = tfco.find_best_candidate_index(
        np.array(objective_list), np.array(violations_list), rank_objectives=False)
    print (best_index, "nu:{}".format(nu), "weighted:{}".format(weighted), "best index from TFCO, not ranked")

    best_index = tfco.find_best_candidate_index(
        np.array(objective_list), np.array(violations_list), rank_objectives=True)

    print (best_index, "nu:{}".format(nu), "weighted:{}".format(weighted), "best index from TFCO, ranked")

   
if __name__ == '__main__':

    with open('/home/sssub/intersectional_bias/Multilingual_Fairness_LREC-master/tp-subset-data-emnlp-6topics.pkl', 'rb') as f:
        data = pickle.load(f)

    topics_lookup = {'Computer & Accessories': 0,
                    'Fashion Accessories': 1,
                    'Fitness & Nutrition': 2,
                    'Tires': 3,
                    'Hotels': 4,
                    'Pets': 5}

    
    print ("---------sampling data ----------")
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

    print ("---------loading tokenizer----------")
    with open(tok_dir+'tp.tkn', 'rb') as rfile:
        tok = pickle.load(rfile)

    for nu in [0.0001, 0.0005, 0.0007, 0.001, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1]:
        for weighted in [True, False]:
            model = constrained_optimization(traindata, devdata, testdata, tok, 
                                            nu = nu, weighted = weighted
                                            )
    


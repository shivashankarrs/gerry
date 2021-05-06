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

from gerry_eval import bias_attribute_selection_simple, Gerrymandering_eval, bias_attribute_selection
import evaluator

tf.random.set_seed(1)
os.environ['PYTHONHASHSEED']=str(1)
np.random.seed(1)

max_len = 150  

import sys
sys.path.append("/home/sssub/intersectional_bias/Multilingual_Fairness_LREC-master/")
    
tp_output_dir = '/home/sssub/intersectional_bias/Multilingual_Fairness_LREC-master/publish/results/TP/adv/'
wt_dir = '/home/sssub/intersectional_bias/Multilingual_Fairness_LREC-master/data/tp/embeddings/'

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

def labelencode(supertopics):
    coded_st = []
    for st in supertopics:
        if st == 0: coded_st.append([1, 0, 0, 0, 0, 0])
        elif st == 1: coded_st.append([0, 1, 0, 0, 0, 0])
        elif st == 2: coded_st.append([0, 0, 1, 0, 0, 0])
        elif st == 3: coded_st.append([0, 0, 0, 1, 0, 0])
        elif st == 4: coded_st.append([0, 0, 0, 0, 1, 0])
        elif st == 5: coded_st.append([0, 0, 0, 0, 0, 1])
    return coded_st

def build_rnn(traindata, valdata, testdata, tokenizer, odir=tp_output_dir, LAMBDA_REVERSAL_STRENGTH = 1):
    '''Train, valid, test RNN
    
        lang: The language name
        odir: output directory of prediction results
    '''
    rnn_size = 200
    epochs = 100

    # clf_path = res_dir + lang + '.clf'
    # don't reload classifier for debug usage

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

    gend_flip_layer = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    gend_in = gend_flip_layer(dp)
    gend_out = Dense(
        units=1, activation='sigmoid', name='gender_classifier'
        )(gend_in)

    ctr_flip_layer = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    ctr_in = ctr_flip_layer(dp)
    ctr_out = Dense(
        units=1, activation='sigmoid', name='country_classifier'
        )(ctr_in)

    eth_flip_layer = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    eth_in = eth_flip_layer(dp)
    eth_out = Dense(
        units=6, activation='softmax', name='super_classifier'
        )(eth_in)

    model = Model(inputs=text_input, outputs=[predicts, gend_out, ctr_out, eth_out])#eth_out
    advmodel  = Model(inputs=text_input, outputs=[gend_out, ctr_out, eth_out])#eth_out

    layer_names = ['hatepredict', 'gender_classifier', 'country_classifier', 'super_classifier']
    loss_dict = {}
    metrics_dict = {}
    adv_loss_dict = {}
    adv_metrics_dict = {}


    for l in layer_names: 
        if l == 'super_classifier': 
            loss_dict[l] = 'categorical_crossentropy'    
            adv_loss_dict[l] = 'categorical_crossentropy'    
            adv_metrics_dict[l] = 'accuracy'
        else: 
            if l != 'hatepredict':
                adv_loss_dict[l] = 'binary_crossentropy'
                adv_metrics_dict[l] = 'accuracy'
            loss_dict[l] = 'binary_crossentropy'
        metrics_dict[l] = 'accuracy'

    model.compile(
        loss=loss_dict, optimizer='rmsprop',
        metrics=metrics_dict, loss_weights = [1, 1, 1, 1]
    )

    advmodel.compile(
        loss=adv_loss_dict, optimizer='rmsprop',
        metrics=adv_metrics_dict, loss_weights = [1, 1, 1]
    )

    print(model.summary())
    validation_results = {}

    for e in range(epochs):
        print('--------------Epoch: {}--------------'.format(e))

        train_iter = evaluator.data_iter_tp(traindata, batch_size=64, tokenizer=tok, MAX_SEQUENCE_LENGTH=max_len)

        # train model
        for x_train, y_train, gender_train, country_train, supertopic_train in train_iter:
            if len(np.unique(y_train)) == 1:
                continue

            enc_super_train = labelencode(supertopic_train)
            
            tmp = model.train_on_batch(
                [x_train], [np.array(y_train), np.array(gender_train), np.array(country_train), np.array(enc_super_train)]
            )

            """advtmp = advmodel.train_on_batch(
                [x_train], [np.array(gender_train), np.array(country_train), np.array(enc_super_train)]
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
        validation_results[e] = (avg_violation, valid_f1)
        print(e, valid_f1, avg_violation, max_violation)
        
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
        results_df.to_csv(odir+'TP_RNN_adv_results_{}_{}.tsv'.format(e, LAMBDA_REVERSAL_STRENGTH), index=False)

    with open(odir+'validation_results_adv_TP_RNN_Multiple_{}.pkl'.format(LAMBDA_REVERSAL_STRENGTH), 'wb') as f:
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
        

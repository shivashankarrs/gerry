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

from gerry_eval import Gerrymandering_eval, get_all_combs, Gerrymandering_groups, bias_attribute_selection_simple
from keras.utils import to_categorical

tf.random.set_seed(1)
os.environ['PYTHONHASHSEED']=str(1)
np.random.seed(1)

MAX_SEQUENCE_LENGTH = 768  

tp_output_dir = '/home/sssub/intersectional_bias/Multilingual_Fairness_LREC-master/publish/results/biasbios/adv/'

import math
def rms(_arr):
    arr = [v for v in _arr if not math.isnan(v)]
    return np.sqrt(np.mean(np.square(arr)))

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

bios_keymapping = {
    'gender':{0:"male", 1:"female"}, 
    'economy': {1: 'High economy',
                0: 'Rest'}
}

def build_mlp(xtrain, ytrain, ytrain_gender, ytrain_econ, xvalid, yvalid, yvalid_gender, yvalid_econ, xtest, ytest, ytest_gender, ytest_econ, LAMBDA_REVERSAL_STRENGTH=1):
    '''Train, valid, test RNN
        lang: The language name
        odir: output directory of prediction results
    '''
    hidden_size = 300
    epochs = 300
    
    # build model architecture
    text_input = Input(
        shape=(MAX_SEQUENCE_LENGTH,), dtype='float32', name='input'
    )
    
    mlp = Dense(
        hidden_size, activation='relu'
    )(text_input) # binary prediction

    dp = Dropout(rate=.2, name='inlp')(mlp)

    predicts = Dense(
        1, activation='sigmoid', name='predict'
    )(dp) # binary prediction

    gend_flip_layer = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    gend_in = gend_flip_layer(dp)
    gend_out = Dense(
        units=1, activation='sigmoid', name='gender_classifier'
        )(gend_in)

    econ_flip_layer = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    econ_in = econ_flip_layer(dp)
    econ_out = Dense(
        units=1, activation='sigmoid', name='economy_classifier'
        )(econ_in)

    model = Model(inputs=text_input, outputs=[predicts, gend_out, econ_out])
    advmodel = Model(inputs=text_input, outputs=[gend_out, econ_out])

    layer_names = ['predict', 'gender_classifier', 'economy_classifier']
    loss_dict = {}
    metrics_dict = {}
    adv_loss_dict = {}
    adv_metrics_dict = {}
    
    for l in layer_names: 
        if l == 'predict':
            loss_dict[l] = 'binary_crossentropy'
            metrics_dict[l] = 'accuracy'
        else:
            loss_dict[l] = 'binary_crossentropy'
            metrics_dict[l] = 'accuracy'
            adv_loss_dict[l] = 'binary_crossentropy'
            adv_metrics_dict[l] = 'accuracy'
            
    model.compile(
        loss=loss_dict, optimizer='rmsprop',
        metrics=metrics_dict, loss_weights = [1, 1, 1]
    )

    advmodel.compile(
        loss=adv_loss_dict, optimizer='rmsprop',
        metrics=adv_metrics_dict, loss_weights = [1, 1]
    )

    print(model.summary())

    validation_results={}

    model.fit(xtrain, [ytrain, ytrain_gender, ytrain_econ], validation_data=(xvalid, [yvalid, yvalid_gender, yvalid_econ]), epochs=1, verbose=0, batch_size=64)

    for iterv in range(epochs):
        history = model.fit(xtrain, [ytrain, ytrain_gender, ytrain_econ], validation_data=(xvalid, [yvalid, yvalid_gender, yvalid_econ]), epochs=1, verbose=0, batch_size=64)
        #advhistory = advmodel.fit(xtrain, [ytrain_gender, ytrain_econ], validation_data=(xvalid, [yvalid_gender, yvalid_econ]), epochs=1, verbose=1, batch_size=64)

        _y_preds = []
        _y_probs = []

        tmp_preds = model.predict([xvalid])
        for item_tmp in tmp_preds[0]:
            _y_probs.append(item_tmp[0])
            _y_preds.append(int(round(item_tmp[0])))
        
        valid_f1 = f1_score(
            y_true=yvalid, y_pred=_y_preds, 
            average='macro',
        )
        print('Validating f1-macro score: ' + str(valid_f1))

        results = {'preds': _y_preds, 'pred_prob': _y_probs, 'gender': yvalid_gender, 'economy': yvalid_econ, 'label': yvalid}
    
        _results_df = pd.DataFrame(results)
    
        gerry_score =   Gerrymandering_eval(_results_df,
                                tasks = ['gender', 'economy'],
                                label = "label", 
                                pred="preds",
                                pred_prob = "pred_prob",
                                all_labels = True,
                                print_results = False
                                )

        
        selected_attributes = bias_attribute_selection_simple(gerry_eval_results = gerry_score, 
                                        attributes = ['gender', 'economy'],
                                        subgroup_min_size = 0,
                                        df = _results_df,
                                        key_mapping = bios_keymapping
                                        )

        max_violation = np.max([i[0] for i in selected_attributes['GAP_t11']])
        avg_violation = np.mean([i[0] for i in selected_attributes['GAP_t11']])
        valid_f1 = f1_score(_results_df['label'], _results_df['preds'], average='macro')
        print(valid_f1, max_violation, avg_violation)
       
        validation_results[iterv] = (avg_violation, valid_f1)

        print('--------------Test--------------------')
        y_preds = []
        y_probs = []
        
        tmp_preds = model.predict([xtest])
        for item_tmp in tmp_preds[0]:
            y_probs.append(item_tmp[0])
            y_preds.append(int(round(item_tmp[0])))

        print("fscore:", f1_score(ytest, y_preds, average='macro'))
        results = {'preds': y_preds, 'pred_prob': y_probs, 'gender': test_gender, 'economy': test_economy, 'label': ytest}
        results_df = pd.DataFrame(results)
        results_df.to_csv(tp_output_dir+'_adv_bios_mlp_results_tc_{}_{}.tsv'.format(iterv, LAMBDA_REVERSAL_STRENGTH), index=False)
    
    with open(tp_output_dir+'adv_bios_validation_tc_{}.pkl'.format(LAMBDA_REVERSAL_STRENGTH), 'wb') as f:
        pickle.dump(validation_results, f)
    return model

if __name__ == '__main__':
    
    xtrain  = np.load('/lt/work/shiva/Fairness/biasbios_location/emnlp_train_cls_tc.npy')
    xvalid  = np.load('/lt/work/shiva/Fairness/biasbios_location/emnlp_dev_cls_tc.npy')
    xtest  = np.load('/lt/work/shiva/Fairness/biasbios_location/emnlp_test_cls_tc.npy')

    with open('/lt/work/shiva/Fairness/biasbios_location/emnlp_train_bios_twoclass.pickle', 'rb') as f:
        traindata = pickle.load(f)
    
    with open('/lt/work/shiva/Fairness/biasbios_location/emnlp_dev_bios_twoclass.pickle', 'rb') as f:
        valdata = pickle.load(f)
    
    with open('/lt/work/shiva/Fairness/biasbios_location/emnlp_test_bios_twoclass.pickle', 'rb') as f:
        testdata = pickle.load(f)
    
    from collections import Counter

    print(xtrain.shape, xvalid.shape, xtest.shape)
    print(len(traindata), len(valdata), len(testdata))
    print(traindata[0].keys())
    print(set([d['economy'] for d in traindata]))

    ytrain = []
    yvalid = []
    ytest = []

    train_gender = []
    valid_gender = []
    test_gender = []

    train_economy = []
    valid_economy = []
    test_economy = []

    for data in traindata:
        _label = 1 if data['p'] == 'surgeon' else 0
        ytrain.append(_label)
        _gender = 1 if data['g'] == 'f' else 0
        train_gender.append(_gender)
        _economy = 1 if data['economy'] == 'High income (H)' else 0
        train_economy.append(_economy)
    
    for data in valdata:
        _label = 1 if data['p'] == 'surgeon' else 0
        yvalid.append(_label)
        _gender = 1 if data['g'] == 'f' else 0
        valid_gender.append(_gender)
        _economy = 1 if data['economy'] == 'High income (H)' else 0
        valid_economy.append(_economy)
    
    for data in testdata:
        _label = 1 if data['p'] == 'surgeon' else 0
        ytest.append(_label)
        _gender = 1 if data['g'] == 'f' else 0
        test_gender.append(_gender)
        _economy = 1 if data['economy'] == 'High income (H)' else 0
        test_economy.append(_economy)

    for LAMBDA_REVERSAL_STRENGTH in [0.0001, 0.001, 0.01, 0.05, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 95, 100]:
        model = build_mlp(xtrain, np.array(ytrain), np.array(train_gender), np.array(train_economy), 
                        xvalid, np.array(yvalid), np.array(valid_gender), np.array(valid_economy), 
                        xtest, np.array(ytest), np.array(test_gender), np.array(test_economy),
                        LAMBDA_REVERSAL_STRENGTH = LAMBDA_REVERSAL_STRENGTH
                        )
        


    



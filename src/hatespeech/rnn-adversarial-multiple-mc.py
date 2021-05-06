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
from gerry_eval import bias_attribute_selection, Gerrymandering_eval

tf.random.set_seed(1)
os.environ['PYTHONHASHSEED']=str(1)
np.random.seed(1)

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

    identity_flip_layer = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    identity_in = identity_flip_layer(dp)
    identity_out = Dense(
        units=16, activation='softmax', name='intersectional'
        )(identity_in)

    model = Model(inputs=text_input, outputs=[predicts, identity_out])#eth_out
    advmodel  = Model(inputs=text_input, outputs=identity_out)
    
    layer_names = ['hatepredict', 'intersectional']
    loss_dict = {}
    metrics_dict = {}
    adv_loss_dict = {}
    adv_metrics_dict = {}
    

    for l in layer_names: 
        if l == 'intersectional':
            loss_dict[l] = 'categorical_crossentropy'
            adv_loss_dict[l] = 'categorical_crossentropy'
            adv_metrics_dict[l] = 'accuracy'
        else:
            loss_dict[l] = 'binary_crossentropy'
        metrics_dict[l] = 'accuracy'

    model.compile(
        loss=loss_dict, optimizer='rmsprop',
        metrics=metrics_dict, loss_weights = [1, 1]
    )

    advmodel.compile(
        loss=adv_loss_dict, optimizer='rmsprop',
        metrics=adv_metrics_dict
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
        train_iter = evaluator.data_iter_mc_intersectional_adv(
            indices_dir+'train.tsv', batch_size=64
        )

        # train model
        for _, x_train, y_train, _, _, _, _, y_tr_identity in train_iter:
            if len(np.unique(y_train)) == 1:
                continue
            
            tmp = model.train_on_batch(
                [x_train], [y_train, y_tr_identity] #y_tr_ethnicity
            )

            """advtmp = advmodel.train_on_batch(
                [x_train], np.array(y_tr_identity) #y_tr_ethnicity
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
    
        selected_attributes = bias_attribute_selection(gerry_eval_results = gerry_score, 
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

        for _, x_test, y_test in test_iter:
            tmp_preds = best_model.predict([x_test])
            #print(tmp_preds[0].shape, tmp_preds[1].shape, len(tmp_preds))
            for item_tmp in tmp_preds[0]:
                y_probs.append(item_tmp[0])
                y_preds.append(int(round(item_tmp[0])))

        with open(odir+'hatespeech_adv_intersectional_{}_{}.tsv'.format(e, LAMBDA_REVERSAL_STRENGTH), 'w') as wfile:
            with open(indices_dir+'test.tsv') as dfile:
                wfile.write(
                    dfile.readline().strip()+'\tpred\tpred_prob\n')
                for idx, line in enumerate(dfile):
                    wfile.write(line.strip()+'\t'+str(y_preds[idx])+'\t'+str(y_probs[idx])+'\n')

        # save the predicted results
        evaluator.eval(
            odir+'hatespeech_adv_intersectional_{}_{}.tsv'.format(e, LAMBDA_REVERSAL_STRENGTH), 
            odir+'hatespeech_adv_intersectional_{}_{}.score'.format(e, LAMBDA_REVERSAL_STRENGTH)
        )
    
    with open(odir+'validation_hatespeech_adv_intersectional_{}.pkl'.format(LAMBDA_REVERSAL_STRENGTH), 'wb') as f:
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


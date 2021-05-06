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
import evaluator
import tensorflow as tf
import pandas as pd
from evaluator import *
import gensim
import pickle
import os
from collections import Counter

tf.random.set_seed(1)
os.environ['PYTHONHASHSEED']=str(1)
np.random.seed(1)

MAX_SEQUENCE_LENGTH = 150  
tp_output_dir = './data/tp/output/'
wt_dir = './data/tp/embeddings/'
res_dir = './resources/classifier/'

def build_rnn(traindata, valdata, testdata, tokenizer, odir=tp_output_dir):
    '''Train, valid, test RNN
        lang: The language name
        odir: output directory of prediction results
    '''
    rnn_size = 200
    epochs = 5
    
    weights = np.load(wt_dir+'tp.npy')

    # build model architecture
    text_input = Input(
        shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='input'
    )

    embeds = Embedding(
        weights.shape[0], weights.shape[1],
        weights=[weights], input_length=MAX_SEQUENCE_LENGTH,
        trainable=True, name='embedding'
    )(text_input)

    bigru = Bidirectional(GRU(
        rnn_size, kernel_initializer="glorot_uniform"), name = 'inlp'
    )(embeds)

    dp = Dropout(rate=.2)(bigru)

    predicts = Dense(
        1, activation='sigmoid', name='predict'
    )(dp) # binary prediction

    model = Model(inputs=text_input, outputs=predicts)
    repmodel  = Model(inputs=text_input, outputs=model.get_layer(name="inlp").output)
    
    model.compile(
        loss='binary_crossentropy', optimizer='rmsprop',
        metrics=['accuracy']
    )
    print(model.summary())

    best_valid_f1 = 0.0
    best_model = None
    best_rep_model = None

    for e in range(epochs):
        print('--------------Epoch: {}--------------'.format(e))

        # load training and batch dataset
        train_iter = evaluator.data_iter_tp(traindata, batch_size=64, tokenizer=tok, MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH)

        # train model
        for x_train, y_train, gender_train, country_train, supertopic_train in train_iter:
            if len(np.unique(y_train)) == 1:
                continue
            
            tmp = model.train_on_batch(
                [x_train], y_train
            )

        # valid model to find the best model
        print('---------------Validation------------')
        valid_iter = evaluator.data_iter_tp(valdata, batch_size=64, tokenizer=tok, MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH, if_shuffle=False)

        y_preds = []
        y_valids = []

        for x_valid, y_valid, _, _, _ in valid_iter:
            tmp_preds = model.predict([x_valid])
            for item_tmp in tmp_preds:
                y_preds.append(round(item_tmp[0]))
            y_valids.extend(y_valid)

        valid_f1 = f1_score(
            y_true=y_valids, y_pred=y_preds, 
            average='weighted',
        )
        print('Validating f1-macro score: ' + str(valid_f1))

        if best_valid_f1 < valid_f1:
            best_valid_f1 = valid_f1
            best_model = model
            best_rep_model = repmodel
            print("best epoch", e)

    print('--------------Test--------------------')
    y_preds = []
    y_probs = []
    test_gender = []
    test_country = []
    test_supertopic = []
    test_label = []

    train_iter = evaluator.data_iter_tp(traindata, batch_size=64, tokenizer=tok, MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH, if_shuffle=False)
    valid_iter = evaluator.data_iter_tp(valdata, batch_size=64, tokenizer=tok, MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH, if_shuffle=False)
    test_iter = evaluator.data_iter_tp(testdata, batch_size=64, tokenizer=tok, MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH, if_shuffle=False)

    ctrain = 0
    for x_train, _, _, _, _ in train_iter:
        if ctrain == 0:
            train_rep_preds = best_rep_model.predict([x_train])
        else:
            _rep_pred = best_rep_model.predict([x_train])
            train_rep_preds = np.append(train_rep_preds, _rep_pred, axis=0)
        ctrain+=1

    cval = 0
    for x_valid, _, _, _, _  in valid_iter:
        if cval == 0:
            val_rep_preds = best_rep_model.predict([x_valid])
        else:
            _rep_pred = best_rep_model.predict([x_valid])
            val_rep_preds = np.append(val_rep_preds, _rep_pred, axis=0)
        cval+=1

    ctest = 0
    for x_test, y_test, gender_test, country_test, st_test  in test_iter:
        if ctest == 0: 
            test_rep_preds = best_rep_model.predict([x_test])
        else:
            _rep_pred = best_rep_model.predict([x_test])
            test_rep_preds = np.append(test_rep_preds, _rep_pred, axis=0)
        tmp_preds = best_model.predict([x_test])
        for item_tmp in tmp_preds:
            y_probs.append(item_tmp[0])
            y_preds.append(int(round(item_tmp[0])))
        test_gender.extend(gender_test)
        test_country.extend(country_test)
        test_supertopic.extend(st_test)
        test_label.extend(y_test)
        ctest+=1

    print(train_rep_preds.shape, val_rep_preds.shape, test_rep_preds.shape)

    np.save(res_dir+'train_rep_tp.npy', train_rep_preds)
    np.save(res_dir+'valid_rep_tp.npy', val_rep_preds)
    np.save(res_dir+'test_rep_tp.npy', test_rep_preds)

    results = {'preds': y_preds, 'pred_prob': y_probs, 'gender': test_gender, 'country': test_country, 'supertopic': test_supertopic, 'label': test_label}
    results_df = pd.DataFrame(results)
    results_df.to_csv(odir+'_tp_results.tsv', index=False)


if __name__ == '__main__':
    with open('tp-subset-data-emnlp-6topics.pkl', 'rb') as f:
        data = pickle.load(f)

    tok_dir = './data/tp/tokenizer/'
   
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

    build_rnn(traindata, devdata, testdata, tokenizer = tok, odir=tp_output_dir)



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


tf.random.set_seed(1)
os.environ['PYTHONHASHSEED']=str(1)
np.random.seed(1)


def build_rnn(lang, odir):
    '''Train, valid, test RNN
    
        lang: The language name
        odir: output directory of prediction results
    '''
    doc_idx = 2
    rnn_size = 200
    max_len = 40 # sequence length
    epochs = 3

    encode_dir = './data/encode/'+lang+'/'
    indices_dir = './data/indices/'+lang+'/'
    wt_dir = './resources/weight/'
    res_dir = './resources/classifier/'

    
    # clf_path = res_dir + lang + '.clf'
    # don't reload classifier for debug usage

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
        train_iter = evaluator.data_iter(
            indices_dir+'train.tsv', batch_size=64
        )

        # train model
        for _, x_train, y_train in train_iter:
            if len(np.unique(y_train)) == 1:
                continue
            
            tmp = model.train_on_batch(
                [x_train], y_train
            )

        # valid model to find the best model
        print('---------------Validation------------')
        valid_iter = evaluator.data_iter(
            indices_dir+'valid.tsv', batch_size=64,
            if_shuffle=False
        )
        y_preds = []
        y_valids = []

        for _, x_valid, y_valid in valid_iter:
            tmp_preds = model.predict([x_valid])
            for item_tmp in tmp_preds:
                y_preds.append(round(item_tmp[0]))
            y_valids.extend(y_valid)

        valid_f1 = f1_score(
            y_true=y_valids, y_pred=y_preds, 
            average='macro',
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

    train_iter = evaluator.data_iter(
            indices_dir+'train.tsv', batch_size=64, 
            if_shuffle=False
        )

    valid_iter = evaluator.data_iter(
            indices_dir+'valid.tsv', batch_size=64,
            if_shuffle=False
        )

    test_iter = evaluator.data_iter(
        indices_dir+'test.tsv', batch_size=64,
        if_shuffle=False
    )

    ctrain = 0
    for _, x_train, y_train in train_iter:
        if ctrain == 0:
            train_rep_preds = best_rep_model.predict([x_train])
        else:
            _rep_pred = best_rep_model.predict([x_train])
            train_rep_preds = np.append(train_rep_preds, _rep_pred, axis=0)
        ctrain+=1

    cval = 0
    for _, x_valid, y_valid in valid_iter:
        if cval == 0:
            val_rep_preds = best_rep_model.predict([x_valid])
        else:
            _rep_pred = best_rep_model.predict([x_valid])
            val_rep_preds = np.append(val_rep_preds, _rep_pred, axis=0)
        cval+=1

    ctest = 0
    for _, x_test, y_test in test_iter:
        if ctest == 0: 
            test_rep_preds = best_rep_model.predict([x_test])
        else:
            _rep_pred = best_rep_model.predict([x_test])
            test_rep_preds = np.append(test_rep_preds, _rep_pred, axis=0)
        tmp_preds = best_model.predict([x_test])
        for item_tmp in tmp_preds:
            y_probs.append(item_tmp[0])
            y_preds.append(int(round(item_tmp[0])))
        ctest+=1

    print(train_rep_preds.shape, val_rep_preds.shape, test_rep_preds.shape)

    np.save(res_dir+'train_rep.npy', train_rep_preds)
    np.save(res_dir+'valid_rep.npy', val_rep_preds)
    np.save(res_dir+'test_rep.npy', test_rep_preds)

    with open(odir+lang+'.tsv', 'w') as wfile:
        with open(indices_dir+'test.tsv') as dfile:
            wfile.write(
                dfile.readline().strip()+'\tpred\tpred_prob\n')
            for idx, line in enumerate(dfile):
                wfile.write(line.strip()+'\t'+str(y_preds[idx])+'\t'+str(y_probs[idx])+'\n')

    # save the predicted results
    evaluator.eval(
        odir+lang+'.tsv', 
        odir+lang+'.score'
    )


if __name__ == '__main__':
    #langs = [
    #    'English', 'Italian', 'Polish', 
    #    'Portuguese', 'Spanish'
    #]
    langs = ['English']
    
    odir = './results/rnn/'
    if not os.path.exists(odir):
        os.mkdir(odir)

    for lang in langs:
        build_rnn(lang, odir)


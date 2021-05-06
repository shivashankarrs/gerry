import os
import sys
import tempfile
import urllib
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
import numpy as np
import tensorflow_constrained_optimization as tfco
from tensorflow_metadata.proto.v0 import schema_pb2
from tfx_bsl.tfxio import tensor_adapter
from tfx_bsl.tfxio import tf_example_record
import tensorflow_model_analysis as tfma
import fairness_indicators as fi
from google.protobuf import text_format
import apache_beam as beam
from sklearn.metrics import f1_score
from tensorflow.keras.layers import Input, Embedding
from tensorflow.keras.layers import Bidirectional, GRU
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

import pandas as pd 

import sys
sys.path.append('../')
import evaluator, random, math, pickle
from gerry_eval import bias_attribute_selection_simple, Gerrymandering_eval

from keras.utils import to_categorical
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
print("TFMA " + tfma.VERSION_STRING)
print("TFDS " + tfds.version.__version__)
print("FI " + fi.version.__version__)


max_len = 150  
BATCH_SIZE = 128

sys.path.append("/home/sssub/intersectional_bias/Multilingual_Fairness_LREC-master/")
sys.path.append("/home/sssub/intersectional_bias/Multilingual_Fairness_LREC-master/outputs/")

odir = '/home/sssub/intersectional_bias/Multilingual_Fairness_LREC-master/publish/results/TP/TFCO/'
wt_dir = '/home/sssub/intersectional_bias/Multilingual_Fairness_LREC-master/data/tp/embeddings/'
tok_dir = '/home/sssub/intersectional_bias/Multilingual_Fairness_LREC-master/data/tp/tokenizer/'

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
    
    inter_data = {}
    labels = {}

    traindata_pd = create_data(traindata)
    for _g, _c, _s, _y in zip(traindata_pd['gender'], traindata_pd['country'], traindata_pd['supertopic'], traindata_pd['y']):
        l = (int(_g), int(_c), int(_s))
        if int(_g) == 0:
            if int(_y) == 1: labels[0] += 1
            inter_data[0] += 1
        elif int(_g) == 1:
            if int(_y) == 1: labels[1] += 1
            inter_data[1] += 1

        if int(_c) == 0:
            if int(_y) == 1: labels[2] += 1
            inter_data[2] += 1
        elif int(_c) == 1:
            if int(_y) == 1: labels[3] += 1
            inter_data[3] += 1

        if int(_s) == 0:
            if int(_y) == 1: labels[4] += 1
            inter_data[4] += 1
        elif int(_s) == 1:
            if int(_y) == 1: labels[5] += 1
            inter_data[5] += 1
        elif int(_s) == 2:
            if int(_y) == 1: labels[6] += 1
            inter_data[6] += 1
        elif int(_s) == 3:
            if int(_y) == 1: labels[7] += 1
            inter_data[7] += 1
        elif int(_s) == 4:
            if int(_y) == 1: labels[8] += 1
            inter_data[8] += 1
        elif int(_s) == 5:
            if int(_y) == 1: labels[9] += 1
            inter_data[9] += 1


    for k,v in inter_data.items():
        inter_data[k] = inter_data[k]/labels[k]

    constraints = []

    contexts_available = [context_subset_0, context_subset_1, context_subset_2, context_subset_3, context_subset_4, context_subset_5, context_subset_6, context_subset_7, context_subset_8, \
                        context_subset_9]

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
        _lmean = 0
        _ctr=0

        train_iter = evaluator.data_iter_tp(traindata, batch_size=BATCH_SIZE, tokenizer=tok, MAX_SEQUENCE_LENGTH=max_len)
        # train model
        for x_train, y_train, gender_train, country_train, supertopic_train in train_iter:
            if x_train.shape[0] < BATCH_SIZE: continue
            
            train_priv_inter = []
            for g, c, s in zip(gender_train, country_train, supertopic_train):
                train_priv_inter.append(inter_enc[(g, c, s)])

            gender_train = to_categorical(gender_train, num_classes=2)
            country_train = to_categorical(country_train, num_classes=2)
            supertopic_train = to_categorical(supertopic_train, num_classes=6)

            input_tensor.assign(x_train)  
            labels_tensor.assign(y_train)
            groups_tensor_0.assign(np.array(gender_train[:, 0]))
            groups_tensor_1.assign(np.array(gender_train[:, 1]))
            groups_tensor_2.assign(np.array(country_train[:, 0]))
            groups_tensor_3.assign(np.array(country_train[:, 1]))
            groups_tensor_4.assign(np.array(supertopic_train[:, 0]))
            groups_tensor_5.assign(np.array(supertopic_train[:, 1]))
            groups_tensor_6.assign(np.array(supertopic_train[:, 2]))
            groups_tensor_7.assign(np.array(supertopic_train[:, 3]))
            groups_tensor_8.assign(np.array(supertopic_train[:, 4]))
            groups_tensor_9.assign(np.array(supertopic_train[:, 5]))
           
            optimizer.minimize(problem, var_list=var_list)
            objective = problem.objective()
            violations = problem.constraints()
            _lmean += np.mean(violations)
            _ctr+=1
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
        results_df.to_csv(odir+'TP_RNN_TFCO_independent_tw_cw_{}_results_{}_{}.tsv'.format(weighted, iterv, nu), index=False)

    with open(odir+'validation_results_TP_TFCO_independent_tw_cw_{}_{}.pkl'.format(weighted, nu), 'wb') as f:
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

    for weighted in [True, False]:
    #for weighted in [False]:
        for nu in [0.0001, 0.0005, 0.0007, 0.001, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1]:
            model = constrained_optimization(traindata, devdata, testdata, tok, 
                                            nu = nu, weighted = weighted
                                            )
    



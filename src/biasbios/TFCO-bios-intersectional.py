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

import sys
sys.path.append('../')

import evaluator, random, math, pickle
import pandas as pd 
from gerry_eval import bias_attribute_selection_simple, Gerrymandering_eval
from keras.utils import to_categorical
import pickle

from tensorflow_constrained_optimization.python.rates import basic_expression
from tensorflow_constrained_optimization.python.rates import binary_rates
from tensorflow_constrained_optimization.python.rates import defaults
from tensorflow_constrained_optimization.python.rates import deferred_tensor
from tensorflow_constrained_optimization.python.rates import expression
from tensorflow_constrained_optimization.python.rates import multiclass_rates
from tensorflow_constrained_optimization.python.rates import subsettable_context
from tensorflow_constrained_optimization.python.rates import term
from tensorflow_constrained_optimization.python.rates import loss

from scipy.stats import hmean

tf.random.set_seed(1)
os.environ['PYTHONHASHSEED']=str(1)
np.random.seed(1)

print("TensorFlow " + tf.__version__)
print("TFMA " + tfma.VERSION_STRING)
print("TFDS " + tfds.version.__version__)
print("FI " + fi.version.__version__)

BATCH_SIZE = 64
MAX_SEQUENCE_LENGTH = 768  
odir = '/home/sssub/intersectional_bias/Multilingual_Fairness_LREC-master/publish/results/biasbios/tfco/'

bios_keymapping = {
    'gender':{0:"male", 1:"female"}, 
    'economy': {1: 'High economy',
                0: 'Rest'}
}

def build_mlp():
    '''Train, valid, test RNN
        lang: The language name
        odir: output directory of prediction results
    '''
    hidden_size = 300
    
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

def intersectional_label_encoder(gender, economy):
    encoder = {}
    indx = 0

    for g in list(set(gender)):
        for c in list(set(economy)):
            l = (g, c)
            encoder[l] = indx
            indx+=1
    return encoder

def constrained_optimization(xtrain, ytrain, train_gender, train_economy, xvalid, yvalid, 
                            valid_gender, valid_economy, xtest, ytest, test_gender, test_economy,
                            nu=0.0001, weighted=True):

    model_constrained = build_mlp()

    def predictions():
        return model_constrained(input_tensor)

    input_tensor = tf.Variable(
        np.zeros((BATCH_SIZE, MAX_SEQUENCE_LENGTH), dtype="float32"),
        name="input")

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


    context = tfco.rate_context(predictions, lambda: labels_tensor)
    objective = custom_error_rate(context)

    context_subset_0 = context.subset(lambda:groups_tensor_0 >0)
    context_subset_1 = context.subset(lambda:groups_tensor_1 >0)
    context_subset_2 = context.subset(lambda:groups_tensor_2 >0)
    context_subset_3 = context.subset(lambda:groups_tensor_3 >0)

    inter_enc = intersectional_label_encoder([0,1], [0,1])

    inter_data = {}
    labels = {}

    for _g, _e, _y in zip(train_gender, train_economy, ytrain):
        l = (int(_g), int(_e))

        if int(_y) == 1: 
            if inter_enc[l] not in labels: labels[inter_enc[l]] = 1
            else: labels[inter_enc[l]] += 1
        if inter_enc[l] not in inter_data: 
            inter_data[inter_enc[l]] = 1
        else: 
            inter_data[inter_enc[l]] += 1

    for k,v in inter_data.items():
        inter_data[k] = inter_data[k]/labels[k]

    constraints = []

    contexts_available = [context_subset_0, context_subset_1, context_subset_2, context_subset_3]

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

    NUM_ITERATIONS = 500 # Number of training iterations.
    objective_list = []
    violations_list = []
    iterv = 0

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    best_hmean_v = 0
    best_hmean_epoch = 0
    best_bias_v = 1
    best_bias_epoch = 0
    validation_results = {}
    best_perf = 0
    best_perf_epoch = 0

    while iterv < NUM_ITERATIONS:
        train_iter = evaluator.data_iter_bios(xtrain, ytrain, train_gender, train_economy, batch_size=BATCH_SIZE)
            
        # train model
        for x_train, y_train, gender_train, economy_train in train_iter:

            if x_train.shape[0] < BATCH_SIZE: continue
            train_priv_inter = []

            for g, e in zip(gender_train, economy_train):
                train_priv_inter.append(inter_enc[(g, e)])

            y_tr_intersectional = to_categorical(train_priv_inter, num_classes=4)

            input_tensor.assign(x_train)  
            labels_tensor.assign(y_train)
            groups_tensor_0.assign(np.array(y_tr_intersectional[:, 0]))
            groups_tensor_1.assign(np.array(y_tr_intersectional[:, 1]))
            groups_tensor_2.assign(np.array(y_tr_intersectional[:, 2]))
            groups_tensor_3.assign(np.array(y_tr_intersectional[:, 3]))
        
            optimizer.minimize(problem, var_list=var_list)
            objective = problem.objective()
            violations = problem.constraints()
            print(objective, np.mean(violations), np.max(violations))

        objective_list.append(objective)
        violations_list.append(violations)
        
        # valid model to find the best model
        print('---------------Validation------------')
        valid_iter = evaluator.data_iter_bios(xvalid, yvalid, valid_gender, valid_economy, batch_size=BATCH_SIZE, if_shuffle=False)

        y_preds = []
        y_valids = []
        y_pred_prob = []
        _valid_gender = []
        _valid_economy = []
        
        for x_valid, y_valid, y_va_gender, y_va_economy in valid_iter:
            tmp_preds = model_constrained.predict([x_valid])
            for item_tmp in tmp_preds:
                y_preds.append(int(round(sigmoid(item_tmp[0]))))
                y_pred_prob.append(sigmoid(item_tmp[0]))
            y_valids.extend(y_valid)
            _valid_gender.extend(y_va_gender)
            _valid_economy.extend(y_va_economy)
        
        valid_f1 = f1_score(
            y_true=y_valids, y_pred=y_preds, 
            average='macro',
        )
        print('Validating f1-macro score: ' + str(valid_f1))

        results = {'pred': y_preds, 'pred_prob': y_pred_prob, 'gender': _valid_gender, 
                    'economy': _valid_economy, 'label': y_valids}

        results_df = pd.DataFrame(results)

        gerry_score =   Gerrymandering_eval(results_df,
                                        tasks = ['gender', 'economy'],
                                        label = "label", 
                                        pred="pred",
                                        pred_prob = "pred_prob",
                                        all_labels = True,
                                        print_results = False
                                        )

        selected_attributes = bias_attribute_selection_simple(gerry_eval_results = gerry_score, 
                                                attributes = ['gender', 'economy'],
                                                subgroup_min_size = 0,
                                                df = results_df,
                                                key_mapping = bios_keymapping
                                                )

        avg_violation = np.mean([i[0] for i in selected_attributes['GAP_t11']])
        max_violation = np.max([i[0] for i in selected_attributes['GAP_t11']])
        validation_results[iterv] = (avg_violation, valid_f1)
        print(iterv, valid_f1, avg_violation, max_violation)
            
        if hmean([valid_f1, 1-avg_violation]) > best_hmean_v:
            best_hmean_epoch = iterv
            best_hmean_v = hmean([valid_f1, 1-avg_violation])
        
        if valid_f1 > best_perf:
            best_perf_epoch = iterv
            best_perf = valid_f1

        if avg_violation < best_bias_v:
            best_bias_epoch = iterv
            best_bias_v = avg_violation

        print(iterv-best_bias_epoch, iterv, valid_f1, avg_violation, max_violation)
        if iterv-best_bias_epoch >= 5:
            print ("check this model: 5", best_bias_epoch, iterv-best_bias_epoch)
        if iterv-best_bias_epoch >= 10:
            print ("check this model: 10", best_bias_epoch, iterv-best_bias_epoch)
        
        iterv += 1
        
        print('--------------Test--------------------')

        y_preds = []
        y_probs = []
        _test_gender = []
        _test_economy = []
        test_label = []

        test_iter = evaluator.data_iter_bios(xtest, ytest, test_gender, test_economy, batch_size=BATCH_SIZE, if_shuffle=False)

        for x_test, y_test, gender_test, economy_test in test_iter:
            tmp_preds = model_constrained.predict([x_test])
            for item_tmp in tmp_preds:
                y_probs.append(sigmoid(item_tmp[0]))
                y_preds.append(int(round(sigmoid(item_tmp[0]))))
            _test_gender.extend(gender_test)
            _test_economy.extend(economy_test)
            test_label.extend(y_test)
        
        results = {'preds': y_preds, 'pred_prob': y_probs, 'gender': _test_gender, 'economy': _test_economy, 'label': test_label}
        results_df = pd.DataFrame(results)
        results_df.to_csv(odir+'Bios_MLP_TFCO_Intersection_tw_cw_{}_results_{}_{}.tsv'.format(weighted, iterv, nu), index=False)

    with open(odir+'validation_results_Bios_MLP_TFCO_Intersectional_tw_cw_{}_{}.pkl'.format(weighted, nu), 'wb') as f:
        pickle.dump(validation_results, f)
        
    best_index = tfco.find_best_candidate_index(
        np.array(objective_list), np.array(violations_list), rank_objectives=False)
    print (best_index, "nu:{}".format(nu), "weighted:{}".format(weighted), "best index from TFCO, not ranked")

    best_index = tfco.find_best_candidate_index(
        np.array(objective_list), np.array(violations_list), rank_objectives=True)
    print (best_index, "nu:{}".format(nu), "weighted:{}".format(weighted), "best index from TFCO, ranked")

    print("best epochs", best_perf_epoch, best_hmean_epoch, best_bias_epoch)

# Setup list of constraints.
# In this notebook, the constraint will just be: FPR to less or equal to 5%.

if __name__ == '__main__':

    xtrain  = np.load('/lt/work/shiva/Fairness/biasbios_location/emnlp_train_cls_tc.npy')
    xvalid  = np.load('/lt/work/shiva/Fairness/biasbios_location/emnlp_dev_cls_tc.npy')
    xtest  = np.load('/lt/work/shiva/Fairness/biasbios_location/emnlp_test_cls_tc.npy')

    with open('/home/sssub/aidalight-backup/data/emnlp_train_bios_twoclass.pickle', 'rb') as f:
        traindata = pickle.load(f)

    with open('/home/sssub/aidalight-backup/data/emnlp_dev_bios_twoclass.pickle', 'rb') as f:
        valdata = pickle.load(f)

    with open('/home/sssub/aidalight-backup/data/emnlp_test_bios_twoclass.pickle', 'rb') as f:
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

    for weighted in [True, False]:
        for nu in [0.0001, 0.0005, 0.0007, 0.001, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1]:
            model = constrained_optimization(xtrain, np.array(ytrain), np.array(train_gender), np.array(train_economy), 
                            xvalid, np.array(yvalid), np.array(valid_gender), np.array(valid_economy), 
                            xtest, np.array(ytest), np.array(test_gender), np.array(test_economy),
                            nu = nu, weighted = weighted
                            )
    

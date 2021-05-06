'''Scripts for evaluation,
    metrics: (macro) F1, AUC, FNED, FPED

    Because the data is skewed distributed, therefore,
    we use the macro f1 score to measure the performance.
'''
import pandas as pd
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle
import numpy as np
from keras.utils import to_categorical
import json
from collections import Counter
from gerry_eval import *
from utils import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences



def cal_fpr(fp, tn):
    '''False positive rate'''
    return fp/(fp+tn)


def cal_fnr(fn, tp):
    '''False negative rate'''
    return fn/(fn+tp)


def cal_tpr(tp, fn):
    '''True positive rate'''
    return tp/(tp+fn)


def cal_tnr(tn, fp):
    '''True negative rate'''
    return tn/(tn+fp)


def eval(dpath, opt):
    '''Fairness Evaluation
        dpath: input eval file path
        opt: output results path
    '''
    df = pd.read_csv(dpath, sep='\t', na_values='x')
    # get the task name from the file, gender or ethnicity
    tasks = ['gender', 'age', 'country', 'ethnicity']

    scores = {
        'accuracy': 0.0,
        'f1-macro': 0.0, # macro f1 score
        'f1-weight': 0.0, # weighted f1 score
        'auc': 0.0,
    }

    # accuracy, f1, auc
    scores['accuracy'] = metrics.accuracy_score(
        y_true=df.label, y_pred=df.pred
    )
    scores['f1-macro'] = metrics.f1_score(
        y_true=df.label, y_pred=df.pred,
        average='macro'
    )
    scores['f1-weight'] = metrics.f1_score(
        y_true=df.label, y_pred=df.pred,
        average='weighted'
    )
    fpr, tpr, _ = metrics.roc_curve(
        y_true=df.label, y_score=df.pred_prob,
    )
    scores['auc'] = metrics.auc(fpr, tpr)

    '''fairness gaps'''
    for task in tasks:

        '''Filter out some tasks'''
        if ('Polish' in dpath or 'Italian' in dpath) and\
             task in ['country', 'ethnicity']:
            continue

        scores[task] = {
            'fned': 0.0, # gap between fnr
            'fped': 0.0, # gap between fpr
            'tped': 0.0, # gap between tpr
            'tned': 0.0, # gap between tnr
        }
        # filter out the one does not have attributes
        task_df = df[df[task].notnull()]
    
        # get overall confusion matrix
        tn, fp, fn, tp = metrics.confusion_matrix(
            y_true=task_df.label, y_pred=task_df.pred
        ).ravel()

        # get the unique types of demographic groups
        uniq_types = task_df[task].unique()

        #print(task, uniq_types)
        
        for group in uniq_types:
            # calculate group specific confusion matrix
            group_df = task_df[task_df[task] == group]
            
            g_tn, g_fp, g_fn, g_tp = metrics.confusion_matrix(
                y_true=group_df.label, y_pred=group_df.pred
            ).ravel()

            # calculate and accumulate the gaps
            scores[task]['fned'] = scores[task]['fned'] + abs(
                cal_fnr(fn, tp)-cal_fnr(g_fn, g_tp)
            )
            scores[task]['fped'] = scores[task]['fped'] + abs(
                cal_fpr(fp, tn)-cal_fpr(g_fp, g_tn)
            )
            scores[task]['tped'] = scores[task]['tped'] + abs(
                cal_tpr(tp, fn)-cal_tpr(g_tp, g_fn)
            )
            scores[task]['tned'] = scores[task]['tned'] + abs(
                cal_tnr(tn, fp)-cal_tnr(g_tn, g_fp)
            )
    with open(opt, 'w') as wfile:
        wfile.write(json.dumps(scores))
    print(scores)

def text_to_seq(text, tokenizer, MAX_SEQUENCE_LENGTH=150):
    return pad_sequences(tokenizer.texts_to_sequences(text), maxlen = MAX_SEQUENCE_LENGTH) 

def data_iter_tp(datap, batch_size, tokenizer, MAX_SEQUENCE_LENGTH = 150, if_shuffle=True, if_sample=False):

    topics_lookup = {'Computer & Accessories': 0,
                'Fashion Accessories': 1,
                'Fitness & Nutrition': 2,
                'Tires': 3,
                'Hotels': 4,
                'Pets': 5}

    doc_idx = 2
    data = {'x': [], 'y': [], 'gender': [], 'country': [], 'supertopic': []}
    text = []
    for line in datap:
        text.append(line['title'].strip() + ' ' + line['text'].strip())

    textseq = text_to_seq(text, tokenizer, MAX_SEQUENCE_LENGTH)

    data['x'] = textseq

    for d in datap:
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

    print(len(data['x']), len(data['gender']), len(data['country']))
            
    # if shuffle the dataset
    if if_shuffle:
        data['x'], data['y'], d['gender'], d['country'], data['supertopic'] = shuffle(data['x'], data['y'], data['gender'], data['country'], data['supertopic'])

    steps = len(data['x']) // batch_size
    if len(data['x']) % batch_size != 0:
        steps += 1
        
    for step in range(steps):
        yield np.asarray(data['x'][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data['y'][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data['gender'][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data['country'][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data['supertopic'][step*batch_size: (step+1)*batch_size])
         

def data_iter(datap, batch_size=64, if_shuffle=True, if_sample=False):
    doc_idx = 2
    data = {'x': [], 'y': []}
    class_wt = dict()
    
    with open(datap) as dfile:
        dfile.readline()
        for line in dfile:
            line = line.strip().split('\t')
            # split indices
            data['x'].append(list(map(int, line[doc_idx].split())))
            data['y'].append(int(line[-1]))

    # if over sample the minority    
    if if_sample:
        label_count = Counter(data['y'])
        for label_tmp in label_count:
            sample_num = label_count.most_common(1)[0][1] - label_count[label_tmp]
            if sample_num == 0:
                continue
            sample_indices = np.random.choice(
                list(range(len(data['y']))),
                size=sample_num
            )
            for idx in sample_indices:
                data['x'].append(data['x'][idx])
                data['y'].append(data['y'][idx])
            
    # calculate the class weight
    class_wt = dict(zip(
        np.unique(data['y']), compute_class_weight(
            'balanced', np.unique(data['y']), 
            data['y']
        )
    ))

    # if shuffle the dataset
    if if_shuffle:
        data['x'], data['y'] = shuffle(data['x'], data['y'])

    steps = len(data['x']) // batch_size
    if len(data['x']) % batch_size != 0:
        steps += 1

    for step in range(steps):
        yield class_wt, \
            np.asarray(data['x'][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data['y'][step*batch_size: (step+1)*batch_size])

def data_iter_adv(datap, batch_size=64, if_shuffle=True, if_sample=False):
    doc_idx = 2
    data = {'x': [], 'y': [], 'gender': [], 'age': [], 'country': [], 'ethnicity': []}
    class_wt = dict()
    
    with open(datap) as dfile:
        dfile.readline()
        for line in dfile:
            line = line.strip().split('\t')
            # split indices
            data['x'].append(list(map(int, line[doc_idx].split())))
            data['y'].append(int(line[-1]))
            data['gender'].append(int(line[4])) #gender
            data['age'].append(int(line[5])) #age
            data['country'].append(int(line[8])) #country
            data['ethnicity'].append(int(line[9])) #ethnicity

    # if shuffle the dataset
    if if_shuffle:
        data['x'], data['y'], data['gender'], data['age'], data['country'], data['ethnicity'] = shuffle(data['x'], data['y'], data['gender'], data['age'], data['country'], data['ethnicity'])

    steps = len(data['x']) // batch_size
    if len(data['x']) % batch_size != 0:
        steps += 1

    for step in range(steps):
        yield class_wt, \
            np.asarray(data['x'][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data['y'][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data['gender'][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data['age'][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data['country'][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data['ethnicity'][step*batch_size: (step+1)*batch_size])


def data_iter_bios(xtrain, ytrain, train_gender, train_economy, batch_size=64, if_shuffle=True):
    data = {'x': [], 'y': [], 'gender': [], 'economy': []}
    
    for i in range(len(ytrain)):
        data['x'].append(xtrain[i, :])
    data['y'] = ytrain
    data['gender'] = train_gender
    data['economy'] = train_economy

    if if_shuffle:
        data['x'], data['y'], data['gender'], data['economy'] = shuffle(data['x'], data['y'], data['gender'], data['economy'])

    steps = len(data['x']) // batch_size
    if len(data['x']) % batch_size != 0:
        steps += 1

    for step in range(steps):
        yield np.asarray(data['x'][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data['y'][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data['gender'][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data['economy'][step*batch_size: (step+1)*batch_size])
            
            
def intersectional_label_encoder(gender, age, country, ethnicity):
    encoder = {}
    indx = 0

    for g in list(set(gender)):
        for a in list(set(age)):
            for c in list(set(country)):
                for e in list(set(ethnicity)):
                    l = (g, a, c, e)
                    encoder[l] = indx
                    indx+=1
    return encoder


def data_iter_mc_intersectional_adv(datap, batch_size=64, if_shuffle=True, if_sample=False):
    doc_idx = 2
    data = {'x': [], 'y': [], 'gender': [], 'age': [], 'country': [], 'ethnicity': [], 'intersectional': []}
    class_wt = dict()
    
    inter_enc = intersectional_label_encoder([0,1], [0,1], [0,1], [0,1])

    with open(datap) as dfile:
        dfile.readline()
        for line in dfile:
            line = line.strip().split('\t')
            # split indices
            data['x'].append(list(map(int, line[doc_idx].split())))
            data['y'].append(int(line[-1]))
            data['gender'].append(int(line[4])) #gender
            data['age'].append(int(line[5])) #age
            data['country'].append(int(line[8])) #country
            data['ethnicity'].append(int(line[9])) #ethnicity
            l = (int(line[4]), int(line[5]), int(line[8]), int(line[9]))
            data['intersectional'].append(inter_enc[l])

    # if shuffle the dataset
    if if_shuffle:
        data['x'], data['y'], data['gender'], data['age'], data['country'], data['ethnicity'], data['intersectional'] = shuffle(data['x'], data['y'], data['gender'], data['age'], data['country'], data['ethnicity'], data['intersectional'])

    steps = len(data['x']) // batch_size
    if len(data['x']) % batch_size != 0:
        steps += 1

    print(data['intersectional'][0:10], "before")
    data['intersectional'] = to_categorical(data['intersectional'])

    print(data['intersectional'][0:10], "after")
    
    for step in range(steps):
        yield class_wt, \
            np.asarray(data['x'][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data['y'][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data['gender'][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data['age'][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data['country'][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data['ethnicity'][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data['intersectional'][step*batch_size: (step+1)*batch_size])

def data_iter_TP_gerry_adv(datap, batch_size, tokenizer, MAX_SEQUENCE_LENGTH = 150, if_shuffle=True):
    data = {'x': [], 'y': [], 'gender': [], 'country': [], 'supertopic': []}

    topics_lookup = {'Computer & Accessories': 0,
                'Fashion Accessories': 1,
                'Fitness & Nutrition': 2,
                'Tires': 3,
                'Hotels': 4,
                'Pets': 5}

    """text = []
    for line in datap:
        text.append(line['title'].strip() + ' ' + line['text'].strip())

    textseq = text_to_seq(text, tokenizer, MAX_SEQUENCE_LENGTH)"""

    
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

    _data = pd.DataFrame(data)

    tasks = ['gender', 'country', 'supertopic']
    attribute_distinct_labels = {attribute:list(_data[~_data[attribute].isnull()][attribute].unique()) for attribute in tasks}
    print (attribute_distinct_labels)
    gerry_combs = Gerrymandering_groups(
        attributes = tasks, 
        attribute_distinct_labels = attribute_distinct_labels
        )
    group_comb_itr = 0
    # iterate all gerry combs
    for task_comb, group_comb in tqdm(gerry_combs):
        group_indices = task_comb_data(_data, task_comb, group_comb)
        group_key = "@".join([str(i)+str(j) for i,j in zip(task_comb, group_comb)])
        _labels = [0] * len(_data)
        lind = 0
        for gid, y in zip(group_indices, _labels): 
            if gid: _labels[lind] = 1
            else: _labels[lind] = 0
            lind+=1
        data["gerry_class_{}".format(group_comb_itr)] = _labels
        group_comb_itr+=1
    
    #data['x'] = textseq

    # if shuffle the dataset
    if if_shuffle:
        (data['x'], data['y'], data["gerry_class_0"], data["gerry_class_1"], data["gerry_class_2"], data["gerry_class_3"], data["gerry_class_4"], data["gerry_class_5"], data["gerry_class_6"], 
        data["gerry_class_7"], data["gerry_class_8"], data["gerry_class_9"], data["gerry_class_10"], data["gerry_class_11"], data["gerry_class_12"], data["gerry_class_13"], data["gerry_class_14"], data["gerry_class_15"], data["gerry_class_16"], 
        data["gerry_class_17"], data["gerry_class_18"], data["gerry_class_19"], data["gerry_class_20"], data["gerry_class_21"], data["gerry_class_22"], data["gerry_class_23"], data["gerry_class_24"], data["gerry_class_25"], 
        data["gerry_class_26"], data["gerry_class_27"], data["gerry_class_28"], data["gerry_class_29"], data["gerry_class_30"], data["gerry_class_31"], data["gerry_class_32"], data["gerry_class_33"], data["gerry_class_34"], 
        data["gerry_class_35"], data["gerry_class_36"], data["gerry_class_37"], data["gerry_class_38"], data["gerry_class_39"], data["gerry_class_40"], data["gerry_class_41"], data["gerry_class_42"], data["gerry_class_43"], 
        data["gerry_class_44"], data["gerry_class_45"], data["gerry_class_46"], data["gerry_class_47"], data["gerry_class_48"], data["gerry_class_49"], data["gerry_class_50"], data["gerry_class_51"], data["gerry_class_52"], 
        data["gerry_class_53"], data["gerry_class_54"], data["gerry_class_55"], data["gerry_class_56"], data["gerry_class_57"], data["gerry_class_58"], data["gerry_class_59"], data["gerry_class_60"], data["gerry_class_61"], 
        ) = shuffle(data['x'], data['y'], data["gerry_class_0"], data["gerry_class_1"], data["gerry_class_2"], data["gerry_class_3"], 
        data["gerry_class_4"], data["gerry_class_5"], data["gerry_class_6"], data["gerry_class_7"], data["gerry_class_8"], data["gerry_class_9"], data["gerry_class_10"], data["gerry_class_11"], data["gerry_class_12"], 
        data["gerry_class_13"], data["gerry_class_14"], data["gerry_class_15"], data["gerry_class_16"], 
        data["gerry_class_17"], data["gerry_class_18"], data["gerry_class_19"], data["gerry_class_20"], data["gerry_class_21"], data["gerry_class_22"], data["gerry_class_23"], data["gerry_class_24"], data["gerry_class_25"], 
        data["gerry_class_26"], data["gerry_class_27"], data["gerry_class_28"], data["gerry_class_29"], data["gerry_class_30"], data["gerry_class_31"], data["gerry_class_32"], data["gerry_class_33"], data["gerry_class_34"], 
        data["gerry_class_35"], data["gerry_class_36"], data["gerry_class_37"], data["gerry_class_38"], data["gerry_class_39"], data["gerry_class_40"], data["gerry_class_41"], data["gerry_class_42"], data["gerry_class_43"], 
        data["gerry_class_44"], data["gerry_class_45"], data["gerry_class_46"], data["gerry_class_47"], data["gerry_class_48"], data["gerry_class_49"], data["gerry_class_50"], data["gerry_class_51"], data["gerry_class_52"], 
        data["gerry_class_53"], data["gerry_class_54"], data["gerry_class_55"], data["gerry_class_56"], data["gerry_class_57"], data["gerry_class_58"], data["gerry_class_59"], data["gerry_class_60"], data["gerry_class_61"])

    steps = len(data['x']) // batch_size
    if len(data['x']) % batch_size != 0:
        steps += 1
    for step in range(steps):
        yield np.asarray(data['x'][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data['y'][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_0"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_1"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_2"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_3"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_4"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_5"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_6"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_7"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_8"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_9"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_10"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_11"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_12"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_13"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_14"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_15"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_16"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_17"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_18"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_19"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_20"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_21"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_22"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_23"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_24"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_25"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_26"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_27"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_28"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_29"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_30"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_31"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_32"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_33"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_34"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_35"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_36"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_37"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_38"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_39"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_40"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_41"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_42"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_43"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_44"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_45"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_46"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_47"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_48"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_49"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_50"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_51"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_52"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_53"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_54"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_55"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_56"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_57"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_58"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_59"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_60"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_61"][step*batch_size: (step+1)*batch_size])
        
def data_iter_gerry_adv(datap, batch_size=64, if_shuffle=True, if_sample=False):
    '''_data = {'x': [], 'y': [], 'gender': [], 'age': [], 'country': [], 'ethnicity': [], 'gerry_class_0': [], 'gerry_class_1': [], 
    'gerry_class_2':[],'gerry_class_3':[],'gerry_class_4':[],'gerry_class_5':[],'gerry_class_6':[], 'gerry_class_7':[],'gerry_class_8':[],
    'gerry_class_9':[],'gerry_class_10':[],'gerry_class_11':[],'gerry_class_12':[],'gerry_class_13':[],'gerry_class_14':[],'gerry_class_15':[],
    'gerry_class_16':[], 'gerry_class_17':[],'gerry_class_18':[],'gerry_class_19':[],'gerry_class_20':[],'gerry_class_21':[],'gerry_class_22':[],
    'gerry_class_23':[],'gerry_class_24':[],'gerry_class_25':[], 'gerry_class_26':[],'gerry_class_27':[],'gerry_class_28':[],'gerry_class_29':[],
    'gerry_class_30':[],'gerry_class_31':[],'gerry_class_32':[],'gerry_class_33':[],'gerry_class_34':[], 
    'gerry_class_35':[],'gerry_class_36':[],'gerry_class_37':[],'gerry_class_38':[],'gerry_class_39':[],'gerry_class_40':[],'gerry_class_41':[],
    'gerry_class_42':[],'gerry_class_43':[], 'gerry_class_44':[],'gerry_class_45':[],'gerry_class_46':[],'gerry_class_47':[],'gerry_class_48':[],
    'gerry_class_49':[],'gerry_class_50':[],'gerry_class_51':[],'gerry_class_52':[], 'gerry_class_53':[],'gerry_class_54':[],'gerry_class_55':[],'gerry_class_56':[],
    'gerry_class_57':[],'gerry_class_58':[],'gerry_class_59':[],'gerry_class_60':[],'gerry_class_61':[], 'gerry_class_62':[],'gerry_class_63':[],'gerry_class_64':[],
    'gerry_class_65':[],'gerry_class_66':[],'gerry_class_67':[],'gerry_class_68':[],'gerry_class_69':[],'gerry_class_70':[], 
    'gerry_class_71':[],'gerry_class_72':[],'gerry_class_73':[],'gerry_class_74':[],'gerry_class_75':[],'gerry_class_76':[], 'gerry_class_77':[],'gerry_class_78':[],'gerry_class_79':[]}'''

    data = {'x': [], 'y': [], 'gender': [], 'age': [], 'country': [], 'ethnicity': []}

    class_wt = dict()
    doc_idx = 2
    with open(datap) as dfile:
        dfile.readline()
        for line in dfile:
            line = line.strip().split('\t')
            # split indices
            data['x'].append(list(map(int, line[doc_idx].split())))
            data['y'].append(int(line[-1]))
            data['gender'].append(int(line[4])) #gender
            data['age'].append(int(line[5])) #age
            data['country'].append(int(line[8])) #country
            data['ethnicity'].append(int(line[9])) #ethnicity

    _data = pd.DataFrame(data)

    tasks = ['gender', 'age', 'country', 'ethnicity']
    attribute_distinct_labels = {attribute:list(_data[~_data[attribute].isnull()][attribute].unique()) for attribute in tasks}
    print (attribute_distinct_labels)
    gerry_combs = Gerrymandering_groups(
        attributes = tasks, 
        attribute_distinct_labels = attribute_distinct_labels
        )
    group_comb_itr = 0
    # iterate all gerry combs
    for task_comb, group_comb in tqdm(gerry_combs):
        group_indices = task_comb_data(_data, task_comb, group_comb)
        group_key = "@".join([str(i)+str(j) for i,j in zip(task_comb, group_comb)])
        _labels = [0] * len(_data)
        lind = 0
        for gid, y in zip(group_indices, _labels): 
            if gid: _labels[lind] = 1
            else: _labels[lind] = 0
            lind+=1
        data["gerry_class_{}".format(group_comb_itr)] = _labels
        group_comb_itr+=1
    
    # if shuffle the dataset
    if if_shuffle:
        (data['x'], data['y'], data["gerry_class_0"], data["gerry_class_1"], data["gerry_class_2"], data["gerry_class_3"], data["gerry_class_4"], data["gerry_class_5"], data["gerry_class_6"], 
        data["gerry_class_7"], data["gerry_class_8"], data["gerry_class_9"], data["gerry_class_10"], data["gerry_class_11"], data["gerry_class_12"], data["gerry_class_13"], data["gerry_class_14"], data["gerry_class_15"], data["gerry_class_16"], 
        data["gerry_class_17"], data["gerry_class_18"], data["gerry_class_19"], data["gerry_class_20"], data["gerry_class_21"], data["gerry_class_22"], data["gerry_class_23"], data["gerry_class_24"], data["gerry_class_25"], 
        data["gerry_class_26"], data["gerry_class_27"], data["gerry_class_28"], data["gerry_class_29"], data["gerry_class_30"], data["gerry_class_31"], data["gerry_class_32"], data["gerry_class_33"], data["gerry_class_34"], 
        data["gerry_class_35"], data["gerry_class_36"], data["gerry_class_37"], data["gerry_class_38"], data["gerry_class_39"], data["gerry_class_40"], data["gerry_class_41"], data["gerry_class_42"], data["gerry_class_43"], 
        data["gerry_class_44"], data["gerry_class_45"], data["gerry_class_46"], data["gerry_class_47"], data["gerry_class_48"], data["gerry_class_49"], data["gerry_class_50"], data["gerry_class_51"], data["gerry_class_52"], 
        data["gerry_class_53"], data["gerry_class_54"], data["gerry_class_55"], data["gerry_class_56"], data["gerry_class_57"], data["gerry_class_58"], data["gerry_class_59"], data["gerry_class_60"], data["gerry_class_61"], 
        data["gerry_class_62"], data["gerry_class_63"], data["gerry_class_64"], data["gerry_class_65"], data["gerry_class_66"], data["gerry_class_67"], data["gerry_class_68"], data["gerry_class_69"], data["gerry_class_70"], 
        data["gerry_class_71"], data["gerry_class_72"], data["gerry_class_73"], data["gerry_class_74"], data["gerry_class_75"], data["gerry_class_76"], 
        data["gerry_class_77"], data["gerry_class_78"], data["gerry_class_79"] ) = shuffle(data['x'], data['y'], data["gerry_class_0"], data["gerry_class_1"], data["gerry_class_2"], data["gerry_class_3"], 
        data["gerry_class_4"], data["gerry_class_5"], data["gerry_class_6"], data["gerry_class_7"], data["gerry_class_8"], data["gerry_class_9"], data["gerry_class_10"], data["gerry_class_11"], data["gerry_class_12"], 
        data["gerry_class_13"], data["gerry_class_14"], data["gerry_class_15"], data["gerry_class_16"], 
        data["gerry_class_17"], data["gerry_class_18"], data["gerry_class_19"], data["gerry_class_20"], data["gerry_class_21"], data["gerry_class_22"], data["gerry_class_23"], data["gerry_class_24"], data["gerry_class_25"], 
        data["gerry_class_26"], data["gerry_class_27"], data["gerry_class_28"], data["gerry_class_29"], data["gerry_class_30"], data["gerry_class_31"], data["gerry_class_32"], data["gerry_class_33"], data["gerry_class_34"], 
        data["gerry_class_35"], data["gerry_class_36"], data["gerry_class_37"], data["gerry_class_38"], data["gerry_class_39"], data["gerry_class_40"], data["gerry_class_41"], data["gerry_class_42"], data["gerry_class_43"], 
        data["gerry_class_44"], data["gerry_class_45"], data["gerry_class_46"], data["gerry_class_47"], data["gerry_class_48"], data["gerry_class_49"], data["gerry_class_50"], data["gerry_class_51"], data["gerry_class_52"], 
        data["gerry_class_53"], data["gerry_class_54"], data["gerry_class_55"], data["gerry_class_56"], data["gerry_class_57"], data["gerry_class_58"], data["gerry_class_59"], data["gerry_class_60"], data["gerry_class_61"], 
        data["gerry_class_62"], data["gerry_class_63"], data["gerry_class_64"], data["gerry_class_65"], data["gerry_class_66"], data["gerry_class_67"], data["gerry_class_68"], data["gerry_class_69"], data["gerry_class_70"], 
        data["gerry_class_71"], data["gerry_class_72"], data["gerry_class_73"], data["gerry_class_74"], data["gerry_class_75"], data["gerry_class_76"], data["gerry_class_77"], data["gerry_class_78"], data["gerry_class_79"])

    steps = len(data['x']) // batch_size
    if len(data['x']) % batch_size != 0:
        steps += 1
    for step in range(steps):
        yield class_wt, \
            np.asarray(data['x'][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data['y'][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_0"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_1"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_2"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_3"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_4"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_5"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_6"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_7"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_8"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_9"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_10"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_11"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_12"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_13"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_14"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_15"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_16"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_17"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_18"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_19"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_20"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_21"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_22"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_23"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_24"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_25"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_26"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_27"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_28"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_29"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_30"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_31"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_32"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_33"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_34"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_35"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_36"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_37"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_38"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_39"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_40"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_41"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_42"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_43"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_44"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_45"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_46"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_47"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_48"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_49"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_50"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_51"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_52"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_53"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_54"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_55"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_56"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_57"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_58"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_59"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_60"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_61"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_62"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_63"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_64"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_65"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_66"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_67"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_68"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_69"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_70"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_71"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_72"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_73"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_74"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_75"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_76"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_77"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_78"][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data["gerry_class_79"][step*batch_size: (step+1)*batch_size])
            
if __name__ == '__main__':
    inter_enc = intersectional_label_encoder([0,1], [0,1], [0,1], [0,1])
    print(inter_enc)
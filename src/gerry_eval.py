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

import json
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np

from utils import full_label_data
from utils import task_comb_data
from utils import binary_label_dist

from itertools import combinations
from tqdm import tqdm

from scipy.spatial import distance

import model_bias_analysis as mba
from collections import defaultdict 

comb_list = [   ['age'], 
                ['ethnicity'],
                ['target_gender'],
                ['age', 'ethnicity'],
                ['age', 'target_gender'],
                ['ethnicity', 'target_gender'],
                ['age', 'ethnicity', 'target_gender']
            ]


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


def eval(df, pred="pred"):
    # get the task name from the file, gender or ethnicity
    tasks = ['gender', 'age', 'country', 'ethnicity']

    scores = {
        'accuracy': 0.0,
        'f1-macro': 0.0, # macro f1 score
        'f1-weight': 0.0, # weighted f1 score
        # 'auc': 0.0,
    }

    # accuracy, f1, auc
    scores['accuracy'] = metrics.accuracy_score(
        y_true=df.label, y_pred=df[pred]
    )
    scores['f1-macro'] = metrics.f1_score(
        y_true=df.label, y_pred=df[pred],
        average='macro'
    )
    scores['f1-weight'] = metrics.f1_score(
        y_true=df.label, y_pred=df[pred],
        average='weighted'
    )
    # fpr, tpr, _ = metrics.roc_curve(
    #     y_true=df.label, y_score=df["pred"]_prob,
    # )
    # scores['auc'] = metrics.auc(fpr, tpr)

    '''fairness gaps'''
    for task in tasks:

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
            y_true=task_df.label, y_pred=task_df[pred]
        ).ravel()
        # print(cal_fnr(fn, tp), 
        #       cal_fpr(fp, tn),
        #       cal_tpr(tp, fn),
        #       cal_tnr(tn, fp)
        #       )
        # print(tn, fp, fn, tp)

        # get the unique types of demographic groups
        uniq_types = task_df[task].unique()
        for group in uniq_types:
            # calculate group specific confusion matrix
            group_df = task_df[task_df[task] == group]
            
            g_tn, g_fp, g_fn, g_tp = metrics.confusion_matrix(
                y_true=group_df.label, y_pred=group_df[pred]
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

    print(scores)
    return scores

def linear_leakage(train_text_embd, train_author_label, test_text_embd, test_author_label, output=True):
    # leakage
    attack_model = LogisticRegression(multi_class='multinomial', max_iter=1000)
    attack_model.fit(train_text_embd, train_author_label)
    leakage = attack_model.score(test_text_embd, test_author_label)
    Y_test_hat = attack_model.predict(test_text_embd)
    F1 = f1_score(test_author_label, Y_test_hat, average = "macro")
    if output:
        print("Leakage Acc: {}".format(leakage))
        print("Leakage F1: {}".format(F1))
    return (leakage, F1)

def task_comb_data(df, 
                task_combs, 
                conditions,):
    selected_rows = np.array([True]*len(df))
    for task, condition in zip(task_combs, conditions):
        selected_rows = selected_rows & (df[task].to_numpy()==condition)
    return selected_rows

def get_all_combs(unique_type_list):
    number_tasks = len(unique_type_list)
    no_unique_types = [len(unique_type) for unique_type in unique_type_list]+[1]
    total_number = np.prod(no_unique_types)
    # print(number_tasks, total_number)
    
    # init 2d matrix
    group_combinations = [[None for j in range(number_tasks)] for i in range(total_number)]

    for single_task_id, single_task_types in enumerate(unique_type_list):
        # print(single_task_id, single_task_types)

        # calculate single repeat time
        single_repeat_time = int(np.prod(no_unique_types[single_task_id+1:]))
        # calculate whole list repeat time
        whole_repeat_time = int(total_number/single_repeat_time/len(single_task_types))
        # print(single_repeat_time, whole_repeat_time)

        # create col number
        task_col = []
        # single repeat
        for t_type in single_task_types:
            task_col = task_col + [t_type]*single_repeat_time
        # whole repeat
        task_col = task_col * whole_repeat_time
        # print(task_col)

        # add to matrix
        for i, v in enumerate(task_col):
            group_combinations[i][single_task_id] = v
    return group_combinations

def combination_eval(in_df,
                    task_combs = comb_list,
                    tasks = ['gender', 'age', 'country', 'ethnicity'],
                    label = "label", 
                    pred="pred",
                    all_labels = False,
                    print_results = True
                    ):
    # save dist at different level
    dist_results = {}
    scores = {} 
    
    if all_labels:
        df = in_df[full_label_data(in_df, tasks)]
    else:
        df = in_df

    overall_dist = binary_label_dist(df)
    dist_results["overall"] = (overall_dist, len(df))

    # overall dist
    overall_label = list(df[label])
    overall_pred = list(df[pred])

    # accuracy, f1, auc
    scores['accuracy'] = metrics.accuracy_score(
        y_true=overall_label, y_pred=overall_pred
    )
    scores['f1-macro'] = metrics.f1_score(
        y_true=overall_label, y_pred=overall_pred,
        average='macro'
    )
    scores['f1-weight'] = metrics.f1_score(
        y_true=overall_label, y_pred=overall_pred,
        average='weighted'
    )

    # get overall confusion matrix
    cnf = metrics.confusion_matrix(
        y_true=overall_label, y_pred=overall_pred
    )
    cnf = cnf/np.sum(cnf,axis=1)[:, np.newaxis]
    t00R, f01R, f10R, t11R = cnf.ravel()
    
    scores['t00'] = t00R
    scores['f01'] = f01R
    scores['f10'] = f10R
    scores['t11'] = t11R

    '''fairness gaps'''
    for task_comb in task_combs:
        # get all group label combinations
        # group_combinations = [p for p in product([0, 1], repeat=len(task_comb))]
        
        comb_uniq_types = [df[~df[t].isnull()][t].unique() for t in task_comb]

        group_combinations = get_all_combs(comb_uniq_types)
        
        scores["-".join(task_comb)] = {}
        
        # a full label subset
        t_df = df[full_label_data(df, task_comb)]

        # tasks rates
        t_cnf = metrics.confusion_matrix(
            y_true=list(t_df[label]), y_pred=list(t_df[pred])
        )

        t_cnf_normalized = t_cnf/np.sum(t_cnf,axis=1)[:, np.newaxis]

        t_t00R, t_f01R, t_f10R, t_t11R = t_cnf_normalized.ravel()
        task_rates = {
            # "cf" : t_cnf,
            "t-t00R" : t_t00R,
            "t-f01R" : t_f01R, 
            "t-f10R" : t_f10R, 
            "t-t11R" : t_t11R,
            "mean-GAP-overall" : 0.0,
            "mean-GAP-subset" : 0.0
        }

        scores["-".join(task_comb)]['subset-rates'] = task_rates

        for group_comb in group_combinations:
            # group scores
            task_comb_scores = {}

            group_df = df[task_comb_data(df, task_comb, group_comb)]
            # print(group_df)
            group_key = "_".join(task_comb+[str(i) for i in group_comb])
            dist_results[group_key] = (binary_label_dist(group_df), len(group_df))

            # group rates
            g_cnf = metrics.confusion_matrix(
                y_true=list(group_df[label]), y_pred=list(group_df[pred])
            )
            g_cnf = g_cnf/np.sum(g_cnf,axis=1)[:, np.newaxis]

            try:
                g_t00R, g_f01R, g_f10R, g_t11R = g_cnf.ravel()
            except:
                print(group_key)
                continue

            task_comb_scores["Number"] = len(group_df)

            task_comb_scores["t00R"] = g_t00R
            task_comb_scores["f01R"] = g_f01R
            task_comb_scores["f10R"] = g_f10R
            task_comb_scores["t11R"] = g_t11R
            
            # GAP compared to overall
            task_comb_scores["GAP-overall-t00R"] = g_t00R - t00R
            task_comb_scores["GAP-overall-f01R"] = g_f01R - f01R
            task_comb_scores["GAP-overall-f10R"] = g_f10R - f10R
            task_comb_scores["GAP-overall-t11R"] = g_t11R - t11R

            scores["-".join(task_comb)]['subset-rates']["mean-GAP-overall"] += (abs(g_f01R - f01R)+abs(g_f10R - f10R))

            # GAP compared to the full label subset 
            task_comb_scores["GAP-subset-t00R"] = g_t00R - t_t00R
            task_comb_scores["GAP-subset-f01R"] = g_f01R - t_f01R
            task_comb_scores["GAP-subset-f10R"] = g_f10R - t_f10R
            task_comb_scores["GAP-subset-t11R"] = g_t11R - t_t11R

            scores["-".join(task_comb)]['subset-rates']["mean-GAP-subset"] += (abs(g_f01R - t_f01R)+abs(g_f10R - t_f10R))
 
            # accuracy, f1, auc
            task_comb_scores['accuracy'] = metrics.accuracy_score(
                y_true=list(group_df[label]), y_pred=list(group_df[pred])
            )
            task_comb_scores['f1-macro'] = metrics.f1_score(
                y_true=list(group_df[label]), y_pred=list(group_df[pred]),
                average='macro'
            )
            task_comb_scores['f1-weight'] = metrics.f1_score(
                y_true=list(group_df[label]), y_pred=list(group_df[pred]),
                average='weighted'
            )
            scores["-".join(task_comb)]["-".join([str(i) for i in group_comb])] = task_comb_scores
        scores["-".join(task_comb)]['subset-rates']["mean-GAP-overall"] = scores["-".join(task_comb)]['subset-rates']["mean-GAP-overall"]/len(group_combinations)
        scores["-".join(task_comb)]['subset-rates']["mean-GAP-subset"] = scores["-".join(task_comb)]['subset-rates']["mean-GAP-subset"]/len(group_combinations)
    return dist_results, scores

# Gerry Fairness
def Gerrymandering_groups(attributes, attribute_distinct_labels):
    '''
    attributes: a list of attribute names, e.g., "gender", "age"
    attribute_distinct_labels: a dictionary where each value is the distinct label list corresponding to the key.
    See /notebooks/dev/Gerrymandering_groups.ipynb
    '''
    attribute_label_pairs = []
    # iterate all combinations of attributes
    for l in range(len(attributes)):
        for attribute_comb in [list(i) for i in combinations(attributes, (l+1))]:
            comb_distinct_labels = [attribute_distinct_labels[comb] for comb in attribute_comb]
            all_att_comb_label_combs = get_all_combs(comb_distinct_labels)
            for att_label in  all_att_comb_label_combs:
                attribute_label_pairs.append((attribute_comb, att_label))

    return attribute_label_pairs

def Gerrymandering_eval(in_df,
                        tasks = ['gender', 'age', 'country', 'ethnicity'],
                        label = "label", 
                        pred = "pred",
                        pred_prob = "pred_prob",
                        all_labels = False,
                        print_results = True
                        ):
    scores = {} 
    
    if all_labels:
        df = in_df[full_label_data(in_df, tasks)]
    else:
        df = in_df
    
    scores["overall"] = {}
    
    overall_label = list(df[label])
    overall_pred = list(df[pred])

    # accuracy, f1, auc
    scores["overall"]['accuracy'] = metrics.accuracy_score(
        y_true=overall_label, y_pred=overall_pred
    )
    scores["overall"]['f1-macro'] = metrics.f1_score(
        y_true=overall_label, y_pred=overall_pred,
        average='macro'
    )
    scores["overall"]['f1-weight'] = metrics.f1_score(
        y_true=overall_label, y_pred=overall_pred,
        average='weighted'
    )

    # get overall confusion matrix
    cnf = metrics.confusion_matrix(
        y_true=overall_label, y_pred=overall_pred
    )
    cnf = cnf/np.sum(cnf,axis=1)[:, np.newaxis]
    t00R, f01R, f10R, t11R = cnf.ravel()
    
    scores["overall"]['t00'] = t00R
    scores["overall"]['f01'] = f01R
    scores["overall"]['f10'] = f10R
    scores["overall"]['t11'] = t11R

    # gerry combinations
    attribute_distinct_labels = {attribute:list(df[~df[attribute].isnull()][attribute].unique()) for attribute in tasks}
    gerry_combs = Gerrymandering_groups(
        attributes = tasks, 
        attribute_distinct_labels = attribute_distinct_labels
        )
    # iterate all gerry combs
    for task_comb, group_comb in tqdm(gerry_combs):
        group_indices = task_comb_data(df, task_comb, group_comb)
        group_key = "@".join([str(i)+str(j) for i,j in zip(task_comb, group_comb)])
        df[group_key] = group_indices
        
        ''' 
        group ACC, F1, tpr, tnr, fpr, fnr.
        '''
        group_df = df[group_indices]
        
        scores[group_key] = {}
        scores[group_key]["Number"] = len(group_df)
        
        # group confusion matrix
        g_cnf = metrics.confusion_matrix(
            y_true=list(group_df[label]), y_pred=list(group_df[pred])
        )
        g_cnf = g_cnf/np.sum(g_cnf,axis=1)[:, np.newaxis]

        try:
            g_t00R, g_f01R, g_f10R, g_t11R = g_cnf.ravel()
            scores[group_key]['t00'] = g_t00R
            scores[group_key]['f01'] = g_f01R
            scores[group_key]['f10'] = g_f10R
            scores[group_key]['t11'] = g_t11R
        except:
            print("Confusion matrix error:", group_key)
            continue
        
        # accuracy, f1, auc
        group_label = list(group_df[label])
        group_pred = list(group_df[pred])
    
        scores[group_key]['accuracy'] = metrics.accuracy_score(
            y_true=group_label, y_pred=group_pred
        )
        scores[group_key]['f1-macro'] = metrics.f1_score(
            y_true=group_label, y_pred=group_pred,
            average='macro'
        )
        
        '''
        AUC based metrices
        '''
        # pos_aseg, neg_aseg = mba.compute_average_squared_equality_gap(df = df, 
        #                                                             subgroup = group_key, 
        #                                                             label = label,
        #                                                             model_name= pred_prob
        #                                                             )
        
        auc = mba.compute_subgroup_auc(df = df, 
                                        subgroup = group_key, 
                                        label = label,
                                        model_name= pred_prob
                                        )
        
        negative_cross_auc = mba.compute_negative_cross_auc(df = df, 
                                                            subgroup = group_key, 
                                                            label = label,
                                                            model_name= pred_prob
                                                            )
                                
        positive_cross_auc = mba.compute_positive_cross_auc(df = df, 
                                                            subgroup = group_key, 
                                                            label = label,
                                                            model_name= pred_prob
                                                            )
        
        # scores[group_key]["pos_aseg"] = pos_aseg
        # scores[group_key]["neg_aseg"] = neg_aseg
        scores[group_key]["auc"] = auc
        scores[group_key]["negative_cross_auc"] = negative_cross_auc
        scores[group_key]["positive_cross_auc"] = positive_cross_auc
        
        
    return scores

def power_mean(series, p):
    if p>50:
        return max(series)
    elif p<50:
        return min(series)
    else:
        total = np.mean(np.power(series, p))
        return np.power(total, 1 / p)

def get_final_metric(scores_df, POWER=-5):
    """
    Aggregating scores based on the formula used in Kaggle.
    """

    metrics_score = defaultdict(list)
    for k, v in scores_df.items():
        for metrics, value in v.items():
            metrics_score[metrics].extend(value)

    bias_score = [
        power_mean(metrics_score["auc"], POWER),
        power_mean(metrics_score["negative_cross_auc"], POWER),
        power_mean(metrics_score["positive_cross_auc"], POWER)
    ]
    return bias_score

hate_speech_keymapping = {
    'gender':{0:"male", 1:"female"}, 
    'age':{0:"old", 1:"young"}, 
    'country':{0:"notInUS", 1:"inUS"}, 
    'ethnicity':{0:"white", 1:"other"}
}

def bias_attribute_selection(gerry_eval_results, 
                            df,
                            attributes = ['gender', 'age', 'country', 'ethnicity'],
                            subgroup_min_size = 500,
                            key_mapping = hate_speech_keymapping,
                            ):
    # read overall results.
    overall_results = gerry_eval_results["overall"]

    # metrices
    f1_macro = []
    GAP_f1_macro = []
    t00 = []
    GAP_t00 = []
    f01 = []
    GAP_f01 = []
    f10 = []
    GAP_f10 = []
    t11 = []
    GAP_t11 = []

    Weighted_GAP_t00 = []
    Weighted_GAP_f01 = []
    Weighted_GAP_f10 = []
    Weighted_GAP_t11 = []

    negative_cross_auc = []
    positive_cross_auc = []

    # overall performance
    overall_accuracy = overall_results['accuracy']
    overall_f1_macro = overall_results['f1-macro']
    overall_f1_weight = overall_results['f1-weight']
    overall_t00 = overall_results['t00']
    overall_f01 = overall_results['f01']
    overall_f10 = overall_results['f10']
    overall_t11 = overall_results['t11']
    
    # treat subgroup_min_size as percentage if 0 <= subgroup_min_size < 1
    subgroup_min_size = int(len(df)*subgroup_min_size) if subgroup_min_size < 1 else subgroup_min_size

    # gerry combinations
    attribute_distinct_labels = {attribute:list(df[~df[attribute].isnull()][attribute].unique()) for attribute in attributes}
    gerry_combs = Gerrymandering_groups(
        attributes = attributes, 
        attribute_distinct_labels = attribute_distinct_labels
        )
    
    # condidate groups
  
    valid_combinations = []
    for l in range(len(attributes)):
        for attribute_comb in [list(i) for i in combinations(attributes, (l+1))]:
            current_all_subgroups = [i for i in gerry_combs if set(i[0]) == set(attribute_comb)]

            current_all_subgroup_ids = ["@".join([str(i)+str(j) for i,j in zip(task_comb, group_comb)]) \
                                                        for task_comb, group_comb in current_all_subgroups]
            subgroup_sizes = [gerry_eval_results[sgid]["Number"] for sgid in current_all_subgroup_ids]
            print (current_all_subgroup_ids, subgroup_sizes)
            if min(subgroup_sizes) >= subgroup_min_size:
                valid_combinations.append(attribute_comb)
            else:
                print("Mini size of group {}:{}".format(attribute_comb, min(subgroup_sizes)))

    # Bias eval of each subgroup

    # filter sub groups
    subgroups = [i for i in gerry_combs if i[0] in valid_combinations]

    # Size the set
    total_number = float(len(df))
    
    for task_comb, group_comb in subgroups:
        group_indices = task_comb_data(df, task_comb, group_comb)
        group_key = "@".join([str(i)+str(j) for i,j in zip(task_comb, group_comb)])
        group_key_readable = "@".join([str(i)+'-'+key_mapping[i][int(j)] for i,j in zip(task_comb, group_comb)])

        group_df = df[group_indices]
        
        p_g0 = len(group_df[(group_df["label"]==0)])/total_number
        p_g1 = len(group_df[(group_df["label"]==1)])/total_number

        subgroup_results = gerry_eval_results[group_key]
        #print(group_key, len(subgroup_results), subgroup_results.keys())
        subgroup_accuracy = subgroup_results['accuracy']
        subgroup_f1_macro = subgroup_results['f1-macro']
        subgroup_t00 = subgroup_results['t00']
        subgroup_f01 = subgroup_results['f01']
        subgroup_f10 = subgroup_results['f10']
        subgroup_t11 = subgroup_results['t11']
        subgroup_negative_cross_auc = subgroup_results['negative_cross_auc']
        subgroup_positive_cross_auc = subgroup_results['positive_cross_auc']

        f1_macro.append((subgroup_f1_macro, group_key_readable))
        t00.append((subgroup_t00, group_key_readable))
        f01.append((subgroup_f01, group_key_readable))
        f10.append((subgroup_f10, group_key_readable))
        t11.append((subgroup_t11, group_key_readable))

        negative_cross_auc.append((subgroup_negative_cross_auc, group_key_readable))
        positive_cross_auc.append((subgroup_positive_cross_auc, group_key_readable))

        GAP_f1_macro.append((abs(overall_f1_macro - subgroup_f1_macro), group_key_readable))

        GAP_t00.append((abs(overall_t00 - subgroup_t00), group_key_readable))
        GAP_f01.append((abs(overall_f01 - subgroup_f01), group_key_readable))
        GAP_f10.append((abs(overall_f10 - subgroup_f10), group_key_readable))
        GAP_t11.append((abs(overall_t11 - subgroup_t11), group_key_readable))

        Weighted_GAP_t00.append((p_g0*abs(overall_t00 - subgroup_t00), group_key_readable))
        Weighted_GAP_f01.append((p_g1*abs(overall_f01 - subgroup_f01), group_key_readable))
        Weighted_GAP_f10.append((p_g0*abs(overall_f10 - subgroup_f10), group_key_readable))
        Weighted_GAP_t11.append((p_g1*abs(overall_t11 - subgroup_t11), group_key_readable))

    return {
        "f1_macro" : f1_macro,
        "t00" : t00,
        "f01" : f01,
        "f10" : f10,
        "t11" : t11,
        "subgroup_negative_cross_auc" : negative_cross_auc,
        "subgroup_positive_cross_auc" : positive_cross_auc,
        "GAP_f1_macro" : GAP_f1_macro,
        "GAP_t00" : GAP_t00,
        "GAP_f01" : GAP_f01,
        "GAP_f10" : GAP_f10,
        "GAP_t11" : GAP_t11,
        "Weighted_GAP_t00" : Weighted_GAP_t00,
        "Weighted_GAP_f01" : Weighted_GAP_f01,
        "Weighted_GAP_f10" : Weighted_GAP_f10,
        "Weighted_GAP_t11" : Weighted_GAP_t11,
    }


def bias_attribute_selection_simple(gerry_eval_results, 
                            df,
                            attributes = ['gender', 'age', 'country', 'ethnicity'],
                            subgroup_min_size = 500,
                            key_mapping = hate_speech_keymapping,
                            ):
    # read overall results.
    overall_results = gerry_eval_results["overall"]

    # metrices
    f1_macro = []
    GAP_f1_macro = []
    t00 = []
    GAP_t00 = []
    f01 = []
    GAP_f01 = []
    f10 = []
    GAP_f10 = []
    t11 = []
    GAP_t11 = []

    Weighted_GAP_t00 = []
    Weighted_GAP_f01 = []
    Weighted_GAP_f10 = []
    Weighted_GAP_t11 = []

    negative_cross_auc = []
    positive_cross_auc = []

    # overall performance
    overall_accuracy = overall_results['accuracy']
    overall_f1_macro = overall_results['f1-macro']
    overall_f1_weight = overall_results['f1-weight']
    overall_t00 = overall_results['t00']
    overall_f01 = overall_results['f01']
    overall_f10 = overall_results['f10']
    overall_t11 = overall_results['t11']
    
    # treat subgroup_min_size as percentage if 0 <= subgroup_min_size < 1
    subgroup_min_size = int(len(df)*subgroup_min_size) if subgroup_min_size < 1 else subgroup_min_size

    # gerry combinations
    attribute_distinct_labels = {attribute:list(df[~df[attribute].isnull()][attribute].unique()) for attribute in attributes}
    gerry_combs = Gerrymandering_groups(
        attributes = attributes, 
        attribute_distinct_labels = attribute_distinct_labels
        )
    
    # condidate groups

    total_number = float(len(df))
    
    for task_comb, group_comb in tqdm(gerry_combs):
        group_indices = task_comb_data(df, task_comb, group_comb)
        group_key = "@".join([str(i)+str(j) for i,j in zip(task_comb, group_comb)])
        group_key_readable = "@".join([str(i)+'-'+key_mapping[i][int(j)] for i,j in zip(task_comb, group_comb)])
        group_df = df[group_indices]

        if len(group_df) > subgroup_min_size:
            p_g0 = len(group_df[(group_df["label"]==0)])/total_number
            p_g1 = len(group_df[(group_df["label"]==1)])/total_number

            subgroup_results = gerry_eval_results[group_key]
            #print(group_key, len(subgroup_results), subgroup_results.keys())
            subgroup_accuracy = subgroup_results['accuracy']
            subgroup_f1_macro = subgroup_results['f1-macro']
            subgroup_t00 = subgroup_results['t00']
            subgroup_f01 = subgroup_results['f01']
            subgroup_f10 = subgroup_results['f10']
            subgroup_t11 = subgroup_results['t11']
            subgroup_negative_cross_auc = subgroup_results['negative_cross_auc']
            subgroup_positive_cross_auc = subgroup_results['positive_cross_auc']

            f1_macro.append((subgroup_f1_macro, group_key_readable))
            t00.append((subgroup_t00, group_key_readable))
            f01.append((subgroup_f01, group_key_readable))
            f10.append((subgroup_f10, group_key_readable))
            t11.append((subgroup_t11, group_key_readable))

            negative_cross_auc.append((subgroup_negative_cross_auc, group_key_readable))
            positive_cross_auc.append((subgroup_positive_cross_auc, group_key_readable))

            GAP_f1_macro.append((abs(overall_f1_macro - subgroup_f1_macro), group_key_readable))

            GAP_t00.append((abs(overall_t00 - subgroup_t00), group_key_readable))
            GAP_f01.append((abs(overall_f01 - subgroup_f01), group_key_readable))
            GAP_f10.append((abs(overall_f10 - subgroup_f10), group_key_readable))
            GAP_t11.append((abs(overall_t11 - subgroup_t11), group_key_readable))

            Weighted_GAP_t00.append((p_g0*abs(overall_t00 - subgroup_t00), group_key_readable))
            Weighted_GAP_f01.append((p_g1*abs(overall_f01 - subgroup_f01), group_key_readable))
            Weighted_GAP_f10.append((p_g0*abs(overall_f10 - subgroup_f10), group_key_readable))
            Weighted_GAP_t11.append((p_g1*abs(overall_t11 - subgroup_t11), group_key_readable))

    return {
        "f1_macro" : f1_macro,
        "t00" : t00,
        "f01" : f01,
        "f10" : f10,
        "t11" : t11,
        "subgroup_negative_cross_auc" : negative_cross_auc,
        "subgroup_positive_cross_auc" : positive_cross_auc,
        "GAP_f1_macro" : GAP_f1_macro,
        "GAP_t00" : GAP_t00,
        "GAP_f01" : GAP_f01,
        "GAP_f10" : GAP_f10,
        "GAP_t11" : GAP_t11,
        "Weighted_GAP_t00" : Weighted_GAP_t00,
        "Weighted_GAP_f01" : Weighted_GAP_f01,
        "Weighted_GAP_f10" : Weighted_GAP_f10,
        "Weighted_GAP_t11" : Weighted_GAP_t11,
    }
    

def return_1st_element(input_list):
    return np.array([i[0] for i in input_list])

def bias_metric_aggregation(eval_scores, 
                            power_scalar=1,
                            weights_of_metrics = None
                            ):
    """
    aggregation over all sub-groups
    
    power_scalar: the exponent of the power mean. If power_scalar is set to be np.inf, 
    """
    

    if weights_of_metrics == None:
        weights_of_metrics = {
            "agg_f1": 1, 
            "agg_f10_rate": -1,
            "agg_f01_rate": -1,
            "agg_false_rate": -1, 
            "agg_t00_rate": 1,
            "agg_t11_rate": 1,
            "agg_true_rate": 1, 
            "agg_auc": 1, 
            "agg_GAP_t00": -1,
            "agg_GAP_t11": -1,
            "agg_GAP": -1,
            "agg_weighted_GAP_t00": -1,
            "agg_weighted_GAP_t11": -1,
            }
    # min f1_macro
    agg_f1 = power_mean(return_1st_element(eval_scores["f1_macro"]), -1*power_scalar)

    # max f10, f01
    agg_f10_rate = power_mean(return_1st_element(eval_scores["f10"]), power_scalar)
    agg_f01_rate = power_mean(return_1st_element(eval_scores["f01"]), power_scalar)

    agg_false_rate = power_mean((return_1st_element(eval_scores["f10"])+return_1st_element(eval_scores["f01"])), power_scalar)

    # min t11, t00
    agg_t00_rate = power_mean(return_1st_element(eval_scores["t00"]), -1*power_scalar)
    agg_t11_rate = power_mean(return_1st_element(eval_scores["t11"]), -1*power_scalar)

    agg_true_rate = power_mean((return_1st_element(eval_scores["t00"])+return_1st_element(eval_scores["t11"])), -1*power_scalar)

    #? min AUC
    agg_subgroup_negative_cross_auc = power_mean(return_1st_element(eval_scores["subgroup_negative_cross_auc"]), -1*power_scalar)
    agg_subgroup_positive_cross_auc = power_mean(return_1st_element(eval_scores["subgroup_positive_cross_auc"]), -1*power_scalar)

    agg_auc = np.mean([agg_subgroup_negative_cross_auc, agg_subgroup_positive_cross_auc])

    # max gap
    agg_GAP_t00 = power_mean(return_1st_element(eval_scores['GAP_t00']), power_scalar)
    agg_GAP_t11 = power_mean(return_1st_element(eval_scores['GAP_t11']), power_scalar)

    # max weghted gap
    agg_weighted_GAP_t00 = power_mean(return_1st_element(eval_scores['Weighted_GAP_t00']), power_scalar)
    agg_weighted_GAP_t11 = power_mean(return_1st_element(eval_scores['Weighted_GAP_t11']), power_scalar)
    
    # max group gap
    agg_GAP = power_mean((return_1st_element(eval_scores['GAP_t00'])+return_1st_element(eval_scores['GAP_t11'])), power_scalar)

    weighted_sum = (weights_of_metrics["agg_f1"] * agg_f1 +
                    weights_of_metrics["agg_f10_rate"] * agg_f10_rate +
                    weights_of_metrics["agg_f01_rate"] * agg_f01_rate +
                    weights_of_metrics["agg_false_rate"] * agg_false_rate +
                    weights_of_metrics["agg_t00_rate"] * agg_t00_rate +
                    weights_of_metrics["agg_t11_rate"] * agg_t11_rate +
                    weights_of_metrics["agg_true_rate"] * agg_true_rate +
                    weights_of_metrics["agg_auc"] * agg_auc +
                    weights_of_metrics["agg_GAP_t00"] * agg_GAP_t00 +
                    weights_of_metrics["agg_GAP_t11"] * agg_GAP_t11 +
                    weights_of_metrics["agg_GAP"] * agg_GAP +
                    weights_of_metrics["agg_weighted_GAP_t00"] * agg_weighted_GAP_t00 +
                    weights_of_metrics["agg_weighted_GAP_t11"] * agg_weighted_GAP_t11
                    )

    single_metices = {
        "agg_f1":agg_f1, 
        "agg_f10_rate":agg_f10_rate, 
        "agg_f01_rate":agg_f01_rate,
        "agg_false_rate":agg_false_rate,
        "agg_t00_rate":agg_t00_rate,
        "agg_t11_rate":agg_t11_rate,
        "agg_true_rate":agg_true_rate, 
        "agg_auc":agg_auc, 
        "agg_GAP":agg_GAP,
        "agg_weighted_GAP_t00":agg_weighted_GAP_t00,
        "agg_weighted_GAP_t11":agg_weighted_GAP_t11
    }

    return weighted_sum, single_metices

def iteration_selection(iteration_data, selection_method = None, selection_parameters = None):
    """
    Input:
        iteration_data: dictionary which includes dev and test evaluation scores at each iteration.
        
        selection_method:
            - performance: given the min performance or max performance loss, find the smallest bias
            - fairness: given the min fairness or min fairness gain, find the best performance
            - linear: given the weights, find the best fairness-performance combination
        
        selection_parameters: a dictionary of parameters that are needed for model selection. 
    Output:
        best_iteration: the index of the chosen iteration based on dev set.
    """
    parameters = {
        # author info that are considered
        "attributes" : ['gender', 'age', 'country', 'ethnicity'],
                
        # minmum size of a subgroup that can be considered
        "subgroup_min_size" : 0,
                
        # dev dataframe
        "df" : None,
                
        # power scalar in the generalized mean of sub-group scores
        "power_scalar" : 1,
                
        # weights of each metric when merging them together
        "weights_of_metrics" : None,
                
        # overall performance metric
        "performance_metric" : "f1-macro",
                
        # min performance, this is used when selection_method == "performance"
        "min_performance" : 0.5,

        # max bias that can be accepted, this is used when selection_method == "bias"
        "min_fairness" : 0,

        # weights of main performance and fairness score when combining them lienarly.
        # notice that both of these two scores are the larger the better
        "performance_fairness_weights" : np.array([1, 1])
    }

    # update parameters
    if selection_parameters != None:
        for key in selection_parameters.keys():
            parameters[key] = selection_parameters[key]
    
    # save performance-fairness score of each iteration
    score_of_iterations = []

    # iterate each iteration
    for iteration in iteration_data.keys():
        # iteration overall performance
        iter_overall_performance = iteration_data[iteration]['dev']['overall'][parameters["performance_metric"]]

        # get scores of each valid sub-group
        iteration_scores = bias_attribute_selection(
            gerry_eval_results = iteration_data[iteration]['dev'], 
            attributes = parameters["attributes"],
            subgroup_min_size = parameters["subgroup_min_size"],
            df = parameters["df"]
        )

        # aggregation over sub-group scores
        iter_fairness_score, _ = bias_metric_aggregation(
            eval_scores = iteration_scores, 
            power_scalar = parameters["power_scalar"],
            weights_of_metrics = parameters["weights_of_metrics"]
        )

        # calculate a score of current iteration
        if selection_method == "performance":
            # overall performance is larger than the threshold
            if iter_overall_performance >= parameters["min_performance"]:
                score_of_iterations.append((iter_fairness_score, iteration))
        elif selection_method == "fairness":
            # fairness is larger than the threshold
            if iter_fairness_score >= parameters["min_fairness"]:
                score_of_iterations.append((iter_overall_performance, iteration))
        elif selection_method == "linear":
            # add a linear combination of the overall performance and fairness
            tradeoff_score = parameters["performance_fairness_weights"] @ np.array([iter_overall_performance, iter_fairness_score])
            score_of_iterations.append((tradeoff_score, iteration))
        else:
            print("Unknown selection method")
            return -1
    
    # find the best iteration
    if len(score_of_iterations) == 0:
        best_iteration = '1'
    else:
        # find the best iteration
        best_iteration = sorted(score_of_iterations, reverse=True)[0][1]

    return best_iteration, score_of_iterations


# Faster than is_pareto_efficient_simple, but less readable.
def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

def get_pareto_frontiers(iteration_data, selection_parameters = None):
    """
    Input:
        iteration_data: dictionary which includes dev and test evaluation scores at each iteration.
        
        selection_method:
            - performance: given the min performance or max performance loss, find the smallest bias
            - fairness: given the min fairness or min fairness gain, find the best performance
            - linear: given the weights, find the best fairness-performance combination
        
        selection_parameters: a dictionary of parameters that are needed for model selection. 
    Output:
        best_iteration: the index of the chosen iteration based on dev set.
    """
    parameters = {
        # author info that are considered
        "attributes" : ['gender', 'age', 'country', 'ethnicity'],
                
        # minmum size of a subgroup that can be considered
        "subgroup_min_size" : 0,
                
        # dev dataframe 
        "df" : None,
                
        # power scalar in the generalized mean of sub-group scores
        "power_scalar" : np.inf,
                
        # weights of each metric when merging them together
        "weights_of_metrics" : None,
                
        # overall performance metric
        "performance_metric" : "f1-macro",
                
    }

    # update parameters
    if selection_parameters != None:
        for key in selection_parameters.keys():
            parameters[key] = selection_parameters[key]

    # Init best_iteration
    best_iteration = None
    
    # save performance-fairness score of each iteration
    df_iteration = []
    df_fairness = []
    df_performance = []

    # iterate each iteration
    for iteration in tqdm(iteration_data.keys()):
        # iteration overall performance
        iter_overall_performance = iteration_data[iteration]['dev']['overall'][parameters["performance_metric"]]

        # get scores of each valid sub-group
        iteration_scores = bias_attribute_selection(
            gerry_eval_results = iteration_data[iteration]['dev'], 
            attributes = parameters["attributes"],
            subgroup_min_size = parameters["subgroup_min_size"],
            df = parameters["df"]
        )

        # aggregation over sub-group scores
        iter_fairness_score, _ = bias_metric_aggregation(
            eval_scores = iteration_scores, 
            power_scalar = parameters["power_scalar"],
            weights_of_metrics = parameters["weights_of_metrics"]
        )

        df_iteration.append(iteration)
        df_fairness.append(iter_fairness_score)
        df_performance.append(iter_overall_performance)

    trade_off_df = pd.DataFrame({
        "iteration" : df_iteration,
        "fairness" : df_fairness,
        "performance" : df_performance,
    })

    costs = -1*trade_off_df[["fairness", "performance"]].to_numpy()
    pareto_frontiers = is_pareto_efficient(costs, return_mask = True)
    trade_off_df["Pareto"] = pareto_frontiers

    pareto_frontiers_df = trade_off_df[trade_off_df["Pareto"]==True]

    return pareto_frontiers_df, trade_off_df

def pareto_selection(pareto_frontiers_df, 
                    selection_method = "fairness",
                    fairness_column = "fairness", 
                    performance_column = "performance",
                    dist_to_best_column = "distance",
                    min_performance = None,
                    min_fairness = None,
                    best_point = None,
                    scale = True,
                    distance_type = "euclidean",
                    selection_parameters = None):
    """
    Input:
        pareto_frontiers: data frame which are pareto frontiers 
        
        selection_method:
            - performance: find the best overall performace
            - fairness: find the best fairness
            - distance: find the closest point to the best point
        
        selection_parameters: a dictionary of parameters that are needed for model selection. 
    Output:
        select_models: data frame including selected models
    """
    pareto_frontiers_df = pareto_frontiers_df.copy()
    best_iteration = None

    # filtering by min fairness and performance
    if min_performance is None:
        min_performance = min(pareto_frontiers_df[performance_column])
    if min_fairness is None:
        min_fairness = min(pareto_frontiers_df[fairness_column])

    pareto_frontiers_df = pareto_frontiers_df[(pareto_frontiers_df[fairness_column]>=min_fairness)&(pareto_frontiers_df[performance_column]>=min_performance)]
    
    if len(pareto_frontiers_df) == 0:
        return best_iteration, pareto_frontiers_df 

    # calculate a score of current iteration
    if selection_method == "performance":        
        
        best_iteration = pareto_frontiers_df[pareto_frontiers_df[performance_column] == max(pareto_frontiers_df[performance_column])]

        return best_iteration, pareto_frontiers_df
    elif selection_method == "fairness":
        best_iteration = pareto_frontiers_df[pareto_frontiers_df[fairness_column] == max(pareto_frontiers_df[fairness_column])]
        
    elif selection_method == "distance":
        if best_point is None:
            # best overall performace and fairness saprately
            best_performance = max(pareto_frontiers_df[performance_column])
            best_fairness = max(pareto_frontiers_df[fairness_column])

            best_point = [best_performance, best_fairness]
        else:
            try:
                best_point = [float(i) for i in best_point]
                assert len(best_point) == 2
            except:
                raise
        
        # Standardization 
        if scale == True:
            min_performance = min(pareto_frontiers_df[performance_column])

            min_fairness = min(pareto_frontiers_df[fairness_column])

            performance_scalar = best_point[0] - min_performance
            fairness_scalar = best_point[1] - min_fairness

            best_point = [1, 1]
        else:
            min_performance = 0
            min_fairness = 0
            
            performance_scalar = 1
            fairness_scalar = 1
        
        distance_to_best = []
        for scores in zip(pareto_frontiers_df[performance_column], pareto_frontiers_df[fairness_column]):
            overall, fairness = scores
            if distance_type == "euclidean":
                dst = distance.euclidean(
                    best_point, 
                    [
                        (overall-min_performance)/(performance_scalar+1e-9), 
                        (fairness-min_fairness)/(fairness_scalar+1e-9)
                        ]
                    )
            else:
                print("TBD")
                raise
            distance_to_best.append(dst)
        
        pareto_frontiers_df[dist_to_best_column] = distance_to_best

        best_iteration = pareto_frontiers_df[pareto_frontiers_df[dist_to_best_column] == min(pareto_frontiers_df[dist_to_best_column])]
    else:
        print("Unknown selection method")
        raise

    return best_iteration, pareto_frontiers_df

def get_TPR(y_main, y_hat_main, y_protected):
    
    all_y = list(Counter(y_main).keys())
    
    protected_vals = defaultdict(dict)
    for label in all_y:
        for i in range(np.max(y_protected)+1):
            used_vals = (y_main == label) & (y_protected == i)
            y_label = y_main[used_vals]
            y_hat_label = y_hat_main[used_vals]
            protected_vals['y:{}'.format(label)]['p:{}'.format(i)] = (y_label == y_hat_label).mean()
            
    diffs = {}
    for k, v in protected_vals.items():
        vals = list(v.values())
        diffs[k] = vals[0] - vals[1]
    return protected_vals, diffs

def rms(arr):
    return np.sqrt(np.mean(np.square(arr)))
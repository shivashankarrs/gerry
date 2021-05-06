import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations_with_replacement  
from itertools import product
from tqdm import tqdm

import pickle

def save_pkl(f_path, objective):
    with open(f_path,"wb") as filehandler:
        pickle.dump(objective,filehandler)

def load_pkl(f_path):
    with open(f_path,'rb') as f:
        object_file = pickle.load(f)
    return object_file


def binary_label_dist(df, col_name="label"):
    label_dist = Counter(df[col_name])
    label_dist = np.array([label_dist[0], label_dist[1]])
    label_dist = label_dist/sum(label_dist)
    return label_dist

def full_label_data(df, tasks):
    selected_rows = np.array([True]*len(df))
    for task in tasks:
        selected_rows = selected_rows & df[task].notnull().to_numpy()
    return selected_rows

def single_analysis(in_df, 
                    tasks = ['gender', 'age', 'country', 'ethnicity'],
                    all_labels = False
                    ):
    # save dist at different level
    dist_results = {}    
    
    if all_labels:

        df = in_df[full_label_data(in_df, tasks)]
    else:
        df = in_df

    overall_dist = binary_label_dist(df)
    dist_results["overall"] = (overall_dist, len(df))
    
    '''fairness gaps'''
    for task in tasks:
        # filter out the one does not have attributes
        task_df = df[df[task].notnull()]
        dist_results[task] = (binary_label_dist(task_df), len(task_df))

        # get the unique types of demographic groups
        uniq_types = task_df[task].unique()
        for group in uniq_types:
            # calculate group specific confusion matrix
            group_df = task_df[task_df[task] == group]
            group_key = "_".join([task, str(group)])
            dist_results[group_key] = (binary_label_dist(group_df), len(group_df))
    return dist_results

def task_comb_data(df, task_combs, conditions):
    selected_rows = np.array([True]*len(df))
    for task, condition in zip(task_combs, conditions):
        #print(task, condition)
        selected_rows = selected_rows & (df[task].to_numpy()==condition)
    return selected_rows

def combination_analysis(in_df,
                         task_combs,
                         tasks = ['gender', 'age', 'country', 'ethnicity'],
                         all_labels = False
                         ):
    # save dist at different level
    dist_results = {}    
    
    if all_labels:
        df = in_df[full_label_data(in_df, tasks)]
    else:
        df = in_df

    overall_dist = binary_label_dist(df)
    dist_results["overall"] = (overall_dist, len(df))
    
    '''fairness gaps'''
    for task_comb in task_combs:
        group_combinations = [p for p in product([0, 1], repeat=len(task_comb))]

        for group_comb in group_combinations:
            group_df = df[task_comb_data(df, task_comb, group_comb)]
            group_key = "_".join(task_comb+[str(i) for i in group_comb])
            dist_results[group_key] = (binary_label_dist(group_df), len(group_df))
    return dist_results
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

sys.path.append("../")

import evaluator, random, math, pickle
import pandas as pd
from gerry_eval import bias_attribute_selection_simple, Gerrymandering_eval
from gerry_eval import *
from utils import *

tf.random.set_seed(1)
os.environ["PYTHONHASHSEED"] = str(1)
np.random.seed(1)

print("TensorFlow " + tf.__version__)
print("TFMA " + tfma.VERSION_STRING)
print("TFDS " + tfds.version.__version__)
print("FI " + fi.version.__version__)

max_len = 40
lang = "English"
indices_dir = "./data/indices/" + lang + "/"  #input data path
wt_dir = "./resources/weight/"  #path for word embeddings
res_dir = "./resources/classifier/"   #stores results
BATCH_SIZE = 64

from tensorflow_constrained_optimization.python.rates import basic_expression
from tensorflow_constrained_optimization.python.rates import binary_rates
from tensorflow_constrained_optimization.python.rates import defaults
from tensorflow_constrained_optimization.python.rates import deferred_tensor
from tensorflow_constrained_optimization.python.rates import expression
from tensorflow_constrained_optimization.python.rates import multiclass_rates
from tensorflow_constrained_optimization.python.rates import subsettable_context
from tensorflow_constrained_optimization.python.rates import term
from tensorflow_constrained_optimization.python.rates import loss


def _is_multiclass(context):
    """Returns True iff we're given a multiclass context."""
    if not isinstance(context, subsettable_context.SubsettableContext):
        raise TypeError("context must be a SubsettableContext object")
    raw_context = context.raw_context
    return raw_context.num_classes is not None


def custom_error_rate(
    context,
    penalty_loss=loss.SoftmaxLoss(),
    constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS,
):
    if _is_multiclass(context):
        return multiclass_rates.error_rate(
            context=context, penalty_loss=penalty_loss, constraint_loss=constraint_loss
        )

    print("binary classification problem")

    return binary_rates.error_rate(
        context=context, penalty_loss=penalty_loss, constraint_loss=constraint_loss
    )


def build_rnn():
    """Train, valid, test RNN

    lang: The language name
    odir: output directory of prediction results
    """
    doc_idx = 2
    rnn_size = 200
    epochs = 3
    # load embedding weights
    weights = np.load(wt_dir + lang + ".npy")

    # build model architecture
    text_input = Input(shape=(max_len,), dtype="int32", name="input")
    embeds = Embedding(
        weights.shape[0],
        weights.shape[1],
        weights=[weights],
        input_length=max_len,
        trainable=True,
        name="embedding",
    )(text_input)

    bigru = Bidirectional(
        GRU(rnn_size, kernel_initializer="glorot_uniform"), name="inlp"
    )(embeds)

    dp = Dropout(rate=0.2)(bigru)
    predicts = Dense(1, name="predict")(dp)  # binary prediction
    model = Model(inputs=text_input, outputs=predicts)
    """model.compile(
        loss='binary_crossentropy', optimizer='rmsprop',
        metrics=['accuracy']
    )"""
    print(model.summary())
    return model


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def constrained_optimization(lang, odir, nu=0.0001, weighted=True):
    model_constrained = build_rnn()

    input_tensor = tf.Variable(
        np.zeros((BATCH_SIZE, max_len), dtype="int32"), name="input"
    )

    def predictions():
        return model_constrained(input_tensor)

    labels_tensor = tf.Variable(np.zeros(BATCH_SIZE, dtype="float32"), name="labels")

    groups_tensor_0 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_0"
    )

    groups_tensor_1 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_1"
    )

    groups_tensor_2 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_2"
    )

    groups_tensor_3 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_3"
    )

    groups_tensor_4 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_4"
    )

    groups_tensor_5 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_5"
    )

    groups_tensor_6 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_6"
    )

    groups_tensor_7 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_7"
    )

    groups_tensor_8 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_8"
    )

    groups_tensor_9 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_9"
    )

    groups_tensor_10 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_10"
    )

    groups_tensor_11 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_11"
    )

    groups_tensor_12 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_12"
    )

    groups_tensor_13 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_13"
    )

    groups_tensor_14 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_14"
    )

    groups_tensor_15 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_15"
    )

    groups_tensor_16 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_16"
    )

    groups_tensor_17 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_17"
    )

    groups_tensor_18 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_18"
    )

    groups_tensor_19 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_19"
    )

    groups_tensor_20 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_20"
    )

    groups_tensor_21 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_21"
    )

    groups_tensor_22 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_22"
    )

    groups_tensor_23 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_23"
    )

    groups_tensor_24 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_24"
    )

    groups_tensor_25 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_25"
    )

    groups_tensor_26 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_26"
    )

    groups_tensor_27 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_27"
    )

    groups_tensor_28 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_28"
    )

    groups_tensor_29 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_29"
    )

    groups_tensor_30 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_30"
    )

    groups_tensor_31 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_31"
    )

    groups_tensor_32 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_32"
    )

    groups_tensor_33 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_33"
    )

    groups_tensor_34 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_34"
    )

    groups_tensor_35 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_35"
    )

    groups_tensor_36 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_36"
    )

    groups_tensor_37 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_37"
    )

    groups_tensor_38 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_38"
    )

    groups_tensor_39 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_39"
    )

    groups_tensor_40 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_40"
    )

    groups_tensor_41 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_41"
    )

    groups_tensor_42 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_42"
    )

    groups_tensor_43 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_43"
    )

    groups_tensor_44 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_44"
    )

    groups_tensor_45 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_45"
    )

    groups_tensor_46 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_46"
    )

    groups_tensor_47 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_47"
    )

    groups_tensor_48 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_48"
    )

    groups_tensor_49 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_49"
    )

    groups_tensor_50 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_50"
    )

    groups_tensor_51 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_51"
    )

    groups_tensor_52 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_52"
    )

    groups_tensor_53 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_53"
    )

    groups_tensor_54 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_54"
    )

    groups_tensor_55 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_55"
    )

    groups_tensor_56 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_56"
    )

    groups_tensor_57 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_57"
    )

    groups_tensor_58 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_58"
    )

    groups_tensor_59 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_59"
    )

    groups_tensor_60 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_60"
    )

    groups_tensor_61 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_61"
    )

    groups_tensor_62 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_62"
    )

    groups_tensor_63 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_63"
    )

    groups_tensor_64 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_64"
    )

    groups_tensor_65 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_65"
    )

    groups_tensor_66 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_66"
    )

    groups_tensor_67 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_67"
    )

    groups_tensor_68 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_68"
    )

    groups_tensor_69 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_69"
    )

    groups_tensor_70 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_70"
    )

    groups_tensor_71 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_71"
    )

    groups_tensor_72 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_72"
    )

    groups_tensor_73 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_73"
    )

    groups_tensor_74 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_74"
    )

    groups_tensor_75 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_75"
    )

    groups_tensor_76 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_76"
    )

    groups_tensor_77 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_77"
    )

    groups_tensor_78 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_78"
    )

    groups_tensor_79 = tf.Variable(
        np.zeros(BATCH_SIZE, dtype="float32"), name="groups_79"
    )

    context = tfco.rate_context(predictions, lambda: labels_tensor)
    # Compute the objective using the first stream.
    objective = custom_error_rate(context)

    context_subset_0 = context.subset(lambda: groups_tensor_0 > 0)
    context_subset_1 = context.subset(lambda: groups_tensor_1 > 0)
    context_subset_2 = context.subset(lambda: groups_tensor_2 > 0)
    context_subset_3 = context.subset(lambda: groups_tensor_3 > 0)
    context_subset_4 = context.subset(lambda: groups_tensor_4 > 0)
    context_subset_5 = context.subset(lambda: groups_tensor_5 > 0)
    context_subset_6 = context.subset(lambda: groups_tensor_6 > 0)
    context_subset_7 = context.subset(lambda: groups_tensor_7 > 0)
    context_subset_8 = context.subset(lambda: groups_tensor_8 > 0)
    context_subset_9 = context.subset(lambda: groups_tensor_9 > 0)
    context_subset_10 = context.subset(lambda: groups_tensor_10 > 0)
    context_subset_11 = context.subset(lambda: groups_tensor_11 > 0)
    context_subset_12 = context.subset(lambda: groups_tensor_12 > 0)
    context_subset_13 = context.subset(lambda: groups_tensor_13 > 0)
    context_subset_14 = context.subset(lambda: groups_tensor_14 > 0)
    context_subset_15 = context.subset(lambda: groups_tensor_15 > 0)
    context_subset_16 = context.subset(lambda: groups_tensor_16 > 0)
    context_subset_17 = context.subset(lambda: groups_tensor_17 > 0)
    context_subset_18 = context.subset(lambda: groups_tensor_18 > 0)
    context_subset_19 = context.subset(lambda: groups_tensor_19 > 0)
    context_subset_20 = context.subset(lambda: groups_tensor_20 > 0)
    context_subset_21 = context.subset(lambda: groups_tensor_21 > 0)
    context_subset_22 = context.subset(lambda: groups_tensor_22 > 0)
    context_subset_23 = context.subset(lambda: groups_tensor_23 > 0)
    context_subset_24 = context.subset(lambda: groups_tensor_24 > 0)
    context_subset_25 = context.subset(lambda: groups_tensor_25 > 0)
    context_subset_26 = context.subset(lambda: groups_tensor_26 > 0)
    context_subset_27 = context.subset(lambda: groups_tensor_27 > 0)
    context_subset_28 = context.subset(lambda: groups_tensor_28 > 0)
    context_subset_29 = context.subset(lambda: groups_tensor_29 > 0)
    context_subset_30 = context.subset(lambda: groups_tensor_30 > 0)
    context_subset_31 = context.subset(lambda: groups_tensor_31 > 0)
    context_subset_32 = context.subset(lambda: groups_tensor_32 > 0)
    context_subset_33 = context.subset(lambda: groups_tensor_33 > 0)
    context_subset_34 = context.subset(lambda: groups_tensor_34 > 0)
    context_subset_35 = context.subset(lambda: groups_tensor_35 > 0)
    context_subset_36 = context.subset(lambda: groups_tensor_36 > 0)
    context_subset_37 = context.subset(lambda: groups_tensor_37 > 0)
    context_subset_38 = context.subset(lambda: groups_tensor_38 > 0)
    context_subset_39 = context.subset(lambda: groups_tensor_39 > 0)
    context_subset_40 = context.subset(lambda: groups_tensor_40 > 0)
    context_subset_41 = context.subset(lambda: groups_tensor_41 > 0)
    context_subset_42 = context.subset(lambda: groups_tensor_42 > 0)
    context_subset_43 = context.subset(lambda: groups_tensor_43 > 0)
    context_subset_44 = context.subset(lambda: groups_tensor_44 > 0)
    context_subset_45 = context.subset(lambda: groups_tensor_45 > 0)
    context_subset_46 = context.subset(lambda: groups_tensor_46 > 0)
    context_subset_47 = context.subset(lambda: groups_tensor_47 > 0)
    context_subset_48 = context.subset(lambda: groups_tensor_48 > 0)
    context_subset_49 = context.subset(lambda: groups_tensor_49 > 0)
    context_subset_50 = context.subset(lambda: groups_tensor_50 > 0)
    context_subset_51 = context.subset(lambda: groups_tensor_51 > 0)
    context_subset_52 = context.subset(lambda: groups_tensor_52 > 0)
    context_subset_53 = context.subset(lambda: groups_tensor_53 > 0)
    context_subset_54 = context.subset(lambda: groups_tensor_54 > 0)
    context_subset_55 = context.subset(lambda: groups_tensor_55 > 0)
    context_subset_56 = context.subset(lambda: groups_tensor_56 > 0)
    context_subset_57 = context.subset(lambda: groups_tensor_57 > 0)
    context_subset_58 = context.subset(lambda: groups_tensor_58 > 0)
    context_subset_59 = context.subset(lambda: groups_tensor_59 > 0)
    context_subset_60 = context.subset(lambda: groups_tensor_60 > 0)
    context_subset_61 = context.subset(lambda: groups_tensor_61 > 0)
    context_subset_62 = context.subset(lambda: groups_tensor_62 > 0)
    context_subset_63 = context.subset(lambda: groups_tensor_63 > 0)
    context_subset_64 = context.subset(lambda: groups_tensor_64 > 0)
    context_subset_65 = context.subset(lambda: groups_tensor_65 > 0)
    context_subset_66 = context.subset(lambda: groups_tensor_66 > 0)
    context_subset_67 = context.subset(lambda: groups_tensor_67 > 0)
    context_subset_68 = context.subset(lambda: groups_tensor_68 > 0)
    context_subset_69 = context.subset(lambda: groups_tensor_69 > 0)
    context_subset_70 = context.subset(lambda: groups_tensor_70 > 0)
    context_subset_71 = context.subset(lambda: groups_tensor_71 > 0)
    context_subset_72 = context.subset(lambda: groups_tensor_72 > 0)
    context_subset_73 = context.subset(lambda: groups_tensor_73 > 0)
    context_subset_74 = context.subset(lambda: groups_tensor_74 > 0)
    context_subset_75 = context.subset(lambda: groups_tensor_75 > 0)
    context_subset_76 = context.subset(lambda: groups_tensor_76 > 0)
    context_subset_77 = context.subset(lambda: groups_tensor_77 > 0)
    context_subset_78 = context.subset(lambda: groups_tensor_78 > 0)
    context_subset_79 = context.subset(lambda: groups_tensor_79 > 0)

    inter_data = {}

    data = {"x": [], "y": [], "gender": [], "age": [], "country": [], "ethnicity": []}
    doc_idx = 2
    with open(indices_dir + "train.tsv") as dfile:
        dfile.readline()
        for line in dfile:
            line = line.strip().split("\t")
            # split indices
            data["x"].append(list(map(int, line[doc_idx].split())))
            data["y"].append(int(line[-1]))
            data["gender"].append(int(line[4]))  # gender
            data["age"].append(int(line[5]))  # age
            data["country"].append(int(line[8]))  # country
            data["ethnicity"].append(int(line[9]))  # ethnicity

    _data = pd.DataFrame(data)

    tasks = ["gender", "age", "country", "ethnicity"]
    attribute_distinct_labels = {
        attribute: list(_data[~_data[attribute].isnull()][attribute].unique())
        for attribute in tasks
    }
    print(attribute_distinct_labels)
    gerry_combs = Gerrymandering_groups(
        attributes=tasks, attribute_distinct_labels=attribute_distinct_labels
    )
    group_comb_itr = 0
    # iterate all gerry combs
    for task_comb, group_comb in tqdm(gerry_combs):
        group_indices = task_comb_data(_data, task_comb, group_comb)
        group_key = "@".join([str(i) + str(j) for i, j in zip(task_comb, group_comb)])
        subdf = _data[group_indices]
        _labelsize = 0
        for _ylabel in subdf["y"]:
            if _ylabel == 1:
                _labelsize += 1
        inter_data[group_comb_itr] = len(subdf) / _labelsize
        group_comb_itr += 1

    constraints = []

    contexts_available = [
        context_subset_0,
        context_subset_1,
        context_subset_2,
        context_subset_3,
        context_subset_4,
        context_subset_5,
        context_subset_6,
        context_subset_7,
        context_subset_8,
        context_subset_9,
        context_subset_10,
        context_subset_11,
        context_subset_12,
        context_subset_13,
        context_subset_14,
        context_subset_15,
        context_subset_16,
        context_subset_17,
        context_subset_18,
        context_subset_19,
        context_subset_20,
        context_subset_21,
        context_subset_22,
        context_subset_23,
        context_subset_24,
        context_subset_25,
        context_subset_26,
        context_subset_27,
        context_subset_28,
        context_subset_29,
        context_subset_30,
        context_subset_31,
        context_subset_32,
        context_subset_33,
        context_subset_34,
        context_subset_35,
        context_subset_36,
        context_subset_37,
        context_subset_38,
        context_subset_39,
        context_subset_40,
        context_subset_41,
        context_subset_42,
        context_subset_43,
        context_subset_44,
        context_subset_45,
        context_subset_46,
        context_subset_47,
        context_subset_48,
        context_subset_49,
        context_subset_50,
        context_subset_51,
        context_subset_52,
        context_subset_53,
        context_subset_54,
        context_subset_55,
        context_subset_56,
        context_subset_57,
        context_subset_58,
        context_subset_59,
        context_subset_60,
        context_subset_61,
        context_subset_62,
        context_subset_63,
        context_subset_64,
        context_subset_65,
        context_subset_66,
        context_subset_67,
        context_subset_68,
        context_subset_69,
        context_subset_70,
        context_subset_71,
        context_subset_72,
        context_subset_73,
        context_subset_74,
        context_subset_75,
        context_subset_76,
        context_subset_77,
        context_subset_78,
        context_subset_79,
    ]

    for i in range(len(contexts_available)):
        if weighted:
            constraints.append(
                inter_data[i]
                * (
                    tfco.true_positive_rate(context)
                    - tfco.true_positive_rate(contexts_available[i])
                )
                <= nu
            )
            constraints.append(
                inter_data[i]
                * (
                    tfco.true_positive_rate(contexts_available[i])
                    - tfco.true_positive_rate(context)
                )
                <= nu
            )
        else:
            constraints.append(
                (
                    tfco.true_positive_rate(context)
                    - tfco.true_positive_rate(contexts_available[i])
                )
                <= nu
            )
            constraints.append(
                (
                    tfco.true_positive_rate(contexts_available[i])
                    - tfco.true_positive_rate(context)
                )
                <= nu
            )

    problem = tfco.RateMinimizationProblem(objective, constraints)

    optimizer = tfco.LagrangianOptimizerV2(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        num_constraints=problem.num_constraints,
    )

    var_list = (
        model_constrained.trainable_weights
        + problem.trainable_variables
        + optimizer.trainable_variables()
    )

    NUM_ITERATIONS = 100  # Number of training iterations.
    objective_list = []
    violations_list = []
    e = 0

    best_valid_f1 = 0.0
    best_model = None

    best_hmean_v = 0
    best_hmean_epoch = 0
    best_bias_v = 1
    best_bias_epoch = 0
    validation_results = {}
    best_perf = 0
    best_perf_epoch = 0

    while e < NUM_ITERATIONS:
        train_iter = evaluator.data_iter_gerry_adv(
            indices_dir + "train.tsv", batch_size=BATCH_SIZE
        )

        for (
            _,
            x_train,
            y_train,
            y_gerry_class_0,
            y_gerry_class_1,
            y_gerry_class_2,
            y_gerry_class_3,
            y_gerry_class_4,
            y_gerry_class_5,
            y_gerry_class_6,
            y_gerry_class_7,
            y_gerry_class_8,
            y_gerry_class_9,
            y_gerry_class_10,
            y_gerry_class_11,
            y_gerry_class_12,
            y_gerry_class_13,
            y_gerry_class_14,
            y_gerry_class_15,
            y_gerry_class_16,
            y_gerry_class_17,
            y_gerry_class_18,
            y_gerry_class_19,
            y_gerry_class_20,
            y_gerry_class_21,
            y_gerry_class_22,
            y_gerry_class_23,
            y_gerry_class_24,
            y_gerry_class_25,
            y_gerry_class_26,
            y_gerry_class_27,
            y_gerry_class_28,
            y_gerry_class_29,
            y_gerry_class_30,
            y_gerry_class_31,
            y_gerry_class_32,
            y_gerry_class_33,
            y_gerry_class_34,
            y_gerry_class_35,
            y_gerry_class_36,
            y_gerry_class_37,
            y_gerry_class_38,
            y_gerry_class_39,
            y_gerry_class_40,
            y_gerry_class_41,
            y_gerry_class_42,
            y_gerry_class_43,
            y_gerry_class_44,
            y_gerry_class_45,
            y_gerry_class_46,
            y_gerry_class_47,
            y_gerry_class_48,
            y_gerry_class_49,
            y_gerry_class_50,
            y_gerry_class_51,
            y_gerry_class_52,
            y_gerry_class_53,
            y_gerry_class_54,
            y_gerry_class_55,
            y_gerry_class_56,
            y_gerry_class_57,
            y_gerry_class_58,
            y_gerry_class_59,
            y_gerry_class_60,
            y_gerry_class_61,
            y_gerry_class_62,
            y_gerry_class_63,
            y_gerry_class_64,
            y_gerry_class_65,
            y_gerry_class_66,
            y_gerry_class_67,
            y_gerry_class_68,
            y_gerry_class_69,
            y_gerry_class_70,
            y_gerry_class_71,
            y_gerry_class_72,
            y_gerry_class_73,
            y_gerry_class_74,
            y_gerry_class_75,
            y_gerry_class_76,
            y_gerry_class_77,
            y_gerry_class_78,
            y_gerry_class_79,
        ) in train_iter:
            if x_train.shape[0] < BATCH_SIZE:
                continue
            input_tensor.assign(x_train)
            labels_tensor.assign(y_train)
            groups_tensor_0.assign(np.array(y_gerry_class_0))
            groups_tensor_1.assign(np.array(y_gerry_class_1))
            groups_tensor_2.assign(np.array(y_gerry_class_2))
            groups_tensor_3.assign(np.array(y_gerry_class_3))
            groups_tensor_4.assign(np.array(y_gerry_class_4))
            groups_tensor_5.assign(np.array(y_gerry_class_5))
            groups_tensor_6.assign(np.array(y_gerry_class_6))
            groups_tensor_7.assign(np.array(y_gerry_class_7))
            groups_tensor_8.assign(np.array(y_gerry_class_8))
            groups_tensor_9.assign(np.array(y_gerry_class_9))
            groups_tensor_10.assign(np.array(y_gerry_class_10))
            groups_tensor_11.assign(np.array(y_gerry_class_11))
            groups_tensor_12.assign(np.array(y_gerry_class_12))
            groups_tensor_13.assign(np.array(y_gerry_class_13))
            groups_tensor_14.assign(np.array(y_gerry_class_14))
            groups_tensor_15.assign(np.array(y_gerry_class_15))
            groups_tensor_16.assign(np.array(y_gerry_class_16))
            groups_tensor_17.assign(np.array(y_gerry_class_17))
            groups_tensor_18.assign(np.array(y_gerry_class_18))
            groups_tensor_19.assign(np.array(y_gerry_class_19))
            groups_tensor_20.assign(np.array(y_gerry_class_20))
            groups_tensor_21.assign(np.array(y_gerry_class_21))
            groups_tensor_22.assign(np.array(y_gerry_class_22))
            groups_tensor_23.assign(np.array(y_gerry_class_23))
            groups_tensor_24.assign(np.array(y_gerry_class_24))
            groups_tensor_25.assign(np.array(y_gerry_class_25))
            groups_tensor_26.assign(np.array(y_gerry_class_26))
            groups_tensor_27.assign(np.array(y_gerry_class_27))
            groups_tensor_28.assign(np.array(y_gerry_class_28))
            groups_tensor_29.assign(np.array(y_gerry_class_29))
            groups_tensor_30.assign(np.array(y_gerry_class_30))
            groups_tensor_31.assign(np.array(y_gerry_class_31))
            groups_tensor_32.assign(np.array(y_gerry_class_32))
            groups_tensor_33.assign(np.array(y_gerry_class_33))
            groups_tensor_34.assign(np.array(y_gerry_class_34))
            groups_tensor_35.assign(np.array(y_gerry_class_35))
            groups_tensor_36.assign(np.array(y_gerry_class_36))
            groups_tensor_37.assign(np.array(y_gerry_class_37))
            groups_tensor_38.assign(np.array(y_gerry_class_38))
            groups_tensor_39.assign(np.array(y_gerry_class_39))
            groups_tensor_40.assign(np.array(y_gerry_class_40))
            groups_tensor_41.assign(np.array(y_gerry_class_41))
            groups_tensor_42.assign(np.array(y_gerry_class_42))
            groups_tensor_43.assign(np.array(y_gerry_class_43))
            groups_tensor_44.assign(np.array(y_gerry_class_44))
            groups_tensor_45.assign(np.array(y_gerry_class_45))
            groups_tensor_46.assign(np.array(y_gerry_class_46))
            groups_tensor_47.assign(np.array(y_gerry_class_47))
            groups_tensor_48.assign(np.array(y_gerry_class_48))
            groups_tensor_49.assign(np.array(y_gerry_class_49))
            groups_tensor_50.assign(np.array(y_gerry_class_50))
            groups_tensor_51.assign(np.array(y_gerry_class_51))
            groups_tensor_52.assign(np.array(y_gerry_class_52))
            groups_tensor_53.assign(np.array(y_gerry_class_53))
            groups_tensor_54.assign(np.array(y_gerry_class_54))
            groups_tensor_55.assign(np.array(y_gerry_class_55))
            groups_tensor_56.assign(np.array(y_gerry_class_56))
            groups_tensor_57.assign(np.array(y_gerry_class_57))
            groups_tensor_58.assign(np.array(y_gerry_class_58))
            groups_tensor_59.assign(np.array(y_gerry_class_59))
            groups_tensor_60.assign(np.array(y_gerry_class_60))
            groups_tensor_61.assign(np.array(y_gerry_class_61))
            groups_tensor_62.assign(np.array(y_gerry_class_62))
            groups_tensor_63.assign(np.array(y_gerry_class_63))
            groups_tensor_64.assign(np.array(y_gerry_class_64))
            groups_tensor_65.assign(np.array(y_gerry_class_65))
            groups_tensor_66.assign(np.array(y_gerry_class_66))
            groups_tensor_67.assign(np.array(y_gerry_class_67))
            groups_tensor_68.assign(np.array(y_gerry_class_68))
            groups_tensor_69.assign(np.array(y_gerry_class_69))
            groups_tensor_70.assign(np.array(y_gerry_class_70))
            groups_tensor_71.assign(np.array(y_gerry_class_71))
            groups_tensor_72.assign(np.array(y_gerry_class_72))
            groups_tensor_73.assign(np.array(y_gerry_class_73))
            groups_tensor_74.assign(np.array(y_gerry_class_74))
            groups_tensor_75.assign(np.array(y_gerry_class_75))
            groups_tensor_76.assign(np.array(y_gerry_class_76))
            groups_tensor_77.assign(np.array(y_gerry_class_77))
            groups_tensor_78.assign(np.array(y_gerry_class_78))
            groups_tensor_79.assign(np.array(y_gerry_class_79))
            optimizer.minimize(problem, var_list=var_list)
            objective = problem.objective()
            violations = problem.constraints()
        objective_list.append(objective)
        violations_list.append(violations)

        # valid model to find the best model
        print("---------------Validation------------")
        valid_iter = evaluator.data_iter_adv(
            indices_dir + "valid.tsv", batch_size=BATCH_SIZE, if_shuffle=False
        )
        y_preds = []
        y_valids = []
        y_pred_prob = []
        valid_gender = []
        valid_age = []
        valid_country = []
        valid_ethnicity = []

        for (
            _,
            x_valid,
            y_valid,
            y_va_gender,
            y_va_age,
            y_va_country,
            y_va_ethnicity,
        ) in valid_iter:
            tmp_preds = model_constrained.predict([x_valid])
            for item_tmp in tmp_preds:
                y_preds.append(int(round(sigmoid(item_tmp[0]))))
                y_pred_prob.append(sigmoid(item_tmp[0]))
            y_valids.extend(y_valid)
            valid_gender.extend(y_va_gender)
            valid_age.extend(y_va_age)
            valid_country.extend(y_va_country)
            valid_ethnicity.extend(y_va_ethnicity)

        valid_f1 = f1_score(
            y_true=y_valids,
            y_pred=y_preds,
            average="macro",
        )
        print("Validating f1-macro score: " + str(valid_f1))

        results = {
            "pred": y_preds,
            "pred_prob": y_pred_prob,
            "gender": valid_gender,
            "age": valid_age,
            "country": valid_country,
            "ethnicity": valid_ethnicity,
            "label": y_valids,
        }

        results_df = pd.DataFrame(results)

        gerry_score = Gerrymandering_eval(
            results_df,
            tasks=["gender", "age", "country", "ethnicity"],
            label="label",
            pred="pred",
            pred_prob="pred_prob",
            all_labels=True,
            print_results=False,
        )

        selected_attributes = bias_attribute_selection_simple(
            gerry_eval_results=gerry_score,
            attributes=["gender", "age", "country", "ethnicity"],
            subgroup_min_size=0,
            df=results_df,
        )
        avg_violation = np.mean([i[0] for i in selected_attributes["GAP_t11"]])
        max_violation = np.max([i[0] for i in selected_attributes["GAP_t11"]])

        validation_results[e] = (avg_violation, valid_f1)

        from scipy.stats import hmean

        if hmean([valid_f1, 1 - avg_violation]) > best_hmean_v:
            best_hmean_epoch = e
            best_hmean_v = hmean([valid_f1, 1 - avg_violation])

        if valid_f1 > best_perf:
            best_perf_epoch = e
            best_perf = valid_f1

        if avg_violation < best_bias_v:
            best_bias_epoch = e
            best_bias_v = avg_violation

        print(e - best_bias_epoch, e, valid_f1, avg_violation, max_violation)
        if e - best_bias_epoch >= 5:
            print("check this model: 5", best_bias_epoch, e - best_bias_epoch)
        if e - best_bias_epoch >= 10:
            print("check this model: 10", best_bias_epoch, e - best_bias_epoch)

        best_valid_f1 = valid_f1
        best_model = model_constrained
        e += 1

        print("--------------Test--------------------")
        y_preds = []
        y_probs = []

        test_iter = evaluator.data_iter(
            indices_dir + "test.tsv", batch_size=BATCH_SIZE, if_shuffle=False
        )

        for _, x_test, y_test in test_iter:
            tmp_preds = best_model.predict([x_test])
            for item_tmp in tmp_preds:
                y_probs.append(sigmoid(item_tmp[0]))
                y_preds.append(int(round(sigmoid(item_tmp[0]))))

        with open(
            odir + "hatespeech_tfco_gerry_cw_{}_tw_{}_{}.tsv".format(weighted, e, nu),
            "w",
        ) as wfile:
            with open(indices_dir + "test.tsv") as dfile:
                wfile.write(dfile.readline().strip() + "\tpred\tpred_prob\n")
                for idx, line in enumerate(dfile):
                    wfile.write(
                        line.strip()
                        + "\t"
                        + str(y_preds[idx])
                        + "\t"
                        + str(y_probs[idx])
                        + "\n"
                    )

        # save the predicted results
        evaluator.eval(
            odir + "hatespeech_tfco_gerry_cw_{}_tw_{}_{}.tsv".format(weighted, e, nu),
            odir + "hatespeech_tfco_gerry_cw_{}_tw_{}_{}.score".format(weighted, e, nu),
        )

    with open(
        odir + "validation_hatespeech_tfco_gerry_cw_{}_{}.pkl".format(weighted, nu),
        "wb",
    ) as f:
        pickle.dump(validation_results, f)

    best_index = tfco.find_best_candidate_index(
        np.array(objective_list), np.array(violations_list), rank_objectives=False
    )
    print(best_index, "best index from TFCO, not ranked")

    best_index = tfco.find_best_candidate_index(
        np.array(objective_list), np.array(violations_list), rank_objectives=True
    )
    print(best_index, "best index from TFCO, ranked")

    print("best epochs", best_perf_epoch, best_hmean_epoch, best_bias_epoch)


if __name__ == "__main__":

    odir = "/output/directory/"

    for weighted in [True, False]:
        # for weighted in [False]:
        for nu in [
            0.0001,
            0.0005,
            0.0007,
            0.001,
            0.005,
            0.007,
            0.01,
            0.03,
            0.05,
            0.07,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            1,
        ]:
            model = constrained_optimization(lang, odir, nu=nu, weighted=weighted)

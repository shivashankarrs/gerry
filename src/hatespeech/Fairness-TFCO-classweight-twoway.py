import os
import sys
import tempfile
import urllib
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow_constrained_optimization as tfco
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

tf.random.set_seed(1)
os.environ["PYTHONHASHSEED"] = str(1)
np.random.seed(1)

print("TensorFlow " + tf.__version__)
"""print("TFMA " + tfma.VERSION_STRING)
print("TFDS " + tfds.version.__version__)
print("FI " + fi.version.__version__)"""

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


def intersectional_label_encoder(gender, age, country, ethnicity):
    encoder = {}
    indx = 0

    for g in list(set(gender)):
        for a in list(set(age)):
            for c in list(set(country)):
                for e in list(set(ethnicity)):
                    l = (g, a, c, e)
                    encoder[l] = indx
                    indx += 1
    return encoder


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

    context = tfco.rate_context(predictions, lambda: labels_tensor)
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

    inter_data = {}
    labels = {}

    inter_enc = intersectional_label_encoder([0, 1], [0, 1], [0, 1], [0, 1])

    with open(indices_dir + "train.tsv") as dfile:
        dfile.readline()
        for line in dfile:
            line = line.strip().split("\t")
            l = (int(line[4]), int(line[5]), int(line[8]), int(line[9]))
            if int(line[-1]) == 1:
                if inter_enc[l] not in labels:
                    labels[inter_enc[l]] = 1
                else:
                    labels[inter_enc[l]] += 1
            if inter_enc[l] not in inter_data:
                inter_data[inter_enc[l]] = 1
            else:
                inter_data[inter_enc[l]] += 1

        for k, v in inter_data.items():
            inter_data[k] = inter_data[k] / labels[k]

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
        train_iter = evaluator.data_iter_mc_intersectional_adv(
            indices_dir + "train.tsv", batch_size=BATCH_SIZE
        )
        _lmean = 0
        _ctr = 0
        for (
            _,
            x_train,
            y_train,
            y_tr_gender,
            y_tr_age,
            y_tr_country,
            y_tr_ethnicity,
            y_tr_intersectional,
        ) in train_iter:
            if x_train.shape[0] < BATCH_SIZE:
                continue
            input_tensor.assign(x_train)
            labels_tensor.assign(y_train)
            groups_tensor_0.assign(np.array(y_tr_intersectional[:, 0]))
            groups_tensor_1.assign(np.array(y_tr_intersectional[:, 1]))
            groups_tensor_2.assign(np.array(y_tr_intersectional[:, 2]))
            groups_tensor_3.assign(np.array(y_tr_intersectional[:, 3]))
            groups_tensor_4.assign(np.array(y_tr_intersectional[:, 4]))
            groups_tensor_5.assign(np.array(y_tr_intersectional[:, 5]))
            groups_tensor_6.assign(np.array(y_tr_intersectional[:, 6]))
            groups_tensor_7.assign(np.array(y_tr_intersectional[:, 7]))
            groups_tensor_8.assign(np.array(y_tr_intersectional[:, 8]))
            groups_tensor_9.assign(np.array(y_tr_intersectional[:, 9]))
            groups_tensor_10.assign(np.array(y_tr_intersectional[:, 10]))
            groups_tensor_11.assign(np.array(y_tr_intersectional[:, 11]))
            groups_tensor_12.assign(np.array(y_tr_intersectional[:, 12]))
            groups_tensor_13.assign(np.array(y_tr_intersectional[:, 13]))
            groups_tensor_14.assign(np.array(y_tr_intersectional[:, 14]))
            groups_tensor_15.assign(np.array(y_tr_intersectional[:, 15]))
            optimizer.minimize(problem, var_list=var_list)
            objective = problem.objective()
            violations = problem.constraints()
            _lmean += np.mean(violations)
            _ctr += 1
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
            odir
            + "hatespeech_tfco_intersectional_cw_{}_tw_{}_{}.tsv".format(
                weighted, e, nu
            ),
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
            odir
            + "hatespeech_tfco_intersectional_cw_{}_tw_{}_{}.tsv".format(
                weighted, e, nu
            ),
            odir
            + "hatespeech_tfco_intersectional_cw_{}_tw_{}_{}.score".format(
                weighted, e, nu
            ),
        )

    with open(
        odir
        + "validation_hatespeech_tfco_intersectional_cw_{}_{}.pkl".format(weighted, nu),
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

    # for weighted in [False]:
    # for nu in [0.0001, 0.0005, 0.0007, 0.001, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1]:

    for nu in [0.001, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1]:
        for weighted in [True, False]:
            model = constrained_optimization(lang, odir, nu=nu, weighted=weighted)

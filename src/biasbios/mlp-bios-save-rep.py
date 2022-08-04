"""MLP --- run from old path.
"""
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
from keras.utils import to_categorical

tf.random.set_seed(1)
os.environ["PYTHONHASHSEED"] = str(1)
np.random.seed(1)

MAX_SEQUENCE_LENGTH = 768


res_dir = "./data/bios/representation/" #stores the hidden representations


def build_mlp(xtrain, ytrain, xvalid, yvalid, xtest, ytest):
    """
    Train, valid, test for MLP model. This is the unconstrained model, which doesn't optimize for fairness. Representations are saved for running INLP approach.
    """
    hidden_size = 300
    epochs = 10

    # build model architecture
    text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="float32", name="input")

    mlp = Dense(hidden_size, activation="relu")(text_input)  # binary prediction

    dp = Dropout(rate=0.1, name="inlp")(mlp)

    # predicts = Dense(
    #    28, activation='softmax', name='predict'
    # )(dp) # binary prediction

    predicts = Dense(1, activation="sigmoid", name="predict")(dp)  # binary prediction

    model = Model(inputs=text_input, outputs=predicts)
    repmodel = Model(inputs=text_input, outputs=model.get_layer(name="inlp").output)

    model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    print(model.summary())

    history = model.fit(
        xtrain,
        ytrain,
        validation_data=(xvalid, yvalid),
        epochs=epochs,
        verbose=1,
        batch_size=64,
    )

    print("--------------Test--------------------")
    y_preds = []
    y_probs = []

    tmp_preds = model.predict([xtest])
    for item_tmp in tmp_preds:
        y_probs.append(item_tmp[0])
        y_preds.append(int(round(item_tmp[0])))

    ytrue = []
    for _y in ytest:
        ytrue.append(np.argmax(_y))

    train_rep_preds = repmodel.predict([xtrain])
    val_rep_preds = repmodel.predict([xvalid])
    test_rep_preds = repmodel.predict([xtest])

    print("fscore:", f1_score(y_preds, ytest, average="macro"))
    print(train_rep_preds.shape, val_rep_preds.shape, test_rep_preds.shape)

    np.save(res_dir + "train_rep_bios_mlp_tc.npy", train_rep_preds)
    np.save(res_dir + "valid_rep_bios_mlp_tc.npy", val_rep_preds)
    np.save(res_dir + "test_rep_bios_mlp_tc.npy", test_rep_preds)

    results = {
        "preds": y_preds,
        "pred_prob": y_probs,
        "gender": [],
        "economy": [],
        "label": ytest,
    }
    return results


def load_dictionary(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    text2index = {}
    for line in lines:
        k, v = line.strip().split("\t")
        v = int(v)
        text2index[k] = v
    return text2index


if __name__ == "__main__":

    tp_output_dir = "./data/bios/output/"
    biasbios_train_raw = "emnlp_train_cls_tc.npy"
    biasbios_valid_raw = "emnlp_dev_cls_tc.npy"
    biasbios_test_raw = "emnlp_test_cls_tc.npy"

    biasbios_train_embed = "emnlp_train_bios_twoclass.pickle"
    biasbios_valid_embed = "emnlp_dev_bios_twoclass.pickle"
    biasbios_test_embed = "emnlp_test_bios_twoclass.pickle"

    xtrain = np.load(biasbios_train_raw)
    xvalid = np.load(biasbios_valid_raw)
    xtest = np.load(biasbios_test_raw)

    with open(biasbios_train_embed, "rb"
    ) as f:
        traindata = pickle.load(f)

    with open(biasbios_valid_embed, "rb"
    ) as f:
        valdata = pickle.load(f)

    with open(biasbios_test_embed, "rb"
    ) as f:
        testdata = pickle.load(f)

    from collections import Counter

    print(xtrain.shape, xvalid.shape, xtest.shape)
    print(len(traindata), len(valdata), len(testdata))
    print(traindata[0].keys())
    print(set([d["economy"] for d in traindata]))

    ytrain = []
    yvalid = []
    ytest = []

    train_gender = []
    valid_gender = []
    test_gender = []

    train_economy = []
    valid_economy = []
    test_economy = []

    prof2index = load_dictionary("../resources/professions.txt")

    for data in traindata:
        _label = 1 if data["p"] == "surgeon" else 0
        ytrain.append(_label)
        _gender = 1 if data["g"] == "f" else 0
        train_gender.append(_gender)
        _economy = 1 if data["economy"] == "High income (H)" else 0
        train_economy.append(_economy)

    for data in valdata:
        _label = 1 if data["p"] == "surgeon" else 0
        yvalid.append(_label)
        _gender = 1 if data["g"] == "f" else 0
        valid_gender.append(_gender)
        _economy = 1 if data["economy"] == "High income (H)" else 0
        valid_economy.append(_economy)

    for data in testdata:
        _label = 1 if data["p"] == "surgeon" else 0
        ytest.append(_label)
        _gender = 1 if data["g"] == "f" else 0
        test_gender.append(_gender)
        _economy = 1 if data["economy"] == "High income (H)" else 0
        test_economy.append(_economy)

    print(ytrain[0:5], "before")

    results = build_mlp(
        xtrain, np.array(ytrain), xvalid, np.array(yvalid), xtest, np.array(ytest)
    )
    results["economy"] = test_economy
    results["gender"] = test_gender
    results_df = pd.DataFrame(results)
    results_df.to_csv(tp_output_dir + "bios_mlp_results.tsv", index=False)

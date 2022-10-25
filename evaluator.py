import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import StratifiedKFold

def build_kfold(data1, data2, labels, k=10, shuffle=False, seed):
    np.random.seed(seed)
    np.random.shuffle(data1)
    np.random.seed(seed)
    np.random.shuffle(data2)
    np.random.seed(seed)
    np.random.shuffle(labels)

    skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=None)
    kfoldList = []
    for train_index, test_index in skf.split(data1, labels):
        X1_train, X1_test = data1[train_index], data1[test_index]
        X2_train, X2_test = data2[train_index], data2[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        kfoldList.append({
            "X1_train": X1_train,
            "X1_test": X1_test,
            "X2_train": X2_train,
            "X2_test": X2_test,
            "y_train": y_train,
            "y_test": y_test
        })
    return kfoldList


def pred2label(y_pred):
    y_pred = np.round(np.clip(y_pred, 0, 1))

    return y_pred

def precision(y_true, y_pred):
    print(y_pred)
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)

    precision = tp / (tp+fp+K.epsilon())

    return precision

def recall(y_true, y_pred):
    print(y_pred)
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))

    tp = K.sum(y_pos * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    recall = tp / (tp+fn+K.epsilon())

    return recall

def f1score(y_true, y_pred):
    print(y_pred)
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp+fn+K.epsilon())

    f1score = 2*(precision*recall)/(precision+recall+K.epsilon())

    return f1score

def aucScore(y_true, y_pred):

    ptas = tf.stack([binary_PTA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.stack([binary_PFA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.concat([tf.ones((1,)), pfas], axis=0)
    binSizes = -(pfas[1:] - pfas[:-1])
    s = ptas * binSizes
    return K.sum(s, axis=0)

def acc(y_true, y_pred):
    print(y_pred)
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    acc = (tp + tn) / (tp + tn + fp + fn + K.epsilon())
    return acc
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P

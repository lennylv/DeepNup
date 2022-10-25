import argparse
import model_dn
import evaluator
import os
import pickle
import numpy as np
import tensorflow as tf
import pdb


parser = argparse.ArgumentParser(description='Nucleosome Classification Experiment')
parser.add_argument('-p', '--path', dest='path', type=str, default=r"D:\DeepNup\code_and_data\data\setting1\pickle_H",
                    help='Pickle file Path')

parser.add_argument('-ohn', '--ohnuc', dest='nuc_one_hot_Pickle', type=str, default="one_hot_nuc.pickle",
                    help='Nucleosome filename')
parser.add_argument('-ohl', '--ohlin', dest='link_one_hot_Pickle', type=str, default="one_hot_link.pickle",
                    help='Linker filename')

parser.add_argument('-pn', '--pnuc', dest='nuc_PseTNC_Pickle', type=str, default="PseTNC_nuc.pickle",
                    help='Nucleosome filename')
parser.add_argument('-pl', '--plin', dest='link_PseTNC_Pickle', type=str, default="PseTNC_link.pickle",
                    help='Linker filename')

parser.add_argument('-o', '--out', dest='outPath', type=str, default=r"D:\DeepNup\result",
                    help='Output file Path')
parser.add_argument('-e', '--experiments', dest='exp', default='Experiment_H',
                    help='Experiments Name')
parser.add_argument('-f', '-foldName', dest='foldName', default="folds.pickle",
                    help='Folds Filename')

args = parser.parse_args()
inPath = args.path
nuc_one_hot = args.nuc_one_hot_Pickle
link_one_hot = args.link_one_hot_Pickle
nuc_PseTNC = args.nuc_PseTNC_Pickle
link_PseTNC = args.link_PseTNC_Pickle

outPath = args.outPath
expName = args.exp
foldName = args.foldName

metricsList = [evaluator.acc, evaluator.precision, evaluator.recall, evaluator.f1score, evaluator.aucScore]

epochs = 200
batch_size = 64
shuffle = False 
seed = None  

# Number of the species for Roc Curve Figure
fNum = 1

if (os.path.exists(os.path.join(outPath, "elapsed.json"))):
    os.path.join(outPath, "elapsed.json")

m = "dn"

# Create and set model save dir
modelPath = os.path.join(outPath, expName, "models", m)
if (not os.path.isdir(modelPath)):
    os.makedirs(modelPath)

# Load nucleosome and linker and then create dataset and  class labels
with open(os.path.join(inPath, nuc_one_hot), "rb") as fp:
    nuc_one_hot_list = pickle.load(fp)
with open(os.path.join(inPath, link_one_hot), "rb") as fp:
    link_one_hot_list = pickle.load(fp)

with open(os.path.join(inPath, nuc_PseTNC), "rb") as fp:
    nuc_PseTNC_list = pickle.load(fp)
with open(os.path.join(inPath, link_PseTNC), "rb") as fp:
    link_PseTNC_list = pickle.load(fp)


input_size = (147, 4)
labels = np.concatenate(
    (np.ones((len(nuc_one_hot_list), 1), dtype=np.float32), np.zeros((len(link_one_hot_list), 1), dtype=np.float32)),
    axis=0)
one_hot_feature = np.concatenate((nuc_one_hot_list, link_one_hot_list), 0)
PseTNC_feature = np.concatenate((nuc_PseTNC_list, link_PseTNC_list), 0)

data1 = one_hot_feature
data2 = PseTNC_feature


# Create folder to save fold dataset and build kfold
foldPath = os.path.join(outPath, expName, foldName)
folds = evaluator.build_kfold(data1, data2, labels, k=10, shuffle=shuffle, seed=seed)
# pdb.set_trace()
with open(foldPath, "wb") as fp:
    pickle.dump(folds, fp)


evaluations = {
    "Accuracy": [],
    "Precision": [],
    "TPR": [],
    "FPR": [],
    "AUC": [],
    "Sensitivity": [],
    "Specificity": [],
    "MCC": []
}

i = 1
for fold in folds:
    
    tf.keras.backend.clear_session()
    modelCallbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(modelPath, "{}_bestModel-fold{}.hdf5".format(m, i)),
                                           monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False,
                                           mode='auto', period=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=9, verbose=0, mode='auto',
                                         baseline=None, restore_best_weights=False)
    ]
    model = model_dn.dn(metrics=metricsList)
    print(model.summary())
    print('#############fold'+str(i)+'################')
    model.fit(x=[fold["X1_train"], fold['X2_train']], y=fold["y_train"],
              batch_size=batch_size, epochs=epochs, verbose=1, callbacks=modelCallbacks, validation_split=0.05,
              validation_freq=1)

    i += 1

del model





import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
import evaluator
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import argparse
from tensorflow.keras.models import load_model


parser = argparse.ArgumentParser(description='Nucleosome Classification Experiment')
parser.add_argument('-pn', '--plot', dest='plotName', type=str, default="0",
                    help='Plot Title')
parser.add_argument('-p','--path', dest='path', type=str, default=r"D:\experiment",
                    help='Model file Path')
parser.add_argument('-e', '--experiments', dest='exp', default=['Experiment_H',
                                                                'Experiment_D',
                                                                'Experiment_C'],
                    help='Experiments Name')
parser.add_argument('-f', '-foldName', dest='foldName', default="folds.pickle",
                    help='Folds Filename')


args = parser.parse_args()
inPath = args.path
expNames = args.exp
foldName = args.foldName
plotName = args.plotName
del args


names = ['H.sapiens',
         'D.melanogaster',
         'C.elegans']
colors = ['crimson',
          'orange',
          'steelblue']


for (expName, name, colorname) in zip(expNames, names, colors):
    evaluations = {"AUC": []}
    AUCMean = []
    i = 1

    m = "dn"

    fig = plt.figure(i, figsize=(12, 10))
    ax = fig.add_subplot(111)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    modelPath = os.path.join(inPath, expName, "models", m)
    foldPath = os.path.join(inPath, expName)

    if (not os.path.exists(os.path.join(foldPath, foldName))):
        print("Error: Folds not Found in {}".format(os.path.join(foldPath, foldName)))
    else:
        with open(os.path.join(foldPath, foldName), "rb") as fp:
            folds = pickle.load(fp)
        i = 1
        for fold in folds:
            # Check if model alredy exists
            if (os.path.exists(os.path.join(modelPath, "{}_bestModel-fold{}.hdf5".format(m, i)))):

                # load json and create model
                model = load_model(os.path.join(modelPath, "{}_bestModel-fold{}.hdf5".format(m, i)),
                                   custom_objects={"precision": evaluator.precision, "recall": evaluator.recall,
                                                   "f1score": evaluator.f1score, "aucScore": evaluator.aucScore})
                print(model.summary())
            else:
                print("Error: Model not Found")
                break

            y_pred = model.predict([fold["X1_test"], fold["X2_test"]])
            label_pred = evaluator.pred2label(y_pred)

            fpr, tpr, thresholds = roc_curve(fold["y_test"], y_pred)
            auc_ = roc_auc_score(fold["y_test"], y_pred)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            evaluations["AUC"].append(auc_)

            i += 1

        mean_auc = np.mean(evaluations["AUC"])
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_auc = np.std(tprs, axis=0)
        plt.plot(mean_fpr, mean_tpr, lw=2, label='{} (AUROC={:.4f})'.format(name, mean_auc), color=colorname)
        plt.axis('square')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'style': 'italic',
                 'size': 15,
                 }
        plt.xlabel('$1-Specificity(1-S_p)$', font2)
        plt.ylabel('$Sensitivity(S_n)$', font2)
        plt.legend(loc='lower right', fontsize=15)
plt.show()
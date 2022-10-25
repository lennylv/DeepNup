import os
import pickle
import math
import numpy as np
from matplotlib import pyplot as plt
import evaluator
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, confusion_matrix
from sklearn.metrics import roc_auc_score, f1_score
import argparse
from tensorflow.keras.models import load_model
import pdb

parser = argparse.ArgumentParser(description='Nucleosome Classification Experiment')
parser.add_argument('-pn', '--plot', dest='plotName', type=str, default="0",
                    help='Plot Title')
parser.add_argument('-p','--path', dest='path', type=str, default=r"D:\DeepNup\experiment",
                    help='Model file Path')
parser.add_argument('-e', '--experiments', dest='exp', default=r'Experiment_DLNN_C',
                    help='Experiments Name')
parser.add_argument('-f', '-foldName', dest='foldName', default="folds.pickle",
                    help='Folds Filename')

args = parser.parse_args()
inPath = args.path
expName = args.exp
foldName = args.foldName
plotName = args.plotName

del args

i = 1

m = "DLNN"

fig = plt.figure(i, figsize=(12, 10))
ax = fig.add_subplot(111)

accMean = []
accStd = []
mccMean = []
mccStd = []
sensMean = []
sensStd = []
specMean = []
specStd = []
AUCMean = []
AUCStd = []
F1_ScoreMean = []
F1_ScoreStd = []

evaluations = {
        "Accuracy": [],
        "Precision": [],
        "TPR": [],
        "FPR": [],
        "AUC": [],
        "Sensitivity": [],
        "Specificity": [],
        "MCC": [],
        "F1_Score": []
    }
modelPath = os.path.join(inPath, expName,  "models", m)
foldPath = os.path.join(inPath, expName)

if(not os.path.exists(os.path.join(foldPath, foldName))):
        print("Error: Folds not Found in {}".format(os.path.join(foldPath, foldName)))
else:
    with open(os.path.join(foldPath, foldName), "rb") as fp:
        folds = pickle.load(fp)
    i = 1
    for fold in folds:
        #Check if model alredy exists
        if(os.path.exists(os.path.join(modelPath, "{}_bestModel-fold{}.hdf5".format(m, i)))):

            # load json and create model
            model = load_model(os.path.join(modelPath, "{}_bestModel-fold{}.hdf5".format(m, i)),
                            custom_objects={"precision": evaluator.precision, "recall": evaluator.recall, "f1score": evaluator.f1score, "aucScore": evaluator.aucScore})
            print(model.summary())
        else:
            print("Error: Model not Found")
            break

        # y_pred = model.predict([fold["X1_test"], fold["X2_test"]])
        y_pred = model.predict(fold["X1_test"])
        label_pred = evaluator.pred2label(y_pred)
        # Compute precision, recall, sensitivity, specifity, mcc
        acc = accuracy_score(fold["y_test"], label_pred)
        prec = precision_score(fold["y_test"], label_pred)

        conf = confusion_matrix(fold["y_test"], label_pred)
        if(conf[0][0]+conf[1][0]):
            sens = float(conf[0][0])/float(conf[0][0]+conf[1][0])
        else:
            sens = 0.0
        if(conf[1][1]+conf[0][1]):
            spec = float(conf[1][1])/float(conf[1][1]+conf[0][1])
        else:
            spec = 0.0
        if((conf[0][0]+conf[0][1])*(conf[0][0]+conf[1][0])*(conf[1][1]+conf[0][1])*(conf[1][1]+conf[1][0])):
            mcc = (float(conf[0][0])*float(conf[1][1]) - float(conf[1][0])*float(conf[0][1]))/math.sqrt((conf[0][0]+conf[0][1])*(conf[0][0]+conf[1][0])*(conf[1][1]+conf[0][1])*(conf[1][1]+conf[1][0]))
        else:
            mcc = 0.0
        fpr, tpr, thresholds = roc_curve(fold["y_test"], y_pred)
        auc = roc_auc_score(fold["y_test"], y_pred)

        f1score = f1_score(fold['y_test'], label_pred)

        evaluations["Accuracy"].append(acc)
        evaluations["Precision"].append(prec)
        evaluations["TPR"].append(tpr)
        evaluations["FPR"].append(fpr)
        evaluations["AUC"].append(auc)
        evaluations["Sensitivity"].append(sens)
        evaluations["Specificity"].append(spec)
        evaluations["MCC"].append(mcc)
        evaluations["F1_Score"].append(f1score)

        i = i + 1

    #pdb.set_trace()
    accMean.append(np.mean(evaluations["Accuracy"]))
    accStd.append(np.std(evaluations["Accuracy"]))
    mccMean.append(np.mean(evaluations["MCC"]))
    mccStd.append(np.std(evaluations["MCC"]))
    sensMean.append(np.mean(evaluations["Sensitivity"]))
    sensStd.append(np.std(evaluations["Sensitivity"]))
    specMean.append(np.mean(evaluations["Specificity"]))
    specStd.append(np.std(evaluations["Specificity"]))
    AUCMean.append(np.mean(evaluations["AUC"]))
    AUCStd.append(np.std(evaluations["AUC"]))
    F1_ScoreMean.append(np.mean(evaluations["F1_Score"]))
    F1_ScoreStd.append(np.std(evaluations["F1_Score"]))


metricsMean = [accMean[0], sensMean[0], specMean[0], mccMean[0], AUCMean[0], F1_ScoreMean[0]]
metricsStd = [accStd[0], sensStd[0], specStd[0], mccStd[0], AUCStd[0], F1_ScoreStd[0]]
print(accStd)
print(mccStd)
print(sensStd)
print(specStd)

x = range(len(metricsMean))
labels = ["Accuracy", "Sensitivity", "Specificity", "MCC", "AUC", "F1_Score"]
plt.bar(x, metricsMean, width=0.8, bottom=None)
plt.title(plotName)
ax.set_xticks(x)
ax.set_xticklabels(labels)
plt.ylim(0, 1.10)
for i, v in enumerate(metricsMean):
    ax.text(i-.40, v+.05, "{}+-{}".format(round(v,4),round(metricsStd[i],4)), color='blue', fontweight='bold')

plot_dir = os.path.join(modelPath, "plot")
if(not os.path.isdir(plot_dir)):
    os.makedirs(plot_dir)
fig.savefig(os.path.join(plot_dir, "{} Evaluations.png".format(plotName)), bbox_inches='tight')
fig.savefig(os.path.join(plot_dir, "{} Evaluations.svg".format(plotName)), format="svg", bbox_inches='tight')
fig.savefig(os.path.join(plot_dir, "{} Evaluations.eps".format(plotName)), format="eps", bbox_inches='tight')




import pandas as pd
import os
# precision-recall curve and f1
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from matplotlib import pyplot as plt

pth = '../output/2_mlb_loss_count_ones_low_lr'
pred = pd.read_csv(os.path.join(pth, 'pred_raw.csv'))
cols = pred.columns[1:]
pred = pred[pred.columns[1:]].values
test = pd.read_csv(os.path.join(pth, 'test.csv'))
test = test[test.columns[1:]].values
pred_hat = pred.round()
# calculate precision-recall curve
for i in range(test.shape[1]):
    t_i = test[:, i]
    p_i = pred[:, i]
    p_hat_i = pred_hat[:, i]
    precision, recall, thresholds = precision_recall_curve(t_i, p_i)

    # calculate F1 score
    f1 = f1_score(t_i, p_hat_i)

    # calculate precision-recall AUC
    auc_m = auc(recall, precision)

    # calculate average precision score
    ap = average_precision_score(t_i, p_i)

    print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc_m, ap))
    # plot no skill
    plt.plot([0, 1], [0.5, 0.5], linestyle='--')
    # plot the precision-recall curve for the model
    plt.plot(recall, precision, marker='.')
    plt.ylabel('precision')
    plt.xlabel('recall')
    plt.title(cols[i]+'- PR Curve')
    plt.savefig('curves/'+cols[i]+'- PR Curve')
    # show the plot
    plt.show()


    # calculate AUC
    auc_m = roc_auc_score(t_i, p_i)
    print('AUC: %.3f' % auc_m)
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(t_i, p_i)
    # plot no skill
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the precision-recall curve for the model
    plt.plot(fpr, tpr, marker='.')
    # show the plot
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title(cols[i]+'- ROC Curve')
    plt.savefig('curves/'+cols[i]+'- ROC Curve')
    plt.show()
from glob import glob
from sklearn.metrics import multilabel_confusion_matrix
import pandas as pd
import pickle
import os
tests = sorted(glob("../output/*/test.csv"))
preds = sorted(glob("../output/*/pred.csv"))

for test, pred in zip(tests, preds):
    fld = test.split('/')[:-1]
    t = pd.read_csv(test)
    p = pd.read_csv(pred)
    cols = t.columns[1:]
    cm = multilabel_confusion_matrix(t[cols].values.astype('float'),  p[cols].values.astype('float'))
    with open(os.path.join('/'.join(fld), "c_matrix.pkl"), 'wb') as f:
        pickle.dump(cm, f)

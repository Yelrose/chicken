import numpy as np
from sklearn.metrics import roc_auc_score



f = open("./train/truth_train.csv")
lines = f.readlines()
ans_t = [float(a.split(",")[1]) for a in lines]
ans_t = np.array(ans_t,dtype="float32")
f = open("./result.txt")
lines = f.readlines()
ans = [float(a.split(",")[1]) for a in lines]
ans = np.array(ans,dtype="float32")
print roc_auc_score(ans_t,ans)

import numpy as np
import pandas as pd
import os
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

#read data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
# real_test = pd.read_csv("real_test.csv")
count_data = pd.read_csv("count_data1.csv")

train = count_data.iloc[:200000]
# train = count_data["target"].notnull()
print(train)
target = train["target"]

real_test = count_data.iloc[200000:].reset_index(drop = True)
# real_test = count_data["target"].isnull()
# print(real_test)

#训练标签
# target = pd.Series(list(train["target"])*200)
#将所有的转为两列，一列是原数据，一列是count encoding
def trans(df,sign):
    var_s = pd.Series()
    var_size_s = pd.Series()
    for c in range(200):
        print(sign+" num.{}".format(c))
        var_s = pd.concat([df["var_"+str(c)],var_s],axis = 0,ignore_index = True)
        var_size_s = pd.concat([df["var_"+str(c)+"_size"],var_size_s],axis = 0,ignore_index = True)
    df = pd.DataFrame({"var":var_s,"var_size":var_size_s})
    print(df)
    return df



real_test_id = real_test["ID_code"]

params = {
    "objective" : "binary",
    "metric" : "auc",
    "boosting": 'gbdt',
    "max_depth" : 3,
    "num_leaves" : 13,
    "learning_rate" : 0.01,
    "bagging_freq": 5,
    "bagging_fraction" : 0.4,
    "feature_fraction" : 1,
    "min_data_in_leaf": 80,
    # "min_sum_heassian_in_leaf": 10,
    #"lambda_l1" : 5,
    #"lambda_l2" : 5,
    "verbosity" : -1
}

folds = StratifiedKFold(n_splits = 5,shuffle =True,random_state = 77777)
oof = np.zeros((len(train),200))
predictions = np.zeros(len(real_test))
val_auc = []

for fold,(trn_idx,val_idx) in enumerate(folds.split(train.values,target.values)):
    print(">>>>>>>>>>>>>>>>>>>>>>>Fold  {}".format(fold))
    for c in range(200):
        print(">>>>>>>>>>>>>>>>>>>>var_{}".format(c))
        feature_chioce = ["var_"+str(c),"var_"+str(c)+"_size"]
        trn_data = lgb.Dataset(train.iloc[trn_idx][feature_chioce],label = target[trn_idx])
        val_data = lgb.Dataset(train.iloc[val_idx][feature_chioce],label = target[val_idx])
        clf = lgb.train(params,trn_data,100000,valid_sets = [trn_data,val_data],verbose_eval = 400,early_stopping_rounds = 1000)
        oof[val_idx,c:c+1] = clf.predict(train.iloc[val_idx][feature_chioce],num_iteration = clf.best_iteration).reshape((len(val_idx),1))
        val_auc.append(roc_auc_score(target[val_idx],oof[val_idx,c:c+1]))
        print(val_auc[-1])
        predictions += clf.predict(real_test[feature_chioce],num_iteration = clf.best_iteration)
    predictions = (predictions/200)
predictions = predictions/5

mean_auc = np.mean(val_auc)
std_auc = np.std(val_auc)
all_auc = roc_auc_score(target,oof.sum(axis = 1)/200)
print("Mean auc:%.9f,   std:%.9f  All auc:%.9f"  %(mean_auc,std_auc,all_auc))
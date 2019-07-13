import numpy as np
import pandas as pd
import os
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#read data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
# real_test = pd.read_csv("real_test.csv")
count_data = pd.read_csv("count_dat1.csv")

train = count_data.iloc[:200000]
# print(train)
real_test = count_data.iloc[200000:].reset_index(drop = True)
# print(real_test)

# real_test_id = real_test.ID_code
real_test_id = real_test["ID_code"]
features = [c for c in train.columns if c not in ["ID_code","target"]]
# print(features)

target = train["target"]
train = train[features]
real_test = real_test[features]
# print(train)
# print(target)
# print(real_test)

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
oof = np.zeros(len(train))
predictions = np.zeros(len(real_test))
val_auc = []
feature_impotance = pd.DataFrame()

for fold,(trn_idx,val_idx) in enumerate(folds.split(train.values,target.values)):
	print("Fold  {}".format(fold))
	trn_data = lgb.Dataset(train.iloc[trn_idx],label = target[trn_idx])
	val_data = lgb.Dataset(train.iloc[val_idx],label = target[val_idx])
	clf = lgb.train(params,trn_data,100000,valid_sets = [trn_data,val_data],verbose_eval = 500,early_stopping_rounds = 3000)
	oof[val_idx] = clf.predict(train.iloc[val_idx],num_iteration = clf.best_iteration)
	val_auc.append(roc_auc_score(target[val_idx],oof[val_idx]))
	predictions += clf.predict(real_test,num_iteration = clf.best_iteration) / folds.n_splits

	fold_impotance = pd.DataFrame()
	fold_impotance["features"] = features
	fold_impotance["fold"] = fold + 1
	feature_impotance = pd.concat([feature_impotance,fold_impotance],axis = 0)

mean_auc = np.mean(val_auc)
std_auc = np.std(val_auc)
all_auc = roc_auc_score(target,oof)
print("Mean auc:%.9f,   std:%.9f  All auc:%.9f"  %(mean_auc,std_auc,all_auc))

"""
Fold  0
Training until validation scores don't improve for 3000 rounds.
[500]	training's auc: 0.843873	valid_1's auc: 0.82836
[1000]	training's auc: 0.879803	valid_1's auc: 0.860458
[1500]	training's auc: 0.897009	valid_1's auc: 0.875284
[2000]	training's auc: 0.908024	valid_1's auc: 0.884201
[2500]	training's auc: 0.915529	valid_1's auc: 0.889598
[3000]	training's auc: 0.921476	valid_1's auc: 0.894118
[3500]	training's auc: 0.925673	valid_1's auc: 0.897284
[4000]	training's auc: 0.929343	valid_1's auc: 0.899621
[4500]	training's auc: 0.932418	valid_1's auc: 0.901402
[5000]	training's auc: 0.935122	valid_1's auc: 0.903136
[5500]	training's auc: 0.937445	valid_1's auc: 0.904513
[6000]	training's auc: 0.93944	valid_1's auc: 0.905678
[6500]	training's auc: 0.941329	valid_1's auc: 0.906387
[7000]	training's auc: 0.943051	valid_1's auc: 0.906946
[7500]	training's auc: 0.944645	valid_1's auc: 0.907299
[8000]	training's auc: 0.946225	valid_1's auc: 0.907476
[8500]	training's auc: 0.947679	valid_1's auc: 0.907665
[9000]	training's auc: 0.949067	valid_1's auc: 0.907828
[9500]	training's auc: 0.95041	valid_1's auc: 0.907815
[10000]	training's auc: 0.951764	valid_1's auc: 0.907946
[10500]	training's auc: 0.953092	valid_1's auc: 0.907901
[11000]	training's auc: 0.954436	valid_1's auc: 0.908052
[11500]	training's auc: 0.955723	valid_1's auc: 0.907949
[12000]	training's auc: 0.956936	valid_1's auc: 0.907886
[12500]	training's auc: 0.958175	valid_1's auc: 0.907939
[13000]	training's auc: 0.95931	valid_1's auc: 0.907773
[13500]	training's auc: 0.960478	valid_1's auc: 0.907622
Early stopping, best iteration is:
[10961]	training's auc: 0.954319	valid_1's auc: 0.908079
Fold  1
Training until validation scores don't improve for 3000 rounds.
[500]	training's auc: 0.843052	valid_1's auc: 0.819596
[1000]	training's auc: 0.878976	valid_1's auc: 0.854628
[1500]	training's auc: 0.897025	valid_1's auc: 0.871113
[2000]	training's auc: 0.9079	valid_1's auc: 0.880578
[2500]	training's auc: 0.915117	valid_1's auc: 0.887065
[3000]	training's auc: 0.921238	valid_1's auc: 0.891675
[3500]	training's auc: 0.925675	valid_1's auc: 0.894902
[4000]	training's auc: 0.929486	valid_1's auc: 0.897573
[4500]	training's auc: 0.932613	valid_1's auc: 0.899482
[5000]	training's auc: 0.935238	valid_1's auc: 0.901115
[5500]	training's auc: 0.937564	valid_1's auc: 0.902487
[6000]	training's auc: 0.939618	valid_1's auc: 0.90331
[6500]	training's auc: 0.941404	valid_1's auc: 0.904137
[7000]	training's auc: 0.94308	valid_1's auc: 0.904779
[7500]	training's auc: 0.944666	valid_1's auc: 0.905271
[8000]	training's auc: 0.946225	valid_1's auc: 0.905654
[8500]	training's auc: 0.947761	valid_1's auc: 0.905931
[9000]	training's auc: 0.949225	valid_1's auc: 0.906011
[9500]	training's auc: 0.950676	valid_1's auc: 0.906253
[10000]	training's auc: 0.95206	valid_1's auc: 0.906379
[10500]	training's auc: 0.953388	valid_1's auc: 0.90645
[11000]	training's auc: 0.954649	valid_1's auc: 0.90652
[11500]	training's auc: 0.955857	valid_1's auc: 0.906547
[12000]	training's auc: 0.957111	valid_1's auc: 0.906572
[12500]	training's auc: 0.958376	valid_1's auc: 0.906514
[13000]	training's auc: 0.959556	valid_1's auc: 0.90635
[13500]	training's auc: 0.960732	valid_1's auc: 0.906301
[14000]	training's auc: 0.961914	valid_1's auc: 0.906312
[14500]	training's auc: 0.962975	valid_1's auc: 0.906205
Early stopping, best iteration is:
[11550]	training's auc: 0.956004	valid_1's auc: 0.906615
Fold  2
Training until validation scores don't improve for 3000 rounds.
[500]	training's auc: 0.842708	valid_1's auc: 0.825345
[1000]	training's auc: 0.878917	valid_1's auc: 0.859264
[1500]	training's auc: 0.89675	valid_1's auc: 0.875412
[2000]	training's auc: 0.907686	valid_1's auc: 0.883891
[2500]	training's auc: 0.915437	valid_1's auc: 0.8904
[3000]	training's auc: 0.920931	valid_1's auc: 0.894258
[3500]	training's auc: 0.925484	valid_1's auc: 0.897729
[4000]	training's auc: 0.929298	valid_1's auc: 0.900005
[4500]	training's auc: 0.932406	valid_1's auc: 0.902175
[5000]	training's auc: 0.935131	valid_1's auc: 0.903723
[5500]	training's auc: 0.937225	valid_1's auc: 0.904709
[6000]	training's auc: 0.939381	valid_1's auc: 0.9056
[6500]	training's auc: 0.941382	valid_1's auc: 0.906332
[7000]	training's auc: 0.943045	valid_1's auc: 0.906747
[7500]	training's auc: 0.944739	valid_1's auc: 0.906967
[8000]	training's auc: 0.946253	valid_1's auc: 0.907406
[8500]	training's auc: 0.947817	valid_1's auc: 0.907718
[9000]	training's auc: 0.949192	valid_1's auc: 0.907906
[9500]	training's auc: 0.950513	valid_1's auc: 0.90816
[10000]	training's auc: 0.951838	valid_1's auc: 0.90821
[10500]	training's auc: 0.953196	valid_1's auc: 0.908203
[11000]	training's auc: 0.954478	valid_1's auc: 0.908049
[11500]	training's auc: 0.955699	valid_1's auc: 0.908217
[12000]	training's auc: 0.956926	valid_1's auc: 0.908323
[12500]	training's auc: 0.958175	valid_1's auc: 0.908263
[13000]	training's auc: 0.959435	valid_1's auc: 0.908232
[13500]	training's auc: 0.960519	valid_1's auc: 0.908197
[14000]	training's auc: 0.96161	valid_1's auc: 0.908208
[14500]	training's auc: 0.962775	valid_1's auc: 0.908294
[15000]	training's auc: 0.963863	valid_1's auc: 0.908301
Early stopping, best iteration is:
[12173]	training's auc: 0.957359	valid_1's auc: 0.908366
Fold  3
Training until validation scores don't improve for 3000 rounds.
[500]	training's auc: 0.842667	valid_1's auc: 0.826287
[1000]	training's auc: 0.877693	valid_1's auc: 0.859581
[1500]	training's auc: 0.896089	valid_1's auc: 0.87571
[2000]	training's auc: 0.90745	valid_1's auc: 0.885356
[2500]	training's auc: 0.915212	valid_1's auc: 0.891317
[3000]	training's auc: 0.920845	valid_1's auc: 0.895329
[3500]	training's auc: 0.925265	valid_1's auc: 0.898502
[4000]	training's auc: 0.929045	valid_1's auc: 0.900843
[4500]	training's auc: 0.932252	valid_1's auc: 0.902566
[5000]	training's auc: 0.934938	valid_1's auc: 0.903879
[5500]	training's auc: 0.937265	valid_1's auc: 0.904849
[6000]	training's auc: 0.939258	valid_1's auc: 0.905625
[6500]	training's auc: 0.94121	valid_1's auc: 0.906364
[7000]	training's auc: 0.943091	valid_1's auc: 0.907038
[7500]	training's auc: 0.944642	valid_1's auc: 0.907325
[8000]	training's auc: 0.946214	valid_1's auc: 0.90747
[8500]	training's auc: 0.947596	valid_1's auc: 0.907786
[9000]	training's auc: 0.949066	valid_1's auc: 0.907883
[9500]	training's auc: 0.950423	valid_1's auc: 0.907954
[10000]	training's auc: 0.951781	valid_1's auc: 0.907966
[10500]	training's auc: 0.953059	valid_1's auc: 0.907987
[11000]	training's auc: 0.954274	valid_1's auc: 0.907857
[11500]	training's auc: 0.955565	valid_1's auc: 0.907748
[12000]	training's auc: 0.956808	valid_1's auc: 0.907761
[12500]	training's auc: 0.958059	valid_1's auc: 0.907752
[13000]	training's auc: 0.959292	valid_1's auc: 0.907667
[13500]	training's auc: 0.960448	valid_1's auc: 0.907481
Early stopping, best iteration is:
[10660]	training's auc: 0.953453	valid_1's auc: 0.908009
Fold  4
Training until validation scores don't improve for 3000 rounds.
[500]	training's auc: 0.842479	valid_1's auc: 0.833503
[1000]	training's auc: 0.877938	valid_1's auc: 0.863047
[1500]	training's auc: 0.895213	valid_1's auc: 0.877137
[2000]	training's auc: 0.906351	valid_1's auc: 0.88636
[2500]	training's auc: 0.914724	valid_1's auc: 0.893288
[3000]	training's auc: 0.920711	valid_1's auc: 0.897547
[3500]	training's auc: 0.925173	valid_1's auc: 0.900788
[4000]	training's auc: 0.929095	valid_1's auc: 0.903018
[4500]	training's auc: 0.932188	valid_1's auc: 0.905049
[5000]	training's auc: 0.934932	valid_1's auc: 0.906321
[5500]	training's auc: 0.937268	valid_1's auc: 0.907226
[6000]	training's auc: 0.939383	valid_1's auc: 0.908095
[6500]	training's auc: 0.941364	valid_1's auc: 0.908806
[7000]	training's auc: 0.943123	valid_1's auc: 0.909301
[7500]	training's auc: 0.944594	valid_1's auc: 0.909483
[8000]	training's auc: 0.946169	valid_1's auc: 0.90972
[8500]	training's auc: 0.947657	valid_1's auc: 0.909883
[9000]	training's auc: 0.949105	valid_1's auc: 0.909934
[9500]	training's auc: 0.950469	valid_1's auc: 0.90997
[10000]	training's auc: 0.951777	valid_1's auc: 0.910044
[10500]	training's auc: 0.953128	valid_1's auc: 0.909871
[11000]	training's auc: 0.954413	valid_1's auc: 0.909939
[11500]	training's auc: 0.955595	valid_1's auc: 0.90988
[12000]	training's auc: 0.956804	valid_1's auc: 0.909715
[12500]	training's auc: 0.957936	valid_1's auc: 0.909621
Early stopping, best iteration is:
[9760]	training's auc: 0.951136	valid_1's auc: 0.910114
Mean auc:0.908236544,   std:0.001117823  All auc:0.908199318
"""
top 5%

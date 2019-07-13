import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from sklearn import model_selection, metrics, linear_model
import datetime as dt
from scipy import special

switch = True
if switch:

    # read input
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    # print(train)
    # print(test)
    all_data = train

    # magic.1
    #这段代码是使用的第29名的方案，后来来复盘是看到的，因为最先指出fake test的大佬的代码没看懂，当时直接用的他给分好的文件，
    #复盘的时候这位的代码看懂了，而且很好理解，所以在这里是用了29th大佬的real_test代码
    #训练数据的特征名称列表
    #最后找到的真的测试集只有100000个
    train_cols = [c for c in train.columns if c not in ['ID_code', 'target', 'predicted', 'size', 'index']]
    unique_vals = {}
    for c in train_cols:
        sizes = test.groupby(c)['ID_code'].size()             #以某一特征为索引，计算某一特征取值的个数
        unique_vals[c] = sizes.loc[sizes == 1].index          #记录下来某一特征中出现个数为1次的取值
    # print(unique_vals)
    unique_bool = test.copy()
    for c in train_cols:
        unique_bool[c] = test[c].isin(unique_vals[c])        #看看训练数据对应的特征每一个取值是否在只出现一次的字典里，unique_bool是一个Series，里面全是bool
    num_unique = unique_bool[train_cols].sum(axis=1)         #对每一行的bool进行计算求和，得到新的一列num_unique用于统计每一行的bool为True的个数
    print(num_unique)
    test_real = test.loc[num_unique > 0]                     #个数大于零的为真，找出test中样本至少有一个特征在train里是唯一值的为real_test
    print(test_real)
    test_real.to_csv("real_test.csv",index = False)

    
    #magic.2
    #将训练集和真是测试集按行方向合并，因为真实测试集没有target，所以为NaN
    count_data = pd.concat([train, test_real], axis=0, sort=True)
    # print(count_data)
    for i in range(200):
        print("this is Num."+str(i)+" feature")
        c = 'var_' + str(i)  
        count_data[c + 'size'] = count_data.groupby(c)['ID_code'].transform('size')     #统计特征值某个数值在该特征中出现的次数
    print(count_data)
    count_data.to_csv("count_data1.csv",index = False)

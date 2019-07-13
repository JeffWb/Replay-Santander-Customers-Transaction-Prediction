import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from sklearn import model_selection, metrics, linear_model
import datetime as dt
from scipy import special

# Since the following forum topic and script show that relationships between columns don't matter, treat all columns
# individually and then build a meta model to combine the predictions.
# https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/83882
# https://www.kaggle.com/brandenkmurray/randomly-shuffled-data-also-works?scriptVersionId=11467087

# Values which repeat in a column seem to show stronger signal, so include a feature based on that. However, as
# observed in the following kernel, some of the test data seems to be simulated:
# https://www.kaggle.com/yag320/list-of-fake-samples-and-public-private-lb-split?scriptVersionId=11948999
# Therefore exclude these from the counts.
switch = True
if switch:
    test_mode = True

    # read input
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    # print(train)
    # print(test)
    if test_mode:
        all_data = train
    else:
        all_data = pd.concat([train, test], ignore_index=True, sort=True).reset_index()
        train_bool = all_data['target'].notnull()

    # identify the fake test data as in above kernel
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

    # # add the counts
    # #将训练集和真是测试集按行方向合并，因为真实测试集没有target，所以为NaN
    # count_data = pd.concat([train, test_real], axis=0, sort=True)
    # # print(count_data)
    # for i in range(200):
    #     print("this is Num."+str(i)+" feature")
    #     c = 'var_' + str(i)
    #     # all_size = count_data.groupby(c)['ID_code'].size().to_frame(c + '_size')
    #     # print(all_size.sort_values(c + '_size'))                               #sort_index    sort_values
    #     # count_data['rank'] = count_data[c].rank(method='first')         #将数值转为该数在所在行或列的排名  默认axis = 0
    #     # print(count_data)                                            #新添加了一列var_0的排名
    #     # count_data['bin'] = (count_data['rank'] - 1) // 300            
    #     # print(count_data)                                        #新添加了一列bin，相当于进行离散化，bin中代表每个数在第几个区间
        
    #     #只用这个用于产生count_data1
    #     count_data[c + 'size'] = count_data.groupby(c)['ID_code'].transform('size')     #统计特征值某个数值在该特征中出现的次数
    #     # count_data['avg_size_in_bin'] = count_data.groupby('bin')[c+'size'].transform('mean')  #出现在某个区间内的个数的平均值
    #     # count_data[c + '_size_scaled'] = count_data[c+'size'] / count_data['avg_size_in_bin']  #对出现次数进行标准化
    #     # all_size[c + '_size_scaled'] = count_data.groupby(c)[c + '_size_scaled'].first()     #取分组中最大值
    #     # print(all_size)
    #     # count_data = pd.merge(count_data, all_size, 'left', left_on=c, right_index=True)
    #     # all_data = pd.merge(all_data, all_size, 'left', left_on=c, right_index=True)
    #     # print(all_data)
    #     # count_data = count_data.drop(["rank","bin"],axis = 1)
    # print(count_data)
    # count_data.to_csv("count_data2.csv",index = False)


    # add the counts
    #将训练集和真是测试集按行方向合并，因为真实测试集没有target，所以为NaN
    count_data = pd.concat([train, test_real], axis=0, sort=True)
    # print(count_data)
    for i in range(200):
        print("this is Num."+str(i)+" feature")
        c = 'var_' + str(i)
        
        #只用这个用于产生count_data1
        count_data[c + 'size'] = count_data.groupby(c)['ID_code'].transform('size')     #统计特征值某个数值在该特征中出现的次数
    print(count_data)
    count_data.to_csv("count_data2.csv",index = False)
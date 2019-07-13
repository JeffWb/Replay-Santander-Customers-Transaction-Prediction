# Replay-Santander-Customers-Transaction-Prediction
这是一篇关于kaggle比赛的复盘
Santander Customer Transaction Prediction
一 比赛前期
1.	比赛简介
预测客户是否会做出交易，标签为1表示会做出交易，为0表示不会做出交易。训练和测试数据各有200000个，包括200个匿名特征，最终的评价指标为auc得分。

2.	前期数据探索
（1）	数据清洗：判断是否有缺失值  df.isnull();df.info()。该问题中不含有缺失值
（2）	正负样本不平衡问题，正:负 ≈ 7:1，但是由于评价指标为auc，所以样本数据不平衡问题的影响较小。不需要做处理
（3）	训练数据集和测试数据集具有极为相似的分布。具体的对训练集合测试集每一个特征进行sns.distplot，观察其分布，发现每一个特征在训练集合测试集上的分布非常相似。

3.	特征工程
	因为特征都是匿名，所以前期的特征工程没有做，只是简单地对连续值进行了离散化，主要因为离散化的数据鲁棒性更强一些。具体的train[“var”] = pd.cut(train[“var”], bins, labels = [])

4.	模型选择
先从简单的模型开始，一次尝试了LR，Decision Tree，Random Forest，Lightgbm
（1）	LR：使用GridSearchCV搜索最优参数C，最后取模型的预测概率，最终得分0.860
（2） Decision Tree：依旧使用GridSearchCV进行调参，主要参数有特征的划分标准（criterion），树的深度（max_depth），最小划分样本数（min_samples_split）划分时考虑的最大特征数（max_features）,最终得分较低0.766
（3） Random Forest：使用GridSearchCV进行调参，主要参数有决策树个数（n_estimators），特征划分标准(criterion),树的深度（max_depth），划分时考虑的最大特征数（max_features），最小划分样本数（min_samples_split）最终得分为0.854
（4） lightgbm：这里没有选用xgboost，主要是数据较大，lightgbm的运行速度更快。调参方式为手动调节，先将learning_rate设置的大一些，以加快训练，每次固定其他参数，对某一参数进行调节，主要调节参数有：max_depth,num_leaves,learning_rate,bagging_freq,bagging_fraction,feature_fraction,最终得		分为0.900~0.901

二 比赛中期
	此时比赛进入了一个瓶颈期，尝试各种各样的方法，比如数据标准化，PCA特征，选择构造新特征（每一个样本的所有特征值的均值，方差，标准差等等），使用NN，都没有办法突破0.901的分数，几乎所有人都卡在0.901，只有少数人突破了0.901，并且他们将突破0.901的方法称为magic，于是开始广泛的浏览kernel和discussion，寻找magic。 

三 比赛后期
Magic.1 （想法来自kernel）所有特征相互独立

Magic.2（来自于某个kernel）test中存在合成数据，需要去除


四 总结
1.	这个比赛和其他比赛不同，全部为匿名特征，而且测试集中含有合成数据，大部分特征工程也没有很好的提高，比如离散化，标准化，特征选择，够早的常见的特征等。
2.	虽然最后取得了不错的分数，但是仍然有提高的空间，比如使用模型融合stack。

# E-commerce-ressys


我们仅使用XGBoost模型，实现了2%的排名（101/5239），

# 特征工程
我们构建的每个特征根据业务具有可解释性。非线性的多项式通过SHAP值筛选出哪些特征做多项式计算后能为模型做贡献。

特征工程分为人工经验驱动 和 数据驱动，作者认为2应该2条腿走路

* 用户交互特征
某个用户在整个数据集中交互行为数量总和
```python 
temp = user_log.groupby('user_id').size().reset_index().rename(columns={0:'u1'})   
matrix = matrix.merge(temp, on='user_id', how='left') 
```

* 时间间隔特征（模型YouTube推荐系统的Example age） 按照小时计算
```python
temp = user_log.groupby('user_id')['time_stamp'].agg([('F_time','min'), ('L_time','max')]).reset_index()  
temp['u6'] = (temp['F_time'] - temp['L_time']).dt.seconds / 3600 #求出时间间隔dt.seconds  

matrix = matrix.merge(temp[['user_id','u6']], on='user_id', how='left') 
```

* 统计用户操作类型为0，1，2，3的个数  统计每个字段中value 的个数：.unstack()
```python
temp = user_log.groupby('user_id')['action_type'].value_counts().unstack().reset_index().rename(columns={0:'u7', 1:'u8', 2:'u9', 3:'u10'})  
matrix = matrix.merge(temp, on='user_id', how='left')  
```

* 商家交互特征
```python
groups = user_log.groupby('merchant_id')  
temp = groups.size().reset_index().rename(columns={0:'m1'})  
matrix = matrix.merge(temp, on='merchant_id', how='left')  

# 商家与用户的交互情况
temp = groups['user_id'].agg([('m2','nunique')]).reset_index()  
matrix = matrix.merge(temp, on='merchant_id', how='left')  
  
# 商家与商品的交互情况
temp = groups['item_id'].agg([('m3','nunique')]).reset_index()  
matrix = matrix.merge(temp, on='merchant_id', how='left')  
  
# 商家与商品类别的交互情况
temp = groups['cat_id'].agg([('m4', 'nunique')]).reset_index()  
matrix = matrix.merge(temp, on='merchant_id', how='left')  

# 商家与品牌的交互情况
temp = groups['brand_id'].agg([('m5', 'nunique')]).reset_index()  
matrix = matrix.merge(temp, on='merchant_id', how='left') 
```

* 统计商家的点击、浏览、加购等行为【merchantID-action】
```python
# 商家的行为统计
temp = groups['action_type'].value_counts().unstack().reset_index().rename(columns={0:'m6',1:'m7',2:'m8',3:'m9'})  

matrix = matrix.merge(temp, on='merchant_id', how='left') 
```

* 按照merchant_id统计随机负采样的个数
```python
# 找出不相关的特征   求出 满足（label==-1 & merchant_id）的总和  
temp = train_data[train_data['label'] == -1].groupby(['merchant_id']).size().reset_index().rename(columns={0:'m10'})  
matrix = matrix.merge(temp, on='merchant_id', how='left') 
```

* 商家与用户的交互行为
```python
groups = user_log.groupby(['user_id', 'merchant_id'])  
temp = groups.size().reset_index().rename(columns={0:'um1'})  
matrix = matrix.merge(temp, on=['user_id','merchant_id'], how='left') 
```

* 唯一ID交互
```python
# 同一品牌、同一商品、同一品类的交互情况
temp = groups[['item_id','brand_id','cat_id']].nunique().reset_index().rename(columns={'item_id':'um2','brand_id':'um3','cat_id':'um4'})  

matrix = matrix.merge(temp, on=['user_id','merchant_id'], how='left')  
```

* 时间间隔特征
```python
temp = groups['time_stamp'].agg([('F_time','min'), ('L_time','max')]).reset_index() # 一定要使用 reset_index()  

# 第一次购买和最后一次购买的时间差
temp['um6'] = (temp['F_time'] - temp['L_time']).dt.seconds / 3600  
temp.drop(['F_time','L_time'], axis=1, inplace=True)  
matrix = matrix.merge(temp, on=['user_id','merchant_id'], how='left')  


temp = groups['action_type'].value_counts().unstack().reset_index().rename(columns={0:'um7',1:'um8',2:'um9',3:'um10'})  
matrix = matrix.merge(temp, on=['user_id','merchant_id'],how='left') 
```

* 类别特征：比率特征
CTR = 点击量 / 展示量 = click / show_content
CVR = 购买量 / 点击量 = purchase / click

```python 
# 用户购买点击比（CVR）  
matrix['r1'] = matrix.u9 / matrix.u7  
matrix['r11'] = matrix.u10 / matrix.u7  
# 商家购买点击比  
matrix['r2'] = matrix.m8 / matrix.m6  
matrix['r21'] = matrix.m9 / matrix.m6  
# 不同用户不同商家的购买点击比  
matrix['r3'] = matrix.um9 / matrix.um7  
matrix['r31'] = matrix.um10 / matrix.um7
```

* SHAP可得到可以交叉的特征名称
使用SHAP值反推特征交叉，这个步骤是特征工程的第二部分，在baseLine的基础上，使用XGBoost建模。再使用SHAP值计算`shap.TreeModel.interaction()` 得到哪些特征交叉会有明显的提升效果。


* Word2Vec得到商品id embedding + MaxPooling
向量转为标量。参考Airbnb，Word2Vec提取到的item embedding后，取一个向量中的最大值 或 一个item向量中所有数字的平均值作为该item embedding的标量
```python 
import gensim  
# merchant2vec  
model_merchant = gensim.models.Word2Vec(
        user_log_path['merchant_path'].apply(lambda x: x.split(' ')),  
        window=8, sg=0,  
        size=32,  
        iter=10,  
        min_count=0, # 所有商品都需要建立embedding，所以min_count=0  
        hs=0, #使用negative sampling  
        sample=1e-4,# 热门商品降维：高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)  
        negative=100, #  正样本: 负样本= 1：5 or 1:10  windows=5, 负样本为25-50个  
        alpha=0.03, # 学习率  
        min_alpha=0.0007,  
        seed=14, workers=-1)  
model_merchant.save('product2vec.model')

# Word2Vec得到的是向量，对向量进行Pooling操作：

def get_sum_w2v(data, columns, model):  
    data_array = []  
    for index, word in data.iterrows():  
        str_id = word[columns].split(' ')  
        # 得到用户点击过item_id的embedding  
        each_emb = [model[str(i)] for i in str_id]  
        # sum_pooling   
        data_array.append(np.sum(each_emb))  
        break  
    return pd.DataFrame(data_array)
```

* 用户
```python 
# 用户最喜欢的商家、品牌、品类id换成其对应的Embedding  
model = gensim.models.Word2Vec.load('merchant2vec.model')  
matrix['merchant_most_1_emb'] = matrix['merchant_most_1'].astype(int).apply(lambda x: np.sum(model[str(x)]))  
matrix['merchant_id_emb'] = matrix['merchant_id'].astype(int).apply(lambda x: np.sum(model[str(x)]))  

model = gensim.models.Word2Vec.load('brand2vec.model')  
matrix['brand_most_1_emb'] = matrix['brand_most_1'].astype(int).apply(lambda x: np.sum(model[str(x)]))  
	  
model = gensim.models.Word2Vec.load('cat2vec.model')  
matrix['cat_most_1_emb'] = matrix['cat_most_1'].astype(int).apply(lambda x: np.sum(model[str(x)]))

```

* 用户点击行为序列特征
```python 
temp = pd.DataFrame(user_log.groupby('user_id')['merchant_id', 'action_type'].agg(lambda x: list(x)))  
# 列名称改成hist_merchant_id 和 hist_action_type  
temp.columns = ['hist_merchant_id', 'hist_action_type']  
matrix = matrix.merge(temp, on=['user_id'], how='left')  # 统计时间间隔
```


XGBoost

DIN模型


### 未来的改进方向
* 为item_id找邻居，作为新特征喂给模型
模仿语义匹配构建特征工程的思路，使用google预训练好的word2vec语料模型，找到某一个词的同义词，并作为特征喂给模型，这样可以增加模型的泛化能力。在推荐系统中，使用训练好的word2vec模型为movie_id找到相似度id的embedding喂给模型。相当于以某一个item_id为中心，找到他周围的邻居Embedding作为特征喂给模型

* 使用 action_type 作为权重给 hist_item_id 序列加权
用户user_id对每个item_id都有相应的行为记录在 action_type 中，该action_type 也构成句子序列和 item_id 构成的序列相乘

* 数据驱动的方式构造类别特征
SHAP值可以清晰计算出连续特征那个区间范围内对模型的贡献度为正，我们可以把这个区间范围拿出来做构建0-1类别特征，比如，球星的年龄在20-31岁之间对能提升身价，我们可以把20-31岁构建为`是否黄金年龄段`特征。

* 模型融合
模型融合有多种思路，多个模型结果进行融合适合打比赛时使用

工程中的模型融合更多的是 GBDT+LR 的思路，一个模型为下一个模型提取特征。因此，对于DIN模型的改造，可以把FM融进去，用于提取类别特征

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

nodes = pd.read_csv("node_information.csv")
node_dictionary={}
node_list = nodes["id"].values
node_index = list(range(0, len(nodes["id"].values)))# index為: 0~ 
node_dictionary = dict(zip(node_list, node_index))  # using id to find index

period1 = pd.read_csv("Period1.csv")

# 對每一項做index轉換
for index in range(len(period1.values)): #0~max
    for term in range(2): #0,1
        period1.values[index][term] = node_dictionary[period1.values[index][term]]

# 把這些period1的邊加入Graph中
tuples = [tuple(x) for x in period1.values]
G = nx.Graph()
G.add_edges_from(tuples)

# Compute Jaccard Coefficients from g_train
jc_matrix = np.zeros(adj_sparse.shape)
for u, v, p in nx.jaccard_coefficient(G): # (u, v) = node indices, p = Jaccard coefficient
    jc_matrix[u][v] = p
    jc_matrix[v][u] = p # make sure it's symmetric
    
# Normalize array
jc_matrix = jc_matrix / jc_matrix.max()

#把參數存起來，之後就不用算
f = open('Jc_matrix2.txt', 'w')
for line in jc_matrix:
    f.write(str(line)+'\n')
f.close()

# Calculate, store Adamic-Index scores in array
pa_matrix = np.zeros(adj_sparse.shape)
for u, v, p in nx.preferential_attachment(G): # (u, v) = node indices, p = Jaccard coefficient
    pa_matrix[u][v] = p
    pa_matrix[v][u] = p # make sure it's symmetric

# Normalize array
pa_matrix = pa_matrix / pa_matrix.max()
#把參數存起來，之後就不用算
f = open('PA_matrix.txt', 'w')
for line in pa_matrix:
    f.write(str(line)+'\n')
f.close()


# 存好兩個feature了，提取出train set

period2 = pd.read_csv("Period2.csv")
period2 = period2.drop(columns = 'year')
#轉換index
for index in range(len(period2.values)): #0~max
    for term in range(2): #0,1
        period2.values[index][term] = node_dictionary[period2.values[index][term]]

tuples2 = [tuple(x) for x in period2.values]
G.add_edges_from(tuples2)

#把新增的edge的資訊存到data中，準備作為轉換feature後訓練資料用
for index in range(len(period2.values)): #0~max
    for term in range(2): #0,1
        data[index][term] = period2.values[index][term]

# 新增label 
data_label = []
for index in range(len(period2.values)):
    data_label.append(1)

# 接著採樣負樣本- 把G中沒有edge的數對，存起來! 數量則是跟period2 中的正相關edge數相同

data2 = np.zeros(period2.shape)
counter = 0 #確認數量
for index1 in range(adj_sparse.shape[0]):
    for index2 in range (adj_sparse.shape[1]):
        if(G.has_edge(index1,index2) == False and counter < len(period2.values)): # 確保數量不超過period2的數量
            data2[counter][0] = index1
            data2[counter][1] = index2
            counter = counter + 1
            print((index1,index2,counter))
        else:
            break
# 合成正負樣本為訓練資料
train_data = np.vstack((data,data2))
# 新增負label
for index in range(len(period2.values)):
    data_label.append(0)

# 把edge轉換成feature 
train_feature = np.zeros(train_data.shape)
for index in range(train_data.shape[0]):
    train_feature[index][0] = jc_matrix[int(train_data[index][0])][int(train_data[index][1])]
    train_feature[index][1] = pa_matrix[int(train_data[index][0])][int(train_data[index][1])]
        
#開始train吧! 使用logistic regression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(train_feature,data_label)

# predict 看看
test_dataframe = pd.read_csv("TestData.csv")
test_dataframe = test_dataframe.drop(columns = 'year')
#開始提取feature
test_feature = np.zeros(test_dataframe.shape)
outliner_counter = 0
for index in range(test_dataframe.shape[0]): #0~test data總數-1
    if(test_dataframe.values[index][0] not in node_list or test_dataframe.values[index][1] not in node_list ): 
        # 若不在原本的網路 則此feature為0
        test_feature[index][0] = 0
        test_feature[index][1] = 0
        outliner_counter = outliner_counter + 1
    else: # 否則則切換index
        test_dataframe.values[index][0] = node_dictionary[test_dataframe.values[index][0]]
        test_dataframe.values[index][1] = node_dictionary[test_dataframe.values[index][1]]
        #然後從新的index去取得feature
        test_feature[index][0] = jc_matrix[int(test_dataframe.values[index][0])][int(test_dataframe.values[index][1])]
        test_feature[index][1] = pa_matrix[int(test_dataframe.values[index][0])][int(test_dataframe.values[index][1])]

# 得到feature後，開始預測
prediction_result = clf.predict(test_feature)

output_test_dataframe = pd.read_csv("TestData.csv")
output_test_dataframe = output_test_dataframe.drop(columns= ['year','source id'])
se = pd.Series(prediction_result)
output_test_dataframe['label'] = se.values
target_list = list(range(1, 10001))
target_id_series = pd.Series(target_list)
output_test_dataframe['target id'] = target_id_series.values
output_test_dataframe.to_csv('submission.csv',index = False)
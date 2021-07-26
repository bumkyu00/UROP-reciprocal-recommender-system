#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import re

def get_data(path):
    for line in open(path):
        p = re.findall("\d+", line)
        yield [int(p[0]), int(p[1]), int(p[2])]

data_all = tf.data.Dataset.from_generator(get_data, output_signature = 
                                          (tf.TensorSpec(shape=(tf.TensorShape([3])), dtype=tf.int64)), 
                                          args = ("/home/bumkyu00/Deep-Learning/out_reduced.txt",))


# In[2]:


N = 5000             #training에 사용할 사용자 수
L = 16856            #data 개수

dataset_all = data_all.shuffle(L)

dataset_test = dataset_all.take(3000)
dataset_validate = dataset_all.take(3000)
dataset_train = dataset_all.skip(6000)


# In[3]:


F = 20               #feature vector의 차원 수
lev = 10           #각 사용자의 feature vector 크기를 비슷하게 맞추는 leveling 계수
lam = 0           #lambda: regularization 계수

tfvar_feature_from = tf.Variable(tf.random.normal([N, F], mean = 0.5), name = "feature_from") #(N x F) feature vector. 훈련시킬 대상
tfvar_feature_to = tf.Variable(tf.random.normal([N, F], mean = 0.5), name = "feature_to")
optimizer = tf.keras.optimizers.Adagrad(0.1)


# In[4]:


def cost():
    err = tf.constant(0.)
    for element in dataset_train.repeat().batch(2000).take(1):
        for row, col, rating in element.numpy():
            fv1 = tf.expand_dims(tf.abs(tfvar_feature_from)[row - 1], 0)
            fv2 = tf.expand_dims(tf.abs(tfvar_feature_to)[col - 1], 0)
            tmp = tf.square(rating - tf.squeeze(tf.matmul(fv1, fv2, transpose_b = True))) +                   lev * (tf.square(tf.norm(fv1) - 2.236068) + tf.square(tf.norm(fv2) - 2.236068))
            err = err + tmp
    return err


# In[13]:


for i in range(500):
    print(i, end=' ')
    if i % 50 == 0:
        print()
        feature_from = tfvar_feature_from.numpy()
        feature_to = tfvar_feature_to.numpy()
        e = 0
        for element in dataset_train:
            row = element.numpy()[0]
            col = element.numpy()[1]
            rating = element.numpy()[2]
            e += np.square(rating - np.dot(tf.abs(tfvar_feature_from)[row - 1], tf.abs(tfvar_feature_to)[col - 1]))
        print(e/10856)
    optimizer.minimize(loss = cost, var_list = [tfvar_feature_from, tfvar_feature_to])


# In[14]:


feature_from = tf.abs(tfvar_feature_from).numpy()
feature_to = tf.abs(tfvar_feature_to).numpy()
test_error = []
e = 0
for element in dataset_validate:
    row = element.numpy()[0]
    col = element.numpy()[1]
    rating = element.numpy()[2]
    r = np.dot(feature_from[row - 1], feature_to[col - 1])
    if r > 10:
        r = 10
    test_error.append([row, col, rating, r])
    e += np.square(rating -  r)
print(e/3000)
np.savetxt("test_error_lev.txt", test_error, fmt="%f", delimiter=' ')


# In[15]:


train_error = []
e = 0
for element in dataset_train:
    row = element.numpy()[0]
    col = element.numpy()[1]
    rating = element.numpy()[2]
    train_error.append([row, col, rating, np.dot(feature_from[row - 1], feature_to[col - 1])])
    e += np.square(rating -  np.dot(feature_from[row - 1], feature_to[col - 1]))
print(e/10856)
np.savetxt("train_error_lev.txt", train_error, fmt="%f", delimiter=' ')


# In[17]:


import heapq

recommend = list()
for i in range(N):
    recommend.append(list())
for i in range(N):
    l = list()
    for j in range(N):
        if i == j:
            l.append(-1)
            continue
        r1 = np.dot(feature_from[i], feature_to[j])
        r2 = np.dot(feature_from[j], feature_to[i])
        l.append(2 / (1 / r1 + 1 / r2))
    for j in heapq.nlargest(10, l):
        recommend[i].append(l.index(j))

count = np.zeros(N)       #추천이 된 횟수
for i in range(N):
    for j in range(10):
        count[recommend[i][j]] += 1


# In[18]:


cnt = 0
for i in range(N):
    if count[i] > 10:
        cnt += 1
print(cnt)


# In[19]:


cnt = 0
for i in feature_to:
    for j in i:
        cnt += j
print(cnt / 3 / N)


# In[ ]:





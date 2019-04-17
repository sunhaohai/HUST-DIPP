# -*- coding:utf-8 -*-
# author:zjl
import pickle
import numpy as np
from dataloader import get_fetch_20newsgroups_tfidf
from classifier.MLP import MLP

catagories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
dataroot = './datasets/20newsbydate'
X_train, Y_train, X_test, Y_test = get_fetch_20newsgroups_tfidf(catagories, dataroot)

model1 = pickle.load(open('./models/MLP1.dat', 'rb'))
model2 = pickle.load(open('./models/MLP2.dat', 'rb'))
model3 = pickle.load(open('./models/MLP3.dat', 'rb'))
model4 = pickle.load(open('./models/MLP4.dat', 'rb'))
model5 = pickle.load(open('./models/MLP5.dat', 'rb'))

l1 = model1.predict(X_test)
l1 = l1[np.newaxis, :]
l2 = model2.predict(X_test)
l2 = l2[np.newaxis, :]
l3 = model3.predict(X_test)
l3 = l3[np.newaxis, :]
l4 = model4.predict(X_test)
l4 = l4[np.newaxis, :]
l5 = model5.predict(X_test)
l5 = l5[np.newaxis, :]

t = np.concatenate((l1, l2, l3, l4, l5), axis=0)
print(t.shape)
t=t.T
print(t.shape)
print(t)
g = []
for line in t:
    g.append(np.argmax(np.bincount(line)))
g = np.array(g)
r = 0
f = 0
for x in range(len(g)):
    if g[x] == Y_test[x]:
        r += 1
    else:
        f += 1
print(r/(r+f))
#! -*-coding:utf8-*-
import numpy as np
from dataloader import get_fetch_20newsgroups_tfidf
from classifier.MLP import MLP
from optimizer.SGD import SGD

catagories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
dataroot = './datasets/20newsbydate'

X_train, Y_train, X_test, Y_test = get_fetch_20newsgroups_tfidf(None, dataroot)

layer_size = [[X_train.shape[1], 128], [128, 20]]
model = MLP(layer_size)
batch_size = 200
step = 30000
lr = 10
optimazier = SGD(lr=lr)

for i in range(step):
    index = np.random.choice(X_train.shape[0], size=batch_size)
    rand_x = X_train[index]
    rand_y = Y_train[index]
    if i%10 == 1:
        loss = model.loss(rand_x, rand_y)
        acc = model.acc
        print('Step: ', i, '|', 'Loss: ', loss, '|', 'acc: ', acc)
    model.train(rand_x, rand_y, optimazier)

print('Eval===>')
loss = model.loss(X_test, Y_test)
print('Acc: ', model.acc, 'Loss: ', loss)
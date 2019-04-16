#! -*-coding:utf8-*-
import numpy as np
from dataloader import get_fetch_20newsgroups_tfidf
from classifier.MLP import MLP
from optimizer.SGD import SGD

catagories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
dataroot = './datasets/20newsbydate'

X_train, Y_train, X_test, Y_test = get_fetch_20newsgroups_tfidf(catagories, dataroot)
model = MLP()
batch_size = 200
step = 30000
lr = 0.1
optimizer = SGD(lr=lr)

for i in range(30000):
    best_acc = 0
    index = np.random.choice(X_train.shape[0], size=batch_size)
    rand_x = X_train[index]
    rand_y = Y_train[index]
    if i % 10 == 1:
        loss = model.loss(rand_x, rand_y)
        acc = model.acc
        print('Step: ', i, '|', 'Loss: ', loss, '|', 'acc: ', acc)
    if (i+1) % 10000 == 0:
        optimizer.lr /= 10
    if i == 20000:
        for layer in model.layers:
            if str(layer) == "Dropout":
                layer.open = False

    model.train(rand_x, rand_y, optimizer)

print('Eval===>')
loss = model.loss(X_test, Y_test)
print('Acc: ', model.acc, 'Loss: ', loss)

model.test_acc = model.acc
model.test_loss = loss

model.save('./models/MLP.dat')

model2 = MLP()
model2.load('./models/MLP.dat')
print(model2.best_acc, model2.acc)
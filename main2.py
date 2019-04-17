#! -*-coding:utf8-*-
import numpy as np
import pickle
from dataloader import get_fetch_20newsgroups_tfidf
from classifier.MLP import MLP
from optimizer.SGD import SGD

catagories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
dataroot = './datasets/20newsbydate'

X_train, Y_train, X_test, Y_test = get_fetch_20newsgroups_tfidf(catagories, dataroot)
model = MLP()
batch_size = 200
step = 10000
epoch = 500


lr = 0.1
optimizer = SGD(model.parameters, lr=lr)
print(X_train.shape[0])

# for i in range(step):
#     best_acc = 0
#     index = np.random.choice(X_train.shape[0], size=batch_size)
#     rand_x = X_train[index]
#     rand_y = Y_train[index]
#     if i % 10 == 1:
#         loss = model.loss(rand_x, rand_y)
#         acc = model.acc
#         print('Step: ', i, '|', 'Loss: ', loss, '|', 'acc: ', acc)
#     if (i+1) % 3000 == 0:
#         optimizer.lr /= 20
#     # if i == 20000:
#     #     for layer in model.layers:
#     #         if str(layer) == "Dropout":
#     #             layer.open = False
#     model.train(rand_x, rand_y, optimizer)

for i in range(epoch):
    index_list = list(np.arange(X_train.shape[0]))
    while True:
        step = 0
        if len(index_list) < batch_size:
            break
        else:
            index = np.random.choice(index_list, batch_size)
            index_list = list(set(index_list)-set(index))
            rand_x = X_train[index]
            rand_y = Y_train[index]
            model.train(rand_x, rand_y, optimizer)
            step += 1
        if step % 10 == 1:
            loss = model.loss(rand_x, rand_y)
            acc = model.acc
            print('Epoch: ', i, '|', 'Loss: ', loss, '|', 'acc: ', acc)

for layer in model.layers:
    if str(layer) == "Dropout":
        layer.open = False
for layer in model.layers:
    if str(layer) == "Batchnorm":
        layer.train = False

print('Eval===>')
loss = model.loss(X_test, Y_test)
print('Acc: ', model.acc, 'Loss: ', loss)

with open('./models/MLP.dat', 'wb') as f:
    pickle.dump(model, f)
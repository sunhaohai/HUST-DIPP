#! -*-coding:utf8-*-
import numpy as np
import pickle
from dataloader import get_fetch_20newsgroups_tfidf
from classifier.MLP import MLP
from optimizer.SGD import SGD
from optimizer.Adagrad import Adagrad
from optimizer.RMSprop import RMSprop
from tensorboardX import SummaryWriter

catagories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
dataroot = './datasets/20newsbydate'

X_train, Y_train, X_test, Y_test = get_fetch_20newsgroups_tfidf(catagories, dataroot, 256)
model = MLP()
batch_size = 200
step = 20000
epoch = 500
lr = 0.1
l2_reg_lambda = 0.1
optimizer = SGD(model.parameters, lr=lr, momentum=0.9)
write = SummaryWriter(log_dir='summary/MLP-4-layer-batchnorm-dropout-FocalLoss-2')

for i in range(epoch):
    index_list = list(np.arange(X_train.shape[0]))
    while True:
        if len(index_list) < batch_size:
            break
        else:
            index = np.random.choice(index_list, batch_size)
            index_list = list(set(index_list)-set(index))
            rand_x = X_train[index]
            rand_y = Y_train[index]
            model.train(rand_x, rand_y, optimizer, l2_reg_lambda=l2_reg_lambda)
            step += 1
        if step % 10 == 1:
            loss = model.loss(rand_x, rand_y)
            write.add_scalar('Train/Loss', loss, step)
            py = model.predict(rand_x)
            acc = np.mean(rand_y==py)
            write.add_scalar('Train/Acc', acc, step)

            TestLoss = model.loss(X_test, Y_test)
            write.add_scalar('Test/Loss', TestLoss, step)
            py = model.predict(X_test)
            TestAcc = np.mean(Y_test == py)
            write.add_scalar('Test/Acc', TestAcc, step)

            print('Epoch: ', i, '|', 'Loss: ', loss, '|', 'acc: ', acc)

write.close()

for layer in model.layers:
    if str(layer) == "Dropout":
        layer.open = False
    if str(layer) == "Batchnorm":
        layer.train = False

print('Eval===>')
loss = model.loss(X_test, Y_test)
py = model.predict(X_test)
TestAcc = np.mean(Y_test == py)
print('Acc: ', TestAcc, 'Loss: ', loss)

with open('./models/MLP-4-layer-batchnorm-dropout-FocalLoss-2.dat', 'wb') as f:
    pickle.dump(model, f)

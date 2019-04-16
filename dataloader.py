#! -*-coding:utf8-*-
from __future__ import print_function

from six.moves import cPickle as pickle
import numpy as np
import os
import platform
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = load_pickle(f)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)    
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte

def get_CIFAR10_data(cifar10_dir, num_training=49000, num_validation=1000, num_test=1000, num_dev=500):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the linear classifier.
    """
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    
    # subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]
    
    # Preprocessing: reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))
    
    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis = 0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image
    
    # add bias dimension and transform into columns
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])
    
    return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev

def get_fetch_20newsgroups_tfidf(categories, data_root=None):
    """获取20分类的tfidf向量数据
    input:
      categories: 获取的类别数据，type=list
      data_root: 数据目录，None的时候下载
    output:
      x_train: 训练向量
      y_train: 训练labels
      x_test: 测试向量
      y_test: 测试labels
    """
    newsgroups_train = fetch_20newsgroups(data_home=data_root, subset='train',  categories=categories)
    newsgroups_test = fetch_20newsgroups(data_home=data_root, subset='test',  categories=categories)
    num_train = len(newsgroups_train.data)
    num_test  = len(newsgroups_test.data)

    vectorizer = TfidfVectorizer(max_features=20)

    X = vectorizer.fit_transform( newsgroups_train.data + newsgroups_test.data )
    X_train = X[0:num_train, :]
    X_test = X[num_train:num_train+num_test,:]

    Y_train = newsgroups_train.target
    Y_test = newsgroups_test.target
    return X_train.A, Y_train, X_test.A, Y_test 

if __name__ == '__main__':
    catagories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
    dataroot = './datasets/20newsbydate'
    X_train, Y_train, X_test, Y_test = get_fetch_20newsgroups_tfidf(catagories, dataroot)  
    pickle.dump((X_train, Y_train, X_test, Y_test), open('./datasets/20_4_train.pkl', 'wb'))
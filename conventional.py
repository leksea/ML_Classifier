import numpy as np
import os
import os.path
import random
import sys
sys.path.append('data')

from load_data import load_data
from scipy.io import loadmat
from sklearn.linear_model import *
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

COMPARE_TYPE = 0
NUM_FOLDS = 10
WINDOWN_WIDTH = 26
STEP_SAMPLES = 5
file_lists = {0: ['2D', '3D'], 1: ['2D', 'en'], 2: ['en', '3D']}

iters = 2000
alpha = 0.117410317596
l1_ratio = 0.00939014538199
enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=iters)
lr = LogisticRegression()
linr = LinearRegression()
svm = SVC()
lsvm = LinearSVC()
svr = SVR()
br = BayesianRidge()
classifiers = {'enet': enet, 'lr': lr, 'linr': linr, 'svr': svr,
			   'svm': svm, 'lsvm': lsvm, 'br': br}


def norm(X):
	X -= np.mean(X)
	X /= np.std(X)
	return X


def compute_weights(X_0, X_1):
	X_0 = X_0.reshape(X_0.shape[0] * X_0.shape[1], X_0.shape[2])
	X_1 = X_1.reshape(X_1.shape[0] * X_1.shape[1], X_1.shape[2])

	m0 = np.mean(X_0, axis=0).T
	m1 = np.mean(X_1, axis=0).T

	S0 = np.cov(X_0.T) * X_0.shape[0]
	S1 = np.cov(X_1.T) * X_1.shape[0]

	Sw = S0 + S1
	SwReg = Sw + np.median(np.diagonal(Sw)) * np.eye(Sw.shape[0])

	return np.linalg.pinv(SwReg).dot(m0-m1)


def LDA_reduce(X_train, y_train, X_test):
	X_0 = X_train[y_train == 0]
	X_1 = X_train[y_train == 1]
	seq_length = X_0.shape[1]
	startSample = np.arange(0, seq_length, STEP_SAMPLES)
	endSample = startSample + WINDOWN_WIDTH
	last_ind = len(endSample) - len(endSample[endSample > seq_length]) + 1
	startSample = startSample[:last_ind]
	endSample = endSample[:last_ind]
	endSample[endSample > seq_length] = seq_length

	X_train_red = np.zeros((X_train.shape[0], last_ind))
	X_test_red = np.zeros((X_test.shape[0], last_ind))
	for i in xrange(last_ind):
		if not i % (last_ind/3):
			print 'Computing weights for window', i+1, 'of', last_ind
		ind = np.arange(startSample[i],endSample[i])
		window_weights = compute_weights(X_0[:,ind,:], X_1[:,ind,:])
		X_train_red[:,i] = np.mean(X_train[:,ind,:].dot(window_weights), axis=1)
		X_test_red[:,i] = np.mean(X_test[:,ind,:].dot(window_weights), axis=1)

	return norm(X_train_red), norm(X_test_red)


def get_data():
	X_0 = loadmat('data/' + file_lists[COMPARE_TYPE][0])
	X_0 = X_0['data']
	X_1 = loadmat('data/' + file_lists[COMPARE_TYPE][1])
	X_1 = X_1['data']

	seed = random.randint(0, 4294967295)
	X = np.concatenate((X_0, X_1))
	y = np.concatenate((np.zeros(X_0.shape[0]), np.ones(X_1.shape[0])))
	np.random.seed(seed)
	np.random.shuffle(X)
	np.random.seed(seed)
	np.random.shuffle(y)
	return X, y


def pseudo_k_fold(X, y, classifier):
	print 'First reducing data...'
	skf = StratifiedKFold(y, n_folds=NUM_FOLDS)
	X_red_list = []
	y_red_list = []
	fold = 0
	for train_index, test_index in skf:
		fold += 1
		print '\nFold', fold
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		# reduce dimensionality using LDA
		print 'Reducing dimensions...'
		_, X_test_red = LDA_reduce(X_train, y_train, X_test)
		X_red_list.append(X_test_red)
		y_red_list.append(y_test)

	seed = random.randint(0, 4294967295)
	X_red = np.concatenate(tuple(X_red_list))
	y_red = np.concatenate(tuple(y_red_list))
	np.random.seed(seed)
	np.random.shuffle(X_red)
	np.random.seed(seed)
	np.random.shuffle(y_red)

	accuracies = []
	fold = 0
	skf = StratifiedKFold(y_red, n_folds=NUM_FOLDS)
	for train_index, test_index in skf:
		fold += 1
		print '\nFold', fold
		X_train, X_test = X_red[train_index], X_red[test_index]
		y_train, y_test = y_red[train_index], y_red[test_index]
		# train using specified classifier
		predictions = classifier.fit(X_train, y_train).predict(X_test)
		# print predictions
		predictions[predictions > 0.5] = 1
		predictions[predictions < 1] = 0
		print 'Computing accuracy...'
		accuracies.append(accuracy_score(y_test, predictions))
		print 'Accuracy for fold', fold, 'is', accuracies[-1], '\n'

	return accuracies


def k_fold_analysis(X, y, classifier):
	accuracies = []
	skf = StratifiedKFold(y, n_folds=NUM_FOLDS)
	fold = 0
	for train_index, test_index in skf:
		fold += 1
		print '\nFold', fold
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		# reduce dimensionality using LDA
		print 'Reducing dimensions...'
		X_train_red, X_test_red = LDA_reduce(X_train, y_train, X_test)
		# train using specified classifier
		print 'Classifying reduced data...'
		predictions = classifier.fit(X_train_red, y_train).predict(X_test_red)
		# print predictions
		predictions[predictions > 0.5] = 1
		predictions[predictions < 1] = 0
		print 'Computing accuracy...'
		accuracies.append(accuracy_score(y_test, predictions))
		print 'Accuracy for fold', fold, 'is', accuracies[-1], '\n'

	return accuracies


if __name__ == '__main__':
	classifier = classifiers[sys.argv[1]]
	print 'Performing analysis on conditions', file_lists[COMPARE_TYPE]
	X, y = get_data()
	accuracies = pseudo_k_fold(X, y, classifier)
	print 'Mean k-fold accuracy is', np.mean(accuracies)
	print 'Individual accuracies:', accuracies
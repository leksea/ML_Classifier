import numpy as np
import sys
import random
import matplotlib.pyplot as plt
from sys import stdout
from operator import itemgetter
from scipy.io import loadmat, savemat
from sklearn.linear_model import *
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.cross_validation import StratifiedKFold
from sklearn.lda import LDA
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

COMPARE_TYPE = '0'
WINDOW_WIDTH = 65
STEP_SAMPLES = 45
NUM_FOLDS = 10
NUM_COMPONENTS = 43
analyses = ['test', 'tune']

WINDOW_RANGE = [45, 50, 55, 60, 65, 70, 75, 80]
STEP_RANGE = [42, 45, 48, 51, 54, 57, 60]
COMPONENTS_RANGE = [35, 40, 45, 50, 55]

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

file_lists = {'0': ['2D_a', '3D_a'], '1': ['2D_b', '3D_b'], \
			  '2': ['2D_a', 'en_a'], '3': ['2D_b', 'en_b'], \
			  '4': ['3D_a', 'en_a'], '5': ['3D_b', 'en_b'], \
			  '6': ['2D_a', '3D_a', 'en_a'], \
			  '7': ['2D_b', '3D_b', 'en_b'], \
			  '8': ['2D_alt', '3D_alt']}


def get_labels(predictions):
	predictions[predictions > 0.5] = 1
	predictions[predictions < 1] = 0
	return predictions


def apply_pca(X, num_components):
	X_pca = np.zeros((X.shape[0], X.shape[1], num_components))
	for i in xrange(X.shape[1]):
		pca = PCA(n_components=num_components)
		X_pca[:,i,:] = pca.fit_transform(X[:,i,:])
	return X_pca


def get_window_data(X, window_width, step_samples):
	seq_length = X.shape[1]
	startSample = np.arange(0, seq_length, step_samples)
	endSample = startSample + window_width
	last_ind = len(endSample) - max(0, len(endSample[endSample > seq_length]) - 1)
	startSample = startSample[:last_ind]
	endSample = endSample[:last_ind]
	endSample[endSample > seq_length] = seq_length

	X_window = np.zeros((X.shape[0], last_ind, X.shape[-1]))
	for i in xrange(last_ind):
		X_window[:,i,:] = np.mean(X[:,startSample[i]:endSample[i],:], axis=1)
	return X_window


def load_data(compare_type):
	files = file_lists[compare_type]
	num_classes = len(files)
	data = []
	labels = []
	for i in xrange(len(files)):
		X = loadmat(files[i])
		try:
			X = X['dat']
		except KeyError:
			X = X['data']
		data.append(X)
		labels.append(np.zeros(X.shape[0]).astype('int8') + i)

	X = np.concatenate(tuple(data))
	y = np.concatenate(tuple(labels))
	seed = random.randint(0, 4294967295)
	np.random.seed(seed)
	np.random.shuffle(X)
	np.random.seed(seed)
	np.random.shuffle(y)
	return X, y


def k_fold_analysis(X, y, classifier):
	accuracies = []
	skf = StratifiedKFold(y, n_folds=NUM_FOLDS)
	X = get_window_data(X, WINDOW_WIDTH, STEP_SAMPLES)
	X = apply_pca(X, NUM_COMPONENTS).reshape(X.shape[0], X.shape[1] * NUM_COMPONENTS)
	print X.shape
	for train_index, test_index in skf:
		stdout.write('.')
		stdout.flush()

		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

		predictions = get_labels(classifier.fit(X_train,y_train).predict(X_test))
		accuracies.append(accuracy_score(y_test, predictions))

	print '\nFinal accuracy is', np.mean(accuracies)
	return accuracies


def tune_hypers(X, y, classifier):
	best_hypers = [(0, 0, 0, 0) for _ in range(5)]
	skf = StratifiedKFold(y, n_folds=NUM_FOLDS)
	num = 0.0
	num_iters = len(WINDOW_RANGE) * len(STEP_RANGE)
	for window_width in WINDOW_RANGE:
		for step_samples in STEP_RANGE:
			num += 1.0
			X_pre_pca = get_window_data(X, window_width, step_samples)
			for num_components in COMPONENTS_RANGE:
				print 'num_components =', num_components,
				X_iter = apply_pca(X_pre_pca, num_components)
				X_iter = X_iter.reshape(X_iter.shape[0], X_iter.shape[1] * X_iter.shape[2])
				accuracies = []
				for train_index, test_index in skf:
					X_train, X_test = X_iter[train_index], X_iter[test_index]
					y_train, y_test = y[train_index], y[test_index]

					predictions = get_labels(classifier.fit(X_train,y_train).predict(X_test))
					accuracies.append(accuracy_score(y_test, predictions))
				acc = np.mean(accuracies)
				if acc > best_hypers[-1][3]:
					best_hypers[-1] = (window_width, step_samples, num_components, acc)
					best_hypers = sorted(best_hypers,key=itemgetter(3),reverse=True)
			print 'Best hypers so far:', best_hypers
			print str(100*num/num_iters)+'% done'
	return best_hypers


if __name__ == '__main__':
	assert len(sys.argv) >= 3, 'Must specify as argument type of analysis and classifier'
	assert len(sys.argv) <= 4, 'Cannot specify more than 3 arguments'
	assert sys.argv[1] in analyses, 'Must specify a valid type of analysis'
	assert sys.argv[2] in classifiers, 'Must speficy a valid classifier'
	if len(sys.argv) == 4:
		assert sys.argv[3] in file_lists, 'Must input a valid compare type'
		COMPARE_TYPE = sys.argv[3]
	classifier = classifiers[sys.argv[2]]
	print '\nLoading data for conditions', file_lists[COMPARE_TYPE]
	X, y = load_data(COMPARE_TYPE)
	if sys.argv[1] == analyses[0]:
		stdout.write('\nPerforming '+str(NUM_FOLDS)+'-fold cross validation on data')
		stdout.flush()
		accuracies = k_fold_analysis(X,y,classifier)
	elif sys.argv[1] == analyses[1]:
		best_hypers = tune_hypers(X,y,classifier)
		print 'Overall best hyperparameters:\n', best_hypers















import numpy as np
import sys
import random
import matplotlib.pyplot as plt
from window_analysis import load_data, get_labels
from sys import stdout
from operator import itemgetter
from numpy.fft import rfft
from scipy.io import loadmat, savemat
from scipy.stats import pearsonr
from sklearn.linear_model import *
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.cross_validation import StratifiedKFold
from sklearn.lda import LDA
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

COMPARE_TYPE = '1'
start_increment = 10
length_increment = 10
max_window = 100
cap = 50
thresh = 0.02
num_comp_1 = 6
num_comp_2 = 120
# thresh_range = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24]
thresh_range = [0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
num_comp_1_range = [5, 6, 7, 8, 9, 10]
num_comp_2_range = [60, 80, 100, 120, 140]
num_folds = 10
is_fft = True

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
freq_response = {'y': True, 'n': False}


class feature_queue:
	def __init__(self, num_electrodes, capacity=100):
		self.capacity = capacity
		self.queue = [[] for i in xrange(num_electrodes)]

	def add(self, feature):
		electrode = feature[3]
		if not len(self.queue[electrode]):
			self.queue[electrode].append(feature)
		elif feature[4] > self.queue[electrode][-1][4] or len(self.queue[electrode]) < self.capacity:
			self.queue[electrode].append(feature)
			self.queue[electrode] = sorted(self.queue[electrode], \
									key=lambda x: x[4],reverse=True)[:self.capacity]


def form_features(X_im, y):
	print 'Forming features...'
	queue = feature_queue(X.shape[2],cap)
	for start in [start_increment * i for i in xrange(int(X.shape[1] / start_increment))]:
		for length in [length_increment * (i+1) for i in xrange(int(max_window / length_increment))]:
			for channel in xrange(X.shape[2]):
				end = min(start+length,X.shape[1])
				# X_re_ft = np.mean(X_re[:,start:end,channel],axis=1)
				# corr, _ = pearsonr(X_re_ft,y)
				# re_feature = (False, start, length, channel, abs(corr))
				# queue.add(re_feature)
				X_im_ft = np.mean(X_im[:,start:end,channel],axis=1)
				corr, _ = pearsonr(X_im_ft,y)
				im_feature = (True, start, length, channel, abs(corr))
				queue.add(im_feature)
	return queue


def intra_electrode_pca(X_im, queue, thresh, num_comp_1, verbose):
	if verbose:
		print 'Performing intra-electorde PCA...'
	pca = PCA(n_components=num_comp_1)
	good_elec = [i for i in xrange(len(queue)) if queue[i][0][4] > thresh]
	X_init = np.zeros((X_im.shape[0], len(good_elec), num_comp_1))
	for i in xrange(len(good_elec)):
		elec = good_elec[i]
		X_elec = np.zeros((X_im.shape[0], len(queue[elec])))
		for j in xrange(len(queue[elec])):
			entry = queue[elec][j]
			start = entry[1]
			end = entry[1] + entry[2]
			X_elec[:,j] = np.mean(X_im[:,start:end,entry[3]],axis=1)
		X_init[:,i,:] = pca.fit_transform(X_elec)
	if verbose:
		print 'done'
	return X_init


def inter_electrode_pca(X_init, num_comp_2, verbose):
	if verbose:
		print 'Performing inter-electrode PCA'
	X_flat = X_init.reshape(X_init.shape[0], X_init.shape[1] * X_init.shape[2])
	X_final = PCA(n_components=min(num_comp_2,X_flat.shape[1])).fit_transform(X_flat)
	if verbose:
		print 'done'
	return X_final


def k_fold_analysis(X, y, classifier, verbose=True):
	if verbose:
		print 'Performing k-fold analysis'
	accuracies = []
	skf = StratifiedKFold(y, n_folds=num_folds)
	for train_index, test_index in skf:
		if verbose:
			stdout.write('.')
			stdout.flush()

		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

		predictions = get_labels(classifier.fit(X_train,y_train).predict(X_test))
		accuracies.append(accuracy_score(y_test, predictions))
	if verbose:
		print '\nFinal accuracy is', np.mean(accuracies)
	return accuracies


def tune_hypers(X_im, y, queue, classifier):
	best_hypers = [(0, 0, 0, 0) for _ in range(5)]
	num = 0
	num_iters = len(thresh_range) * len(num_comp_1_range)
	for thresh in thresh_range:
		for num_comp_1 in num_comp_1_range:
			num += 1.0
			X_init = intra_electrode_pca(X_im, queue, thresh, num_comp_1, False)
			for num_comp_2 in num_comp_2_range:
				X_final = inter_electrode_pca(X_init, num_comp_2, False)
				acc = np.mean(k_fold_analysis(X_final, y, classifier, False))
				if acc > best_hypers[-1][3]:
					best_hypers[-1] = (thresh, num_comp_1, num_comp_2, acc)
					best_hypers = sorted(best_hypers,key=lambda x: x[3],reverse=True)
			print 'Best hypers so far:', best_hypers
			print str(100*num/num_iters)+'% done'


if __name__ == '__main__':
	path = raw_input('Please enter the path to the data (\'.\' if current directory): ')
	while not os.path.isdir(path):
		path = raw_input('Please enter a valid path for the data: ')
	analysis = raw_input("Which type of analysis would you like to run, \'Tune\' or \'Test\'?\n")
	while analysis.lower() not in ['tune', 'test']:
		analysis = raw_input('Please choose from either \'Tune\' or \'Test\': ')
	is_fft = raw_input('Do you want to run analysis in frequency domain [Y/N]?\n')[0].lower()
	while is_fft not in ['y', 'n']:
		is_fft = raw_input('Please answer either yes (Y) or no (N): ')[0].lower()
	is_fft = freq_response[is_fft]
	classifier = raw_input('Which of the following classifiers would you like to use?\n' + \
							str(classifiers.keys()) + ': ').lower()
	while classifier not in classifiers:
		classifier = raw_input('Please enter the name of one of the above classifiers: ').lower()
	classifier = classifiers[classifier]

	print 'Loading data...'
	X, y = load_data(COMPARE_TYPE)
	if is_fft:
		X = rfft(X,axis=1)
		X = np.concatenate((np.real(X),np.imag(X)),axis=1)
	queue = form_features(X,y)
	if analysis.lower() == 'tune':
		best_hypers = tune_hypers(X, y, queue.queue, classifiers['linr'])
		save_name = 'best_params_' + {True: 'fft', False: 'raw'}[is_fft]
		np.save(save_name + '_1', best_hypers[0])
		np.save(save_name + '_2', best_hypers[1])
	else:
		save_name = 'best_params_' + {True: 'fft', False: 'raw'}[is_fft]
		try:
			hypers = np.load(save_name + '_1')
		except:
			hypers = (thresh, num_comp_1, num_comp_2)
		X_init = intra_electrode_pca(X, queue.queue, hypers[0], hypers[1], True)
		X_final = inter_electrode_pca(X_init, hypers[2], True)
		classifier = classifiers['linr']
		accuracies = k_fold_analysis(X_final, y, classifier)
		print accuracies
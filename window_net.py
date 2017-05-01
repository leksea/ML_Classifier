import os
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # for profiling to sync gpu calls disable for full run
import os.path

#import scipy.io
import lasagne    # nn packages for layers nn layers + lstm
import theano
import scipy.io
import random

theano.config.allow_gc=False
theano.scan.allow_gc=False
#theano.config.profile=True

#theano.config.mode = 'FAST_COMPILE'
theano.config.mode = 'FAST_RUN'
#theano.config.mode = 'DEBUG_MODE'
#theano.config.compute_test_value = 'raise'
#theano.config.optimizer = None
#theano.config.exception_verbosity='high'

import sys
sys.path.append('data')

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from window_analysis import load_data, get_window_data, apply_pca
from load_data import generate_batches, make_onehot

import theano.tensor as T
import numpy as np
import time


import logging
# SEED = 1234
SEED = random.randint(0, 4294967295)
np.random.seed(SEED)
BATCH_SIZE = 32
WINDOW_WIDTH = 65
STEP_SAMPLES = 45
NUM_COMPONENTS = 40
N_EPOCHS =  800
VERBOSE = True
N_HID1 = 5
NON_LIN1 = None
N_HID2 = 40
COMPARE_FLAG = '8'
DELAY = 3
LEARNING_RATE = 5e-6
# REG = 0.002
REG = 0


def compute_accuracy(y_batch, y_pred):
	y_pred = np.argmax(y_pred, axis=1)
	return accuracy_score(y_batch, y_pred)

#####################################################################################
#  LOAD DATA                                                                        #
#####################################################################################
print 'Loading data...'
X, y = load_data(COMPARE_FLAG)
X = get_window_data(X, WINDOW_WIDTH, STEP_SAMPLES)
X = apply_pca(X, NUM_COMPONENTS)
X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
# split into train and dev
split = int(0.9 * X.shape[0])
X_train, y_train = X[:split], make_onehot(y[:split])
X_val, y_val = X[split:], make_onehot(y[split:])
print 'Done\n'

####################
# Setup RNN
####################

N_FEATURES = X_train.shape[2]
N_CLASSES = np.unique(y_train).size
NUM_WINDOWS = X_train.shape[1]

print 'Creating layers...'

l_in = lasagne.layers.InputLayer(shape=(BATCH_SIZE, N_FEATURES))
# l_out_dropout = lasagne.layers.DropoutLayer(l_in)
l_out_window = lasagne.layers.DenseLayer(l_out_dropout,
										 num_units=N_HID1,
										 nonlinearity=NON_LIN1, name="%s" % i)
l_out_window = lasagne.layers.DenseLayer(l_in,
										 num_units=N_HID1,
										 nonlinearity=NON_LIN1, name="%s" % i)

	input_layers.append(l_in)
	window_layers.append(l_out_window)

l_merge = lasagne.layers.ConcatLayer(window_layers)
l_dropout = lasagne.layers.DropoutLayer(l_merge)
l_out = lasagne.layers.DenseLayer(l_dropout,
								  num_units=N_CLASSES,
 								  nonlinearity=lasagne.nonlinearities.softmax)

# l_out = lasagne.layers.ReshapeLayer(l_recurrent_out,
#                                     (BATCH_SIZE, LENGTH, N_CLASSES))

print 'Done\n'

print "Total parameters: {}".format(
    sum([p.get_value().size for p in lasagne.layers.get_all_params(l_out)]))


print 'Defining inputs and functions...'

# Cost function is mean squared error
input = T.tensor3('input')
input_dict = {input_layers[i]:input[:,i,:] for i in xrange(NUM_WINDOWS)}
target_output = T.ivector('target_output')
# Cost = mean squared error, starting from delay point

l_output_train = lasagne.layers.get_output(l_out, input_dict)
l_output_test = lasagne.layers.get_output(l_out, input_dict, deterministic=True)
cost_entropy = T.mean(T.nnet.categorical_crossentropy(l_output_train, target_output))
cost_norm = T.sum([lasagne.regularization.l2(layer) for layer in lasagne.layers.get_all_layers(l_out)])
cost = cost_entropy + REG * cost_norm
# Use NAG for training
all_params = lasagne.layers.get_all_params(l_out)
updates = lasagne.updates.nesterov_momentum(cost, all_params, LEARNING_RATE)
# updates = lasagne.updates.sgd(cost, all_params, LEARNING_RATE)

# Theano functions for training, getting output, and computing cost
train = theano.function([input, target_output], cost, updates=updates)
y_pred = theano.function([input], l_output_test)
compute_cost = theano.function([input, target_output], cost)

print 'Done\n'

print 'Begin training\n'

costs = []
c = 0
for n in xrange(N_EPOCHS):
	if not n % 10:
		print 'epoch:', n
	X, y = generate_batches(X_train, y_train, BATCH_SIZE)
	y = y[:,:,0]
	for batch in xrange(X.shape[0]):
		c += 1
		if not c % 100:
			cost = [compute_cost(X[b], y[b]) for b in xrange(X.shape[0])]
			print 'cost:', np.sum(cost)
		if not c % 1000:
			accuracies = [compute_accuracy(y[b], y_pred(X[b])) for b in xrange(X.shape[0])]
			X_v, y_v = generate_batches(X_val, y_val, BATCH_SIZE)
			y_v = y_v[:,:,0]
			val_accuracies = [compute_accuracy(y_v[b], y_pred(X_v[b])) for b in xrange(X_v.shape[0])]
			print 'training accuracy:', np.mean(accuracies)
			print 'validation accuracy:', np.mean(val_accuracies)
		X_b, y_b = X[batch], y[batch]
		costs.append(train(X_b, y_b))






import numpy
import time
import sys
import subprocess
import os
import random
import matplotlib.pyplot as plt

sys.path.append('data')
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from window_analysis import load_data, get_window_data, apply_pca
from rnn.special import model

if __name__ == '__main__':

    s = {'lr':0.004,
         'verbose':1,
         'decay':False, # decay on the learning rate if improvement stops
         'nhidden':1, # number of hidden units
         'lam':0,
         'time_start':0,
         'seed':345,
         'nepochs':3,
         'compare_flag':'8',
         'window_width':65,
         'step_samples':45,
         'num_components':40} #0-> 2D v 3D, 1-> 2D v en, 2-> en v 3D, 3-> all

    folder = os.path.basename(__file__).split('.')[0]
    if not os.path.exists(folder): os.mkdir(folder)

    # load the dataset
    print 'Loading data...'
    X, y = load_data(s['compare_flag'])
    X = get_window_data(X, s['window_width'], s['step_samples'])
    X = apply_pca(X, s['num_components'])
    # split into train and dev
    split = int(0.9 * X.shape[0])
    x_train, y_train = X[:split], y[:split]
    x_dev, y_dev = X[split:], y[split:]
    print 'done'

    nclasses = len(numpy.unique(y_train))
    ntrials = len(x_train)
                   
    # instanciate the model
    numpy.random.seed(s['seed'])
    random.seed(s['seed'])
    rnn = model(    nh = s['nhidden'],
                    nc = nclasses,
                    di = x_train.shape[2],
                    nw = x_train.shape[1],
                    lam = s['lam'] )

    # train with early stopping on validation set
    best_f1 = -numpy.inf
    s['clr'] = s['lr']
    loss = []
    print x_train.shape
    for e in xrange(s['nepochs']):
        # shuffle
        for lst in [x_train, y_train]:
            random.seed(s['seed'])
            random.shuffle(lst)
        s['ce'] = e
        tic = time.time()
        for i in xrange(ntrials):
            trial = x_train[i][s['time_start']:,:]
            label = y_train[i]
            nll, _ = rnn.train(trial, label, s['clr'])
            # p_y = rnn.f(trial)
            # print p_y
            # print numpy.exp(-nll)
            if not i % 100:
                loss.append(numpy.sum(numpy.array([ rnn.compute_loss(x_train[j][s['time_start']:,:], \
                                                                y_train[j]) for j in xrange(ntrials) ])))
                print 'Loss:', loss[-1]
            if s['verbose']:
                print '[learning] epoch %i >> %2.2f%%'%(e,(i+1)*100./ntrials),'completed in %.2f (sec) <<\r'%(time.time()-tic),
                #sys.stdout.flush()
            
        # evaluation
        y_train_pred = [ rnn.classify(x[s['time_start']:,:]).astype('int32') for x in x_train ]
        y_dev_pred = [ rnn.classify(x[s['time_start']:,:]).astype('int32') for x in x_dev ]
        # y_test_pred = [ rnn.classify(x[s['time_start']:,:]).astype('int32') for x in x_test ]
    
        res_train = {'f1': f1_score(y_train, y_train_pred),
                   'p': precision_score(y_train, y_train_pred),
                   'r': recall_score(y_train, y_train_pred),
                   'a': accuracy_score(y_train, y_train_pred)}
        res_dev = {'f1': f1_score(y_dev, y_dev_pred),
                   'p': precision_score(y_dev, y_dev_pred),
                   'r': recall_score(y_dev, y_dev_pred),
                   'a': accuracy_score(y_dev, y_dev_pred)}
        # res_test = {'f1': f1_score(y_test, y_test_pred),
        #             'p': precision_score(y_test, y_test_pred),
        #             'r': recall_score(y_test, y_test_pred),
        #             'a': accuracy_score(y_test, y_test_pred)}

        if res_dev['f1'] > best_f1:
            s['be'] = e
            rnn.save(folder)
            best_f1 = res_dev['f1']
            if s['verbose']:
                print 'NEW BEST: epoch', e, 'valid F1', res_dev['f1'], ' '*20
            s['vf1'], s['vp'], s['vr'] = res_dev['f1'], res_dev['p'], res_dev['r']
        print 'current f1:', res_dev['f1'], 'current precision:', res_dev['p'], \
              'current recall:', res_dev['r'], 'current accuracy:', res_dev['a']
        print 'train f1:', res_train['f1'], 'train accuracy:', res_train['a']
        
        # learning rate decay if no improvement in 10 epochs
        if s['decay'] and abs(s['be']-s['ce']) >= 10: s['clr'] *= 0.5 
        if s['clr'] < 1e-5: break
    print loss
    plt.plot(numpy.array(loss))
    plt.show()
    print 'BEST RESULT: epoch', e, 'dev F1', s['vf1'], 'with the model', folder


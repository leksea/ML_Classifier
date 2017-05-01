import numpy as np
import random
import os
import sys
from numpy import transpose as permute
from scipy.io import loadmat, savemat
from scipy.signal import detrend
from copy import deepcopy as dcpy


front_electrodes = [17, 127, 126, 128, 125, 43, 120, 48, 119, 21, 25, 32, 38, \
					44, 14, 8, 1, 121, 114, 15, 22, 26, 33, 9, 2, 122, 16, 18, \
					23, 27, 34, 10, 3, 123, 116, 11, 19, 24, 4, 124, 48, 120];
electrodes = [i-1 for i in range(1,129) if i not in front_electrodes]
num_channels = 128
empty_map = {'trials': [], 'mean_trials': [], 'groups': []}
# add comment to say what numbers are
conds_map = {0: [4, 1], 1: [4, 3], 2: [0, 5], 3: [0, 3], 4: [2, 5], 5: [2, 1]}
cond_names_map = {0: '2D_a', 1: '2D_b', 2: '3D_a', \
				  3: '3D_b', 4: 'en_a', 5: 'en_b'}
cond_names = {0: '2D', 1: '3D', 2: 'en'}


def get_num_conds(path):
	data_names = os.listdir(path)
	num_cond = 0
	for name in data_names:
		if 'Raw' in name:
			cond = int(name[name.rfind('c00')+3])
			if cond > num_cond:
				num_cond = cond
	return num_cond


def concat(curr_entry, new_data):
	if not len(curr_entry):
		return new_data
	else:
		return np.concatenate((curr_entry,new_data))


def generate_data(path, savepath):
	subjs = [item for item in os.listdir(path) if '.' not in item]
	num_cond = get_num_conds(os.path.join(path, subjs[0]))
	subj_data = [[dcpy(empty_map) for i in range(num_cond)] for j in range(len(subjs))]
	for subj_ind in xrange(len(subjs)):
		subj = subjs[subj_ind]
		print 'Adding subject', subj
		subj_path = os.path.join(path, subj)
		files = [item for item in os.listdir(subj_path) if 'Raw' in item]
		for data_ind in xrange(len(files)):
			data_file = files[data_ind]
			cond = int(data_file[data_file.rfind('c00')+3])-1
			data = loadmat(os.path.join(subj_path,data_file))
			raw_data = data['RawTrial'][:,electrodes]
			num_TP = raw_data.shape[0] / data['NmbEpochs']
			raw_data = permute(raw_data.reshape(num_TP, data['NmbEpochs'], len(electrodes)),axes=(1,2,0))
			valid_epochs = [i for i in range(data['NmbEpochs']) if \
							np.sum(data['IsEpochOK'][i]) > num_channels/2]
			raw_data = raw_data[valid_epochs,:,:]
			num_epochs = raw_data.shape[0]
			if num_epochs:
				entry = subj_data[subj_ind][cond]
				entry['groups'].append(num_epochs)
				entry['trials'] = concat(entry['trials'],raw_data)
				raw_data_mean = np.mean(raw_data, axis=0)[None, :]
				entry['mean_trials'] = concat(entry['mean_trials'],raw_data_mean)
		for cond in xrange(num_cond):
			entry = subj_data[subj_ind][cond]
			entry['groups'] = np.squeeze(entry['groups'])
			entry['trials'] = np.squeeze(entry['trials'])
			entry['mean_trials'] = np.squeeze(entry['mean_trials'])

	if not os.path.isdir(savepath):
		os.makedirs(savepath)

	savemat(os.path.join(savepath,'subj_data'), {'subj_data': subj_data})
	return subj_data


def generate_files_sony(old_subj_data, savepath):
	num_subj = len(old_subj_data)
	num_cond = len(old_subj_data[0])
	subj_data = [[dcpy(empty_map) for i in range(num_cond)] for j in range(num_subj)]
	clip_size = old_subj_data[0][0]['trials'].shape[2] / 2
	num_electrodes = old_subj_data[0][0]['trials'].shape[1]
	for subj_ind in range(num_subj):
		for cond in range(num_cond):
			conds = conds_map[cond]
			old_entry = old_subj_data[subj_ind][cond]
			for i in range(2):
				clip = range(clip_size * i, clip_size * (i+1))
				entry = subj_data[subj_ind][conds[i]]
				entry['trials'] = concat(entry['trials'],old_entry['trials'][:,:,clip])

	if not os.path.isdir(savepath):
		os.makedirs(savepath)

	dats = []
	dats_det = []
	for cond in range(num_cond):
		print 'Generating condition', cond+1, 'files...'
		dats.append(np.zeros((0,clip_size,num_electrodes)))
		dats_det.append(np.zeros((0,clip_size,num_electrodes)))
		for subj in range(num_subj):
			trials = subj_data[subj][cond]['trials']
			dats[-1] = concat(dats[-1],permute(trials,axes=(0,2,1)))
		savename = os.path.join(savepath,cond_names_map[cond])
		savemat(savename, {'dat': dats[-1]})
		dats_det[-1] = detrend(dats[-1],axis=1)
		savename = os.path.join(savepath,cond_names_map[cond] + '_det')
		savemat(savename, {'dat': dats_det[-1]})


def generate_files(subj_data, savepath):
	if not os.path.isdir(savepath):
		os.makedirs(savepath)

	num_subj = len(subj_data)
	num_cond = len(subj_data[0])
	clip_size = subj_data[0][0]['trials'].shape[2]
	num_electrodes = subj_data[0][0]['trials'].shape[1]
	for cond in range(num_cond):
		print 'Generating condition', cond+1, 'files...'
		dat_groups = []
		dat_trials = np.zeros((0,clip_size,num_electrodes))
		dat_mean = np.zeros((0,clip_size,num_electrodes))
		for subj in range(num_subj):
			dat_groups = concat(dat_groups,subj_data[subj][cond]['groups'])
			dat_trials = concat(dat_trials,subj_data[subj][cond]['trials'])
			dat_mean = concat(dat_mean,subj_data[subj][cond]['mean_trials'])

		savename = os.path.join(savepath, 'groups_cond_' + str(cond+1))
		savemat(savename, {'groups': dat_groups})

		savename = os.path.join(savepath, 'trials_cond_' + str(cond+1))
		savemat(savename, {'dat': dat_trials})
		dat_trials = detrend(dat_trials,axis=1)
		savemat(savename + '_det', {'dat': dat_trials})

		savename = os.path.join(savepath, 'mean_cond_' + str(cond+1))
		savemat(savename, {'dat': dat_mean})
		dat_mean = detrend(dat_mean,axis=1)
		savemat(savename + '_det', {'dat': dat_mean})


if __name__ == '__main__':
	path = raw_input('Please enter the path to the data: ')
	while not os.path.isdir(path):
		path = raw_input('Please enter a valid path for the data: ')
	data_path = raw_input(("Please name the path for the new data (only "
						   	"enter an existing folder if you\'d like to "
						   	"overwrite the data in it): "))
	sony_prompt = ("Is this the Sony experiment or an experiment with "
				   "two conditions present in each trial? [Y/N]: ")
	is_sony = raw_input(sony_prompt)[0].lower()
	while is_sony not in ['y', 'n']:
		is_sony = raw_input('Please answer either yes (Y) or no (N): ')[0].lower()
	if os.path.isfile(os.path.join(data_path,'subj_data')):
		subj_data = loadmat(os.path.join(data_path,'subj_data'))
		subj_data = subj_data['subj_data']
	else:
		subj_data = generate_data(path, data_path)
	if is_sony == 'y':
		generate_files_sony(subj_data, data_path)
	else:
		generate_files(subj_data, data_path)
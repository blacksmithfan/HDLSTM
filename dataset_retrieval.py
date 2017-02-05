import numpy as np
import cv2
from random import shuffle
from os import listdir
from os.path import isfile, join
import scipy.io as spio

def loadmat(filename):
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

class Dataset:
    def __init__(self, train_path, test_path, n_classes, shuffleType, seqLength, target, n_hidden):
        # Load training images (path) and labels
        self.train_models = [f for f in listdir(train_path) if isfile(join(train_path, f))]
        self.train_path = train_path
        self.test_path = test_path
        self.target_vector = target
        self.n_hidden = n_hidden
        shuffle(self.train_models)

        self.test_models = [f for f in listdir(test_path) if isfile(join(test_path, f))]
        
        print(len(self.train_models))



        # with open(train_list) as f:
        #     lines = f.readlines()
        #     self.train_image = []
        #     self.train_label = []
        #     if shuffleType == 'normal':
        #         shuffle(lines)
        #         for l in lines:
        #             items = l.split()
        #             self.train_image.append(items[0])
        #             self.train_label.append(int(items[1]))
        #     elif shuffleType == 'seq':
        #         num_seq = len(lines) / seqLength
        #         # ind = np.random.permutation(num_seq)
        #         for i in range(num_seq):
        #             for jj in range(seqLength):
        #                 items = lines[i*seqLength + jj].split()
        #                 self.train_image.append(items[0])
        #                 self.train_label.append(int(items[1]))

        # with open(test_list) as f:
        #     lines = f.readlines()
        #     self.test_image = []
        #     self.test_label = []
        #     if shuffleType == 'normal':
        #         shuffle(lines)
        #         for l in lines:
        #             items = l.split()
        #             self.test_image.append(items[0])
        #             self.test_label.append(int(items[1]))
        #     elif shuffleType == 'seq':
        #         num_seq = len(lines) / seqLength
        #         # ind = np.random.permutation(num_seq)
        #         for i in range(num_seq):
        #             for jj in range(seqLength):
        #                 items = lines[i*seqLength + jj].split()
        #                 self.test_image.append(items[0])
        #                 self.test_label.append(int(items[1]))



        # Load testing images (path) and labels
        #with open(test_list) as f:
        #    lines = f.readlines()
        #    self.test_image = []
        #    self.test_label = []
        #    for l in lines:
        #        items = l.split()
        #        self.test_image.append(items[0])
        #        self.test_label.append(int(items[1]))

        # Init params
        self.train_ptr = 0
        self.test_ptr = 0
        self.train_size = len(self.train_models)
        self.test_size = len(self.test_models)

        self.n_classes = n_classes



    def next_batch(self, batch_size, phase):
        # Get next batch of image (path) and labels
        if phase == 'train':
            if self.train_ptr + batch_size < self.train_size:
                paths = self.train_models[self.train_ptr:self.train_ptr + batch_size]
                # path_f = paths[10::11]
                # print(path_f)
                # labels = self.train_label[self.train_ptr:self.train_ptr + batch_size]
                self.train_ptr += batch_size
            else:
                new_ptr = (self.train_ptr + batch_size)%self.train_size
                paths = self.train_models[self.train_ptr:] + self.train_models[:new_ptr]
                # labels = self.train_label[self.train_ptr:] + self.train_label[:new_ptr]
                self.train_ptr = new_ptr
            mat_content = loadmat(join(self.train_path, paths[0]))

            batch_data = np.array(mat_content['HKS_ShapeNet'])
            batch_label = np.zeros((batch_size, 1))
            for i in xrange(1,len(paths)):
                # img = cv2.imread(paths[i])
                mat_content = loadmat(join(self.train_path, paths[i]))


                batch_data = np.dstack((batch_data, mat_content['HKS_ShapeNet']))
                
                label = np.array(mat_content['shape_label'])
                batch_label[i]=label-1
        elif phase == 'test':
            if self.test_ptr + batch_size < self.test_size:
                paths = self.test_models[self.test_ptr:self.test_ptr + batch_size]
                #labels = self.test_label[self.test_ptr:self.test_ptr + batch_size]
                self.test_ptr += batch_size
            else:
                new_ptr = (self.test_ptr + batch_size)%self.test_size
                paths = self.test_models[self.test_ptr:] + self.test_models[:new_ptr]
                #labels = self.test_label[self.test_ptr:] + self.test_label[:new_ptr]
                self.test_ptr = new_ptr
            mat_content = loadmat(join(self.test_path, paths[0]))

            batch_data = np.array(mat_content['HKS_ShapeNet'])
            batch_label = np.zeros((batch_size, 1))
            for i in xrange(1,len(paths)):
                # img = cv2.imread(paths[i])
                mat_content = loadmat(join(self.test_path, paths[i]))


                batch_data = np.dstack((batch_data, mat_content['HKS_ShapeNet']))
                
                label = np.array(mat_content['shape_label'])

                batch_label[i]=label-1
        else:
            return None, None

        # Read images
        # images = np.ndarray([batch_size, self.crop_size, self.crop_size, 3])

        batch_target = np.zeros((batch_size, self.n_hidden))
        for i in range(batch_size):
            batch_target[i,:] = self.target_vector[batch_label.astype(int)[i],:]

        return batch_data, batch_target

    def final_eval(self, phase):
        if phase == 'train':
            paths_train = self.train_models

            mat_content = loadmat(join(self.train_path, paths_train[0]))
            batch_data = np.array(mat_content['HKS_ShapeNet'])
            label = np.array(mat_content['shape_label'])
            one_hot_labels = np.zeros((len(paths_train), self.n_classes))
            one_hot_labels[0][label-1]=1 # pay attention that the "one_hot_labels" variable has different forms for training data and testing data 

            for i in xrange(1,len(paths_train)):
                # img = cv2.imread(paths[i])
                mat_content = loadmat(join(self.train_path, paths_train[i]))
                batch_data = np.dstack((batch_data, mat_content['HKS_ShapeNet']))
                label = np.array(mat_content['shape_label'])
                one_hot_labels[i][label-1]=1
        if phase == 'test':
            paths_test = self.test_models
            mat_content = loadmat(join(self.test_path, paths_test[0]))
            batch_data = np.array(mat_content['HKS_ShapeNet'])
            label = np.array(mat_content['shape_label'])
            one_hot_labels = np.zeros((len(paths_test), 1)) # pay attention that the "one_hot_labels" variable has different forms for training data and testing data 
            one_hot_labels[0]=label-1

            for i in xrange(1,len(paths_test)):
                # img = cv2.imread(paths[i])
                mat_content = loadmat(join(self.test_path, paths_test[i]))
                batch_data = np.dstack((batch_data, mat_content['HKS_ShapeNet']))
                label = np.array(mat_content['shape_label'])
                one_hot_labels[i]=label-1

        return batch_data, one_hot_labels

    def final_eval_label(self, phase):
        if phase == 'train':
            paths_train = self.train_models

            mat_content = loadmat(join(self.train_path, paths_train[0]))
            batch_data = np.array(mat_content['HKS_ShapeNet'])
            label = np.array(mat_content['shape_label'])
            one_hot_labels = np.zeros((len(paths_train), 1))
            one_hot_labels[0]=label-1

            for i in xrange(1,len(paths_train)):
                # img = cv2.imread(paths[i])
                mat_content = loadmat(join(self.train_path, paths_train[i]))
                batch_data = np.dstack((batch_data, mat_content['HKS_ShapeNet']))
                label = np.array(mat_content['shape_label'])
                one_hot_labels[i]=label-1
        if phase == 'test':
            paths_test = self.test_models
            mat_content = loadmat(join(self.test_path, paths_test[0]))
            batch_data = np.array(mat_content['HKS_ShapeNet'])
            label = np.array(mat_content['shape_label'])
            one_hot_labels = np.zeros((len(paths_test), 1))
            one_hot_labels[0]=label-1

            for i in xrange(1,len(paths_test)):
                # img = cv2.imread(paths[i])
                mat_content = loadmat(join(self.test_path, paths_test[i]))
                batch_data = np.dstack((batch_data, mat_content['HKS_ShapeNet']))
                label = np.array(mat_content['shape_label'])
                one_hot_labels[i]=label-1

        return batch_data, one_hot_labels
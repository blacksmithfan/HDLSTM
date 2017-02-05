import numpy as np
import cv2
from random import shuffle
class Dataset:
    def __init__(self, train_list, test_list, n_classes, shuffleType, seqLength, CNN_type, target, n_hidden):
        # Load training images (path) and labels
        with open(train_list) as f:
            lines = f.readlines()
            self.train_image = []
            self.train_label = []
            if shuffleType == 'normal':
                shuffle(lines)
                for l in lines:
                    items = l.split()
                    self.train_image.append(items[0])
                    self.train_label.append(int(items[1]))
            elif shuffleType == 'seq':
                num_seq = len(lines) / seqLength
                # ind = np.random.permutation(num_seq)
                for i in range(num_seq):
                    for jj in range(seqLength):
                        items = lines[i*seqLength + jj].split()
                        self.train_image.append(items[0])
                        self.train_label.append(int(items[1]))

        with open(test_list) as f:
            lines = f.readlines()
            self.test_image = []
            self.test_label = []
            if shuffleType == 'normal':
                shuffle(lines)
                for l in lines:
                    items = l.split()
                    self.test_image.append(items[0])
                    self.test_label.append(int(items[1]))
            elif shuffleType == 'seq':
                num_seq = len(lines) / seqLength
                # ind = np.random.permutation(num_seq)
                for i in range(num_seq):
                    for jj in range(seqLength):
                        items = lines[i*seqLength + jj].split()
                        self.test_image.append(items[0])
                        self.test_label.append(int(items[1]))



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
        self.target_vector = target
        self.n_hidden = n_hidden
        self.train_size = len(self.train_label)
        self.test_size = len(self.test_label)
        if CNN_type == 'vgg':
            self.crop_size = 224
        else:
            self.crop_size = 227
        self.scale_size = 256
        self.mean = np.array([122., 104., 100.])
        # self.mean = np.array([104., 117., 124.])
        self.n_classes = n_classes

    def next_batch(self, batch_size, phase):
        # Get next batch of image (path) and labels
        if phase == 'train':
            if self.train_ptr + batch_size < self.train_size:
                paths = self.train_image[self.train_ptr:self.train_ptr + batch_size]
                # path_f = paths[10::11]
                # print(path_f)
                labels = self.train_label[self.train_ptr:self.train_ptr + batch_size]
                self.train_ptr += batch_size
            else:
                new_ptr = (self.train_ptr + batch_size)%self.train_size
                paths = self.train_image[self.train_ptr:] + self.train_image[:new_ptr]
                labels = self.train_label[self.train_ptr:] + self.train_label[:new_ptr]
                self.train_ptr = new_ptr
        elif phase == 'test':
            if self.test_ptr + batch_size < self.test_size:
                paths = self.test_image[self.test_ptr:self.test_ptr + batch_size]
                labels = self.test_label[self.test_ptr:self.test_ptr + batch_size]
                self.test_ptr += batch_size
            else:
                new_ptr = (self.test_ptr + batch_size)%self.test_size
                paths = self.test_image[self.test_ptr:] + self.test_image[:new_ptr]
                labels = self.test_label[self.test_ptr:] + self.test_label[:new_ptr]
                self.test_ptr = new_ptr
        else:
            return None, None

        # Read images
        images = np.ndarray([batch_size, self.crop_size, self.crop_size, 3])
        for i in xrange(len(paths)):
            img = cv2.imread(paths[i])
            # print(paths[i])
            h, w, c = img.shape
            assert c==3

            img = cv2.resize(img, (self.scale_size, self.scale_size))
            img = img.astype(np.float32)
            img -= self.mean
            shift = int((self.scale_size-self.crop_size)/2)
            img_crop = img[shift:shift+self.crop_size, shift:shift+self.crop_size, :]
            images[i] = img_crop

        # Expand labels
        # one_hot_labels = np.zeros((batch_size, self.n_classes))
        # for i in xrange(len(labels)):
        #     one_hot_labels[i][labels[i]] = 1
        batch_target = np.zeros((batch_size, self.n_hidden))
        for i in range(batch_size):
            # print(labels[i])
            batch_target[i,:] = self.target_vector[labels[i],:]


        one_hot_labels = batch_target
        return images, one_hot_labels

    def final_eval(self, phase):
        if phase == 'train':
            paths = self.train_image
            labels = self.train_label

        if phase == 'test':
            paths = self.test_image
            labels = self.test_label

        images = np.ndarray([len(paths), self.crop_size, self.crop_size, 3])
        for i in xrange(len(paths)):
            img = cv2.imread(paths[i])
            # print(paths[i])
            h, w, c = img.shape
            assert c==3

            img = cv2.resize(img, (self.scale_size, self.scale_size))
            img = img.astype(np.float32)
            img -= self.mean
            shift = int((self.scale_size-self.crop_size)/2)
            img_crop = img[shift:shift+self.crop_size, shift:shift+self.crop_size, :]
            images[i] = img_crop

        batch_target = np.zeros((len(paths), self.n_hidden))
        for i in range(len(paths)):
            # print(labels[i])
            batch_target[i,:] = self.target_vector[labels[i],:]

        return images, batch_target, labels


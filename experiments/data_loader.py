import numpy as np
from pathlib import Path
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ImageDataLoader:
    def __init__(self, batch_size, classes, samples_per_class, data_folder, flatten):
        """
        creates a data loader for the given data folder
        :param batch_size: number of samples per batch
        :param classes: list of class file names, if classes is None all npy files in data folder are used
        :param samples_per_class: number of samples per class
        :param data_folder: path to the folder where the data is stored
        :param flatten: whether to flatten the data
        """
        self.flatten = flatten
        self.batch_size = batch_size
        self.classes = classes
        folder = Path(data_folder)
        if self.classes is None:
            self.classes = [file.name for file in folder.glob("*.npy")]
        self.samples_per_class = samples_per_class
        self.data_folder = data_folder
        self.total_samples = len(self.classes) * self.samples_per_class
        self.train_samples_per_class = int(self.samples_per_class * .8)
        self.test_samples_per_class = samples_per_class - self.train_samples_per_class
        self.train_samples = []
        for i in range(len(self.classes)):
            for j in range(self.train_samples_per_class):
                self.train_samples.append((i, j))
        self.train_samples = np.array(self.train_samples)
        self.test_samples = []
        for i in range(len(self.classes)):
            for j in range(self.train_samples_per_class , self.train_samples_per_class + self.test_samples_per_class):
                self.test_samples.append((i, j))
        self.test_samples = np.array(self.test_samples)
        self.current_train_batch = 0
        self.current_test_batch = 0
        self.data_access = []
        for i in range(len(self.classes)):
            path = folder / self.classes[i]
            self.data_access.append(np.load(path, mmap_mode="r"))
        self.input_size = self.data_access[0][0].shape[0]
        self.num_classes = len(self.classes)

    def shuffle(self):
        '''
        shuffles the data
        '''
        np.random.shuffle(self.train_samples)
        self.current_train_batch = 0

    def next_batch(self, train):
        '''
        returns the next batch of training data or testing data
        :param train: determines whether to return training or testing data
        :return: next batch of data
        '''
        if train and self.current_train_batch*self.batch_size < self.train_samples.shape[0]:
            batch_X, batch_y = self._create_batch(self.current_train_batch, self.train_samples)
            self.current_train_batch += 1
            return batch_X, batch_y
        elif not train and self.current_test_batch*self.batch_size < self.test_samples.shape[0]:
            batch_X, batch_y = self._create_batch(self.current_test_batch, self.test_samples)
            self.current_test_batch += 1
            return batch_X, batch_y
        elif not train and self.current_test_batch*self.batch_size >= self.test_samples.shape[0]:
            self.current_test_batch = 0
            return None, None
        return None, None

    def _create_batch(self, batch_index, samples):
        '''
        gets the batch X and y from the dataset given a batch index
        :param batch_index: the index of the batch needed
        :param samples: tuples of (class_idx, sample_idx)
        :return: batch_X, batch_y from the dataset
        '''
        start = batch_index * self.batch_size
        end = min(start + self.batch_size, samples.shape[0])

        batch_samples = samples[start:end]

        batch_X = [
            self.data_access[class_idx][sample_idx]
            for class_idx, sample_idx in batch_samples
        ]

        batch_X = torch.from_numpy(
            np.asarray(batch_X, dtype=np.float32)
        ).to(device) / 255.0

        if not self.flatten:
            batch_X = batch_X.reshape(-1, 1, 28, 28)

        labels = batch_samples[:, 0]

        batch_y = torch.zeros(
            len(labels),
            self.num_classes,
            device=device
        )

        batch_y[
            torch.arange(len(labels), device=device),
            torch.from_numpy(labels).long().to(device)
        ] = 1.0

        return batch_X, batch_y


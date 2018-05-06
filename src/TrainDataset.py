import torchvision.datasets as datasets
import os
import numpy as np
from tqdm import tqdm

class TrainDataset(datasets.ImageFolder):
    '''
    '''
    def __init__(self, dir, transform=None):

        super(testDataset, self).__init__(dir,transform)
        self.validation_images = self.get_paths(dir)
        self.transform = transform

    def get_paths(self,dataroot):
        file_list = sorted(os.listdir(dataroot))
        path = []
        label = 0
        for file in file_list:
            second_root = os.path.join(dataroot,file)
            for jpg in os.listdir(second_root):
                path.append((os.path.join(second_root,jpg),label))
            label += 1
            if(label == 6400):
                label = 0
        return path

    def __getitem__(self, index):
        '''

        Args:
            index: Index of the triplet or the matches - not of a single image

        Returns:

        '''

        def transform(img_path):
            """Convert image into numpy array and apply transformation
               Doing this so that it is consistent with all other datasets
               to return a PIL Image.
            """

            img = self.loader(img_path)
            return self.transform(img)

        (path,label) = self.validation_images[index]
        img = transform(path)
        return img,label


    def __len__(self):
        return len(self.validation_images)



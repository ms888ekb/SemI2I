from torch.utils import data
import os
import torch
import numpy as np
from glob import glob
from osgeo import gdal, gdalconst
from torchvision import transforms


def normalize_image(image):
    image = image.astype(np.uint8)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    img_tr = transform(image)

    transform_norm = transforms.Compose([
        transforms.Normalize(mean=(0.5, 0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5, 0.5))
    ])
    img_normalized = transform_norm(img_tr)

    return img_normalized


class GeoDataLoaderStd(data.Dataset):
    def __init__(self, x_data=None, y_data=None, size=None, shuffle=True):
        if y_data is None:
            self.iamsource = False
        else:
            self.iamsource = True
        self.source_data = self.__lookup(x_data)
        if y_data is not None:
            self.source_labels = y_data
        self.src_indexes = []
        self.resize_size = size
        self.shuffle = shuffle
        self.dataset_size = None
        self.__on_init()

    def get_sample_batch(self, num=1):
        sample_indexes = np.random.randint(0, high=len(self), size=num)
        return self[sample_indexes[0]]

    def get_num_samples(self):
        return self.dataset_size

    def __on_init(self):
        source_data_index = np.arange(len(self.source_data))
        if self.shuffle:
            np.random.shuffle(source_data_index)
        self.dataset_size = len(source_data_index)
        self.src_indexes = source_data_index

    @staticmethod
    def __lookup(path):
        mask = os.path.join(path, "*.TIF")
        f_list = glob(mask)
        return f_list

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        index = self.src_indexes[index]
        image_ds = gdal.Open(self.source_data[index], gdalconst.GA_ReadOnly)
        image_georef = image_ds.GetGeoTransform()
        assert image_ds is not None, "The input image is None. Check the image path."
        image = image_ds.ReadAsArray()
        assert image.shape[0] == 4, "Wrong input image band number"
        image = np.asarray(image, np.float)
        image = image.transpose((1, 2, 0))
        image = normalize_image(image)
        return image.clone().detach().type(dtype=torch.float).cpu(), image_georef


class GeoDataLoaderWithName(data.Dataset):
    def __init__(self, x_data=None, y_data=None, size=None, shuffle=False):
        if y_data is None:
            self.iamsource = False
        else:
            self.iamsource = True
        self.source_data = self.__lookup(x_data)
        if y_data is not None:
            self.source_labels = y_data
        self.src_indexes = []
        self.resize_size = size
        self.shuffle = shuffle
        self.dataset_size = None
        self.__on_init()

    def get_sample_batch(self, num=1):
        sample_indexes = np.random.randint(0, high=len(self), size=num)
        return self[sample_indexes[0]]

    def get_num_samples(self):
        return self.dataset_size

    def __on_init(self):
        source_data_index = np.arange(len(self.source_data))
        if self.shuffle:
            np.random.shuffle(source_data_index)
        self.dataset_size = len(source_data_index)
        self.src_indexes = source_data_index

    @staticmethod
    def __lookup(path):
        mask = os.path.join(path, "*.TIF")
        f_list = glob(mask)
        return f_list

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        index = self.src_indexes[index]
        image_ds = gdal.Open(self.source_data[index], gdalconst.GA_ReadOnly)
        name = os.path.split(self.source_data[index])[1]
        image_georef = image_ds.GetGeoTransform()
        crs = image_ds.GetProjection()
        assert image_ds is not None, "The input image is None. Check the image path."
        image = image_ds.ReadAsArray()
        assert image.shape[0] == 4, f"Wrong input image band number: given shape: {image.shape}"
        image = np.asarray(image, np.float)
        image = image.transpose((1, 2, 0))
        image = normalize_image(image)
        image_ds = None

        if self.iamsource:
            sample_name = os.path.split(self.source_data[index])[1]
            if not os.path.exists(os.path.join(self.source_labels, sample_name)):
                print(f'Corresponding label for {sample_name} was not found.')
            label_ds = gdal.Open(os.path.join(self.source_labels, sample_name), gdalconst.GA_ReadOnly)
            label = label_ds.ReadAsArray()
            label = np.asarray(label, np.uint8)
            return image.clone().detach().type(dtype=torch.float).cpu(), label.copy(), image_georef, crs, name
        else:
            return image.clone().detach().type(dtype=torch.float).cpu(), image_georef, crs, name


if __name__ == '__main__':
    pass

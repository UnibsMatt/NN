from torch.utils import data
import os
from PIL import Image
import numpy as np
from TorchTest.Utilities import to_tensor


class BatchLoader(data.Dataset):
    def __init__(self, folder, input_shape, prediction=False):
        self.input_shape = (input_shape[0], input_shape[1])
        self.x_data = []
        self.y_data = []
        i = 0
        if prediction:
            for file in os.listdir(folder):
                self.x_data.append(os.path.join(folder, file))
                self.y_data.append(i)
                i += 1
            assert (len(self.x_data) == len(self.y_data))
        else:
            for folders in os.listdir(folder):
                sub_dir = os.path.join(folder, folders)
                for file in os.listdir(sub_dir):
                    self.x_data.append(os.path.join(sub_dir, file))
                    self.y_data.append(i)
                i += 1
            assert (len(self.x_data) == len(self.y_data))

    def __len__(self):
        assert (len(self.x_data) == len(self.y_data))
        return len(self.x_data)

    def __getitem__(self, index):
        image = Image.open(self.x_data[index])
        image = image.resize(self.input_shape)
        image = np.asarray(image)
        image_tensor = to_tensor(image)
        return image_tensor, np.asarray(self.y_data[index]), self.x_data[index]

    def sample_size(self):
        return len(self.x_data)

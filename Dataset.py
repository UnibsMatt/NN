from torch.utils.data import Dataset
import os
import numpy as np
import h5py
import torchvision.transforms as transforms
from TorchTest.Utilities import data_split, safe_split, to_tensor, convert_image_list_to_h5
from TorchTest.transormation import ToTens


class AllDataSet(Dataset):
    def __init__(self, root, input_size, split=.8, augmentation=False, h5py_filename="dataset.h5py"):
        self.input_size = input_size
        self.split = safe_split(split)
        self.augmentation = augmentation
        self.root = root
        self.samples_name = []
        self.label = []
        self.data_set = "train"

        self.transformation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(270, expand=False),
            transforms.RandomVerticalFlip(),
            #transforms.Resize(self.input_size),
            transforms.ColorJitter(brightness=.08, hue=.05, saturation=.05),
            transforms.RandomGrayscale(),
            ToTens(),
            #RandomEraseSquare(),

        ])
        self.load_data()
        self.train_sample, self.validation_sample = data_split(self.samples_name, self.split)

        convert_image_list_to_h5(self.samples_name[:, 0], input_size, h5py_filename, compression=6)

    def load_data(self):
        i = 0
        for folder in os.listdir(self.root):
            class_path = os.path.join(self.root, folder)
            for _, sub_folder, file in os.walk(class_path):
                self.samples_name.extend(os.path.join(class_path, file_name) for file_name in file)
                self.label.extend([i for x in range(len(file))])
            i += 1
        # trasformo in numpy array
        self.samples_name = np.asarray(self.samples_name)
        self.label = np.asarray(self.label)
        # concateno su un nuovo asse i file con le loro label
        self.samples_name = np.stack((self.samples_name, self.label), axis=1)

    def __len__(self):
        if self.data_set == "train":
            return len(self.train_sample)
        else:
            return len(self.validation_sample)

    def __getitem__(self, index):

        if self.data_set == "train":
            examination_data = self.train_sample[index]
        else:
            examination_data = self.validation_sample[index]

        with h5py.File("data_set_paste.h5py", 'r') as hf:
            image = hf.get(examination_data[0])

            if self.augmentation:
                # se c'e augmentation passo il tensore nel trasformer
                image_tensor = self.transformation(image[:])
            else:
                image_tensor = to_tensor(image[:])

        return image_tensor, int(examination_data[1]), examination_data[0]

    def change_data_set(self, data_set):
        if data_set not in ["train", "validation"]:
            print("Invalid dataset: Available are train or validation")

        self.data_set = data_set

    def get_train_val(self):
        return self.train_sample, self.validation_sample

    def print_train_validation_samples(self, path):
        data_type = ["train", "validation"]
        data_set = self.get_train_val()
        for index, data in enumerate(data_type):

            file = open(path + "/" + data+".txt", "a+")
            for item in data_set[index]:
                file.write(item[0])
                file.write("\n")

            file.close()

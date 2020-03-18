import torch
from Dataset import AllDataSet
from torch.utils.data import DataLoader
from Utilities import conf_matrix, save_mismatch, create_work_folder
import numpy as np


class Validation:
    def __init__(self, model, path, input_shape):

        self.input_shape = np.asarray(input_shape)

        self.validation_root_folder = "ValidationResult"
        self.work_directory_path = create_work_folder(self.validation_root_folder, path)

        self.dataset = AllDataSet(path, self.input_shape[:2], split=1, augmentation=False)

        self.generator = DataLoader(self.dataset, batch_size=32, shuffle=True)

        self.device = torch.device("cuda")

        self.model = model.to(self.device)
        self.run()

    def run(self):
        self.model.eval()
        print("Start Evaluation\n")

        true_label = []
        prediction_label = []
        confidences = []
        image_name = []
        with torch.no_grad():
            for batch_idx, (images, label, name) in enumerate(self.generator):

                x_train = images.to(self.device)
                y_train = label.long().to(self.device)

                #out = self.model(x_train)
                out = self.model(x_train, torch.zeros(1))

                #predicted_class, confidence, pos_x, pos_y = out[:, 1], out[:, 0], out[:, 2], out[:, 3]
                predicted_class, confidence, pos_x, pos_y = out[1], out[0], out[2], out[3]

                prediction_label.extend(predicted_class.cpu().numpy())
                true_label.extend(y_train.cpu())
                image_name.extend(name)
                confidences.extend(confidence.cpu().numpy())

                x_train.cpu()
                y_train.cpu()
                print('\rStep [%d/%d]' % (batch_idx, self.generator.__len__()), end="", flush=True)

            mismatch = conf_matrix(prediction_label, true_label, ["bad", "good"], path=self.work_directory_path)
            mismatch = mismatch.squeeze()
            image_name = np.asarray(image_name)
            prediction_label = np.asarray(prediction_label)
            confidences = np.asarray(confidences)
            save_mismatch(self.work_directory_path, image_name[mismatch], confidences[mismatch], ["bad", "good"], prediction_label[mismatch])

import torch.optim as optim
import torch.nn as nn
import torch
from Dataset import AllDataSet
from torch.utils.data import DataLoader
from Utilities import conf_matrix, create_work_folder
import os
import numpy as np
import time


class TrainingA:
    def __init__(self, model, path, input_shape, epochs=65, split=.8, augmentation=False):

        training_root_folder = "TrainingResult"
        self.work_directory_path = create_work_folder(training_root_folder, path)

        # altezza, larghezza, canali
        self.input_shape = np.asarray(input_shape)

        self.dataset = AllDataSet(path, self.input_shape[:2], split=split, augmentation=augmentation)

        self.dataset.print_train_validation_samples(self.work_directory_path)

        self.training_generator = DataLoader(self.dataset, batch_size=64, shuffle=True)
        self.testing_generator = DataLoader(self.dataset, batch_size=48, shuffle=True)

        self.device = torch.device("cuda")
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=0.00003, weight_decay=0.04)
        self.criterion = nn.CrossEntropyLoss()
        self.run(epochs)

    def run(self, epochs):
        self.model.train()
        patience = 5
        model_best_accuracy = 0.0
        for epoch in range(epochs):
            print("Start Training\n")
            train_acc = 0.0
            self.dataset.change_data_set("train")
            self.model.train()

            for batch_idx, (images, label, name) in enumerate(self.training_generator):
                self.model.cuda()
                self.optimizer.zero_grad()

                start = time.time()
                x_train = images.to(self.device)
                y_train = label.long().to(self.device)

                start = time.time()
                out = self.model(x_train, torch.zeros(1).cuda())

                #predicted_class, confidence, pos_x, pos_y = out[:, 1], out[:, 0], out[:, 2], out[:, 3]
                predicted_class, confidence, pos_x, pos_y, heat = out[1], out[0], out[2], out[3], out[4]
                coord = torch.stack((out[2], out[3]), dim=1)
                loss = self.criterion(coord, y_train)

                start = time.time()
                loss.backward()

                self.optimizer.step()
                train_acc += torch.sum(predicted_class == y_train.data)
                print('\rEpoch [%d/%d], Step [%d/%d], Loss [%f]'
                      % (epoch, epochs, batch_idx, self.training_generator.__len__(), loss.item()), end="",
                      flush=True)

            correct_prediction = train_acc.item()
            train_acc = correct_prediction / len(self.dataset)
            print("\nTraining Accuracy = [%f], predicted: [%d/%d]" % (train_acc, correct_prediction, len(self.dataset)))
            x_train.cpu()
            test_acc = self.run_test()

            if test_acc > model_best_accuracy:
                self.save_models(epoch)
                model_best_accuracy = test_acc
                patience = 6
            else:
                print("Model accuracy did not improve!\n")
                patience -= 1

            if test_acc == 1:
                print('Model cannot improve. Early stop')
                return
            patience = self.adjust_learning_rate(patience)

    def run_test(self):
        with torch.no_grad():
            print("#############################\nRunning validation set:")
            self.model.eval()
            test_accuracy = 0.0
            predicted_label = []
            true_label = []
            # cambio dataset_validation dal Dataset
            self.dataset.change_data_set("validation")
            print("Changing validation")
            for batch_idx, (images, label, name) in enumerate(self.testing_generator):

                x_test = images.to(self.device)
                y_test = label.long().to(self.device)
                out = self.model(x_test, torch.zeros(1).cuda())
                out = out.cpu()
                predicted_class, confidence, pos_x, pos_y = out[:, 1], out[:, 0], out[:, 2], out[:, 3]
                predicted_label.extend(predicted_class.cpu())
                true_label.extend(y_test.cpu())
                test_accuracy += torch.sum(predicted_class == y_test.float().cpu())
            x_test.cpu()
            correct_prediction = test_accuracy.item()

            test_accuracy = correct_prediction / len(self.dataset)
            if test_accuracy > .9:
                mismatch = conf_matrix(predicted_label, true_label, ["Good", "Bad"], path=self.work_directory_path)
            print("\nTest Accuracy = [%f], Correctly predicted: [%d/%d]" % (test_accuracy, correct_prediction, len(self.dataset)))

            print("\nTest finish\n#############################")
            return test_accuracy

    def save_models(self, epoch):

        self.model.eval()
        with torch.no_grad():
            torch.save(self.model.state_dict(), os.path.join(self.work_directory_path, "model_dict_{}.pt".format(epoch)))
            torch.save(self.model, os.path.join(self.work_directory_path, "model_{}.pt".format(epoch)))
            torch.save(self.model, os.path.join(self.work_directory_path, "best_model.pt"))

            traced_model = torch.jit.trace(self.model.cpu(), (torch.randint(0, 255, (1, self.input_shape[0], self.input_shape[1], self.input_shape[2])).cpu(), torch.zeros(1).cpu()))
            traced_model.save(os.path.join(self.work_directory_path, "best_model_jit.pt"))

            print("Test accuracy improved: Saving model to model_{}.pt".format(epoch))

    def adjust_learning_rate(self, patience):
        for param_group in self.optimizer.param_groups:
            lr = param_group["lr"]
        if lr < .000000001:
            patience = 99
        if patience == 0:
            print("Lr Reduce callback due to no improvement")
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr * 0.5
                print("Lr: "+str(lr*0.5))
            return 5
        return patience

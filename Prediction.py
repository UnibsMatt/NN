import shutil
from TorchTest.BatchLoader import BatchLoader
from torch.utils.data import DataLoader
from TorchTest.Utilities import *
import torch


class Prediction:
    def __init__(self, model, path, input_shape=(470, 470, 3), output_division=True):

        self.path = path
        self.validation_root_folder = "PredictionResult"
        self.work_directory_path = create_work_folder(self.validation_root_folder, path)

        self.input_shape = np.asarray(input_shape)

        self.dataLoaderTest = BatchLoader(path, input_shape, prediction=True)

        self.generator = DataLoader(self.dataLoaderTest, batch_size=12, shuffle=False)

        self.classes_name = ["Reject", "Good"]
        self.device = torch.device("cuda")
        self.model = model.to(self.device)
        self.output_division = output_division

        if self.output_division:
            make_output_directory(self.work_directory_path, self.classes_name)

        self.run()

    def run(self):
        with torch.no_grad():
            self.model.eval()
            i = 0
            total_label = []
            total_image = []
            for batch_idx, (images, _, name) in enumerate(self.generator):
                total_image.append(images)
                x_test = images.to(self.device)

                #out = self.model(x_test, torch.zeros(1))
                out = self.model(x_test)
                predicted_class, confidence, pos_x, pos_y = out[:, 1], out[:, 0], out[:, 2], out[:, 3]
                #predicted_class, confidence, pos_x, pos_y = out[1], out[0], out[2], out[3]
                #print(predicted_class.cpu().numpy())
                #print(confidence.cpu().numpy())
                # # modelli orobix
                # out = self.model(x_test)
                # predicted_class, confidence, pos_x, pos_y = out[:, 1], out[:, 0], out[:, 2], out[:, 3]
                # ####################################################

                confidence = confidence.cpu().numpy()
                predicted_class = predicted_class.cpu().numpy().astype(int)

                total_label.extend(predicted_class)

                if self.output_division:
                    self.move_result(name, predicted_class)
                self.save_result(name, confidence, predicted_class, rename=False)

    def move_result(self, path_list, result_list):

        for index, name in enumerate(path_list):
            shutil.copy(name, os.path.join(self.work_directory_path, self.classes_name[result_list[index]]))

    def save_result(self, name_list, confidence, class_inf, rename=False):

        with open(self.work_directory_path+"/result_res.txt", 'a') as file:
            for i in range(len(name_list)):
                if rename:
                    file_name, extension = name_list[i].split('.')
                    class_name = self.classes_name[int(class_inf.data[i])]
                    os.rename(name_list[i], file_name+"-"+class_name+"-"+str(int(confidence[i]*100))+"."+extension)
                file.write(name_list[i])
                file.write("\t\tConfidence: "+str(confidence[i]))
                file.write("\t\tClass: "+self.classes_name[int(class_inf[i])]+"\n")


def show_heatmap(raw_heatmap, original, predicted):
    heatmap = np.asarray(raw_heatmap.data.cpu()).squeeze(axis=0)
    heatmap = heatmap[predicted].astype("float64")
    heatmap = heatmap[0]

    original_img = np.asarray(original.data.cpu()*255, dtype=np.uint8).squeeze(axis=0).transpose((1, 2, 0))

    original_img = Image.fromarray(original_img, mode="RGB")

    heatmap = (heatmap - np.min(heatmap)) / np.ptp(heatmap)
    img = Image.fromarray(heatmap, "L")
    img = img.resize((512, 512), Image.BICUBIC)

    plt.imshow(original_img)
    plt.imshow(img, cmap="jet", alpha=.25)

    plt.show()


def make_output_directory(path, classes):
    for index, name in enumerate(classes):
        if not os.path.exists(os.path.join(path, name)):
            os.makedirs(os.path.join(path, name))


from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn import svm
from PIL import Image
from time import strftime
import h5py
import time
import torch
import os


class MyModel(nn.Module):
    def __init__(self, original_model, cutted_model, upsampling_dim):
        super(MyModel, self).__init__()
        self.model = original_model.cuda()
        self.model.eval()
        self.heatmap = cutted_model.cuda()
        self.heatmap.eval()
        self.upsampling = nn.UpsamplingBilinear2d(size=(512, 512))

    def forward(self, x):
        data = self.heatmap(x)
        x2 = self.upsampling(data)
        conv = nn.Conv2d(x2.shape[1], 2, kernel_size=1, bias=False)
        conv = conv.cuda()
        conv.weight = nn.Parameter(torch.ones_like(conv.weight))

        x2 = conv(x2)
        x2 = 255*(x2-x2.min())/(x2.max()-x2.min())
        x2 = x2.int()
        x1 = self.model(x)
        return x1, x2


def conf_matrix(y_pred, y_true, label=None, show=True, path="conf_matrix.png"):
    '''
    :param y_pred: y_predizione encoded [0, 0 ,0 , 1 ]
    :param y_true: y_reali encoded [0, 0 ,1, 0]
    :param label: nome delle classi da mostrare nella matrice
    :param show: bool determina se mostrare o meno la matrice
    :param path: percorso del salvataggio della conf matrix
    :return: ritorna gli index dei mismatch della matrice di confusione
    '''
    # TODO: gestire la non istanza di numpy se necessario
    # if not isinstance(y_pred, np.ndarray):
    #    print("ASD")
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    n_classes = len(np.unique(np.concatenate((y_pred, y_true))))

    mismatch = np.where(y_pred-y_true != 0)
    mismatch = np.asarray(mismatch)

    if label is not None:
        assert len(label) == n_classes, "The number of label don't match the classes"

    # creazione della matrice di confusione
    cm = confusion_matrix(y_true, y_pred)
    samples = cm.sum()
    true_pos = np.trace(cm)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=.5, ax=ax)

    ax.set_xlabel('Predicted labels\nAccuracy: {}%'.format((true_pos/samples)*100))
    ax.set_ylabel('True labels')
    ax.set_xticklabels(label)
    ax.set_yticklabels(label)
    ax.set_title('Confusion Matrix')
    plt.xlim(0, n_classes)
    plt.ylim(0, n_classes)
    ax.invert_yaxis()
    plt.tight_layout()

    plt.savefig(path+"/confusion_matrix.png")
    if show:
        plt.show()
    return mismatch


def append_heatmap(model, input_size, cut_level):

    model_copy = type(model)()
    model_copy.load_state_dict(model.state_dict())
    modules = list(model_copy.children())
    cutted_model = nn.Sequential(*modules[:-cut_level])
    myModel = MyModel(model, cutted_model, input_size)
    return myModel.eval()


def plot_result(result, label, centroidi):

    assert len(result) == len(label), "Input and label length don't match"
    label = np.asarray(label)
    result = np.asarray(result)
    class_index = np.unique(label).astype(int)
    svc = svm.SVC(kernel="linear").fit(result, label)
    figure = plt.figure()
    ax = figure.add_subplot()
    xx, yy = make_meshgrid(result[:, ], result[:, 1])
    plot_contours(ax, svc, xx, yy, alpha=.4)
    for idx in class_index:
        same_class = np.where(label == idx)
        ax.scatter(result[same_class, 0], result[same_class, 1], label='class' + str(idx))
        ax.scatter(centroidi[idx, 0], centroidi[idx, 1], label='Center class' + str(idx), marker='^')

    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def average_center(array):
    array = np.asarray(array)
    assert array.shape[1] == 2, "Lo spazio deve essere bidimensionale"
    a, b = array[0]
    return np.sqrt(np.sum((a - b) ** 2, axis=1))


def readImage(path, resize=(470, 470), flip=False):
    """
    Read image
    """
    x = Image.open(path)

    x = x.resize(resize, Image.BILINEAR)
    if flip:
        x = np.flipud(x)
    x = np.asarray(x)
    return x


def euclidean_distance_numpy(x, y):
    x = np.expand_dims(x, 1).repeat(len(y), axis=1)
    y = np.expand_dims(y, 0).repeat(len(x), axis=0)

    start = time.time()

    res = ((x - y) ** 2).sum(axis=2)

    elapsed = time.time()-start
    #print(elapsed)
    return res# elem x numero centri


def safe_split(split):
    if (split <= 1) and (split >= 0):
        print("Lo split vale " + str(split))
    else:
        print("Lo split deve essere compreso tra [0,1]\nImpostato split di default .8")
        split = .8
    return split


def data_split(data, split):
    data = np.asarray(data)
    num_samples = len(data)
    num_split = int(num_samples*split)
    permutation = np.random.permutation(np.arange(len(data)))
    data = data[permutation]
    return data[:num_split], data[num_split:]


def to_tensor(input_image):
    input_tensor = torch.from_numpy(np.asarray(input_image))
    input_tensor = input_tensor.float()
    #input_tensor = input_tensor/255
    return input_tensor


def save_mismatch(path, name_list, confidence, class_name, class_prediction):
    name_list = np.asarray(name_list)
    confidence = np.asarray(confidence)
    class_name = np.asarray(class_name)
    class_prediction = np.asarray(class_prediction, dtype=int)
    with open(path+"/mismatches_result_res.txt", 'a') as file:
        for i in range(len(name_list)):
            file.write(str(name_list[i]))
            file.write("\t\tConfidence: "+str(confidence[i]))
            file.write("\t\tClass: "+str(class_name[class_prediction[i]])+"\n")


def file_exist(file):
    if os.path.isfile(file):
        return True
    return False


def dir_exist(dir):
    if os.path.isdir(dir):
        return True
    return False


def load_model(model_path):
    try:
        model = torch.jit.load(model_path)
    except:
        print("No torch.jit model found")
        try:
            print("Found torch model")
            model = torch.load(model_path)
        except:
            raise Exception("Nessun modello trovato")

    return model


def true_or_false(state):
    if state in (True, False):
        return True
    return False


def create_work_folder(root_folder, img_path):
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    current_training_folder = img_path + "_" + strftime("%H_%M_%S", time.gmtime())
    result_path = os.path.join(root_folder, current_training_folder)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    return os.path.join(root_folder, current_training_folder)


def swap_axis_input(input_shape):
    input_shape = np.asarray(input_shape)
    if input_shape[0] == 3:
        input_shape[0] = input_shape[2]
        input_shape[2] = 3
    return input_shape


def convert_image_list_to_h5(image_list, image_shape, file_name="data.h5py", compression=0):
    if file_exist(file_name):
        print("Dataset already converted")
        return
    image_shape_list = np.append(np.asarray(image_shape), 3)
    image_shape_list = image_shape_list.tolist()
    for index, image_name in enumerate(image_list):
        print("\rConverting: {index}/{length}".format(index=index+1, length=len(image_list)), end="")

        img = Image.open(image_name)
        img = img.resize(image_shape)
        img = np.asarray(img)

        with h5py.File(file_name, 'a') as hf:
            hf.create_dataset(
                name=image_name,
                data=img,
                shape=image_shape_list,
                compression="gzip",
                compression_opts=compression)
    print("Conversion done")

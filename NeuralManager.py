#import DirectorySystem
import argparse
from TorchTest.TrainingAugmented import *
from TorchTest.Utilities import file_exist, dir_exist, load_model, true_or_false, swap_axis_input
from TorchTest.CNN import *
from TorchTest.Validation import Validation
from TorchTest.Prediction import Prediction


def run_system():
    parser = argparse.ArgumentParser(
        description="NeuralManager permette l'addestramento o la validazione di un modello "
                    "rispetto ad un dataset")

    parser.add_argument("action", nargs=1, help="'training', 'prediction' or 'validation'")
    parser.add_argument("--model_path", nargs="?", help="Path to model")
    parser.add_argument("--train_split", nargs="?", default=.75, type=float, help="training split")
    parser.add_argument("--augmentation", nargs="?", default=False, type=bool, help="Apply data augmentation")
    parser.add_argument("--input_shape", nargs=3, default=(470, 470, 3), type=int, help="Input size tuple")
    parser.add_argument("file_folder", nargs=1, help="path to image folder")
    args = parser.parse_args()

    model = args.model_path
    action = args.action[0]
    file_folder = args.file_folder[0]
    train_split = args.train_split
    augmentation = args.augmentation

    input_shape = tuple(args.input_shape)
    input_shape = swap_axis_input(input_shape)

    if action not in ["training", "validation", "prediction"]:
        print("Action non presente. Le opzioni presenti sono 'training', 'validation' o 'prediction'")
        return

    if not dir_exist(file_folder):
        print("Il percorso specificato non esiste. Path: " + file_folder)
        return

    if action != "training":
        if not file_exist(model):
            print("Il modello non esiste. Modello: " + model)
            return

    if not true_or_false(augmentation):
        print("Augmentation deve avere un argomento. Augmentation: " + augmentation)
        return

    if action == "training":
        print("Lanching training process\n")
        print("Training on: /" + file_folder)
        print("Augmentation: "+str(augmentation))

        running_model = CNNPreProcessed(input_shape=(300, 300, 3))

        TrainingA(running_model, file_folder, input_shape, split=train_split, augmentation=augmentation)

    if action == "validation":

        print("Lanching validation process\n")
        print("Validation on: /" + file_folder)
        running_model = load_model(model)
        Validation(running_model, file_folder, input_shape)

    if action == "prediction":
        print("Lanching prediction process\n")
        print("Prediction on: /" + file_folder)
        running_model = load_model(model)
        Prediction(running_model, file_folder, input_shape)


if __name__ == '__main__':
    os.chdir("TorchTest")
    run_system()

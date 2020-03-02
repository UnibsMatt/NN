import torch
import numpy as np
import random


class ToTens(object):
    def __call__(self, pic):
        pic = torch.from_numpy(np.asarray(pic))
        pic = pic.float()
        return pic


class RandomEraseSquare(object):
    def __init__(self, probability=.5):
        noise = random.randint(0, 255)
        self.noise = noise / 255
        self.probability = probability

    def __call__(self, pic):
        if (random.randint(0, 10)/10) > self.probability:
            top_x = random.randint(0, 400)
            top_y = random.randint(0, 400)
            lato = 25

            pic[top_x:top_x+lato, top_y:top_y+lato, :] = self.noise
        return pic

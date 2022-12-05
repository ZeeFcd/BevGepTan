import numpy as np
from keras.utils import load_img, img_to_array
import random
import os


def input_target_split(train_dir, labels):
    dataset = []
    count = 0
    for label in labels:
        folder = os.path.join(train_dir, label)
        for i, image in enumerate(os.listdir(folder)):
            img = load_img(os.path.join(folder, image), target_size=(150, 150))
            img = img_to_array(img)
            img = img / 255.0
            dataset.append((img, count))
        print(f'\rCompleted: {label}')
        count += 1
    random.shuffle(dataset)
    X, y = zip(*dataset)

    return np.array(X), np.array(y)

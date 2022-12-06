import numpy as np
import tensorflow as tf
from keras.utils import load_img, img_to_array
import random
import os


def input_target_split(train_dir, labels):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.5),
        tf.keras.layers.experimental.preprocessing.RandomContrast(0.5)
    ])

    dataset = []
    count = 0
    for label in labels:
        folder = os.path.join(train_dir, label)
        for i, image in enumerate(os.listdir(folder)):
            img = load_img(os.path.join(folder, image), target_size=(150, 150))
            img = img_to_array(img)
            img = img / 255.0
            dataset.append((img, count))
            for j in range(1, 3):
                aug_img = data_augmentation(img)
                dataset.append((aug_img.numpy(), count))
        print(f'\rCompleted: {label}')
        count += 1
    random.shuffle(dataset)
    X, y = zip(*dataset)

    return np.array(X), np.array(y)

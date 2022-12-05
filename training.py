import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import ELU, PReLU, LeakyReLU
from fileread import input_target_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


labels = ['paper', 'scissors', 'rock']
X, y = input_target_split('images', labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42)
print('check shapes: ', X_train.shape, y_train.shape, X_test.shape, y_test.shape)

train_lab_categorical = tf.keras.utils.to_categorical(y_train, num_classes=3, dtype='uint8')
test_lab_categorical = tf.keras.utils.to_categorical(y_test, num_classes=3, dtype='uint8')

train_im, valid_im, train_lab, valid_lab = train_test_split(X_train, train_lab_categorical, test_size=0.20,
                                                            stratify=train_lab_categorical,
                                                            random_state=40, shuffle=True)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=3, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

train_dropout = model.fit(train_im, train_lab, batch_size=32,epochs=5,verbose=1,validation_data=(valid_im, valid_lab))





# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

print()


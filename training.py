import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import ReLU
from fileread import input_target_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


labels = ['paper', 'scissors', 'rock']
X, y = input_target_split('images', labels)
print('Full dataset shapes: ', X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42)
print('train/test shapes: ', X_train.shape, y_train.shape, X_test.shape, y_test.shape)

train_lab_categorical = tf.keras.utils.to_categorical(y_train, num_classes=3, dtype='uint8')
test_lab_categorical = tf.keras.utils.to_categorical(y_test, num_classes=3, dtype='uint8')

train_im, valid_im, train_lab, valid_lab = train_test_split(X_train, train_lab_categorical, test_size=0.20,
                                                            stratify=train_lab_categorical,
                                                            random_state=40, shuffle=True)

print('train/valid shapes: ', train_im.shape, train_lab.shape, valid_im.shape, valid_lab.shape)
print()
model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(150,150,3)))
model.add(ReLU())
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(32, (3, 3), activation='linear',padding='same'))
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(64, activation='linear'))
model.add(ReLU())
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

train_dropout = model.fit(train_im, train_lab, batch_size=64, epochs=20, verbose=1, validation_data=(valid_im, valid_lab))

test_eval = model.evaluate(X_test, test_lab_categorical, verbose=1)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

accuracy = train_dropout.history['accuracy']
val_accuracy = train_dropout.history['val_accuracy']
loss = train_dropout.history['loss']
val_loss = train_dropout.history['val_loss']
epochs = range(len(accuracy))

plt.figure()
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig("correctness.png")

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig("loss.png")

predicted_classes = model.predict(X_test)
predicted_classes = np.argmax(np.round(predicted_classes), axis=1)
correct = np.where(predicted_classes == y_test)[0]
print("Found correct labels:", len(correct))

incorrect = np.where(predicted_classes != y_test)[0]
print("Found incorrect labels:", len(incorrect))

print('--------------------')
report = classification_report(y_test, predicted_classes, target_names=labels, output_dict=True)
dat = pd.DataFrame.from_dict(report)
dat = dat.drop('support', axis=0)
dat = dat.drop(['accuracy', 'macro avg', 'weighted avg'], axis=1)
print(dat)
dat.to_csv('report.csv')

print('--------------------')
print(test_eval)
with open('eval.csv', 'w') as f:
    f.write(f"{test_eval[0]},{test_eval[1]}")

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)






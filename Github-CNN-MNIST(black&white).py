"""
CNN MODEL
MNIST dataset
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_digits

# number of labels(Y)
category = 10
labels = [0,1,2,3,4,5,6,7,8,9]

# load data and split into train/test set
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# normalize input data
x_train = x_train.reshape(x_train.shape[0], 28,28, 1)
x_test = x_test.reshape(x_test.shape[0], 28,28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# One-hot encoding
y_train2 = tf.keras.utils.to_categorical(y_train, category)
y_test2 = tf.keras.utils.to_categorical(y_test, category)

# built CNN model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                                 padding="same",activation='relu',
                                 input_shape=(28,28,1)))
model.add(tf.keras.layers.Conv2D(filters=40, kernel_size=(2, 2),
                                 padding="same", activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(filters=40, kernel_size=(2, 2),
                                 padding="same", activation='relu'))
model.add(tf.keras.layers.Dropout(rate=0.01))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(units=category,activation=tf.nn.softmax))

model.summary()

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer="adam",metrics=['accuracy'])

# training
history = model.fit(x_train, y_train2,epochs=10)

# save model
with open("model_black_white.json", "w") as json_file:
   json_file.write(model.to_json())
# save weights
model.save_weights("model_black_white.h5")

# test
score = model.evaluate(x_test, y_test2, batch_size=128)
print("loss:",score[0],"  ","accuracy:",score[1])
predict = model.predict(x_test)
print("First Ans:",np.argmax(predict[0]),"Category:",labels[np.argmax(predict[0])])
print("Second Ans:",np.argmax(predict[1]),"Category:",labels[np.argmax(predict[1])])
print("Third Ans:",np.argmax(predict[2]),"Category:",labels[np.argmax(predict[2])])

# first pic
# show how accuracy/loss change throughout process
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('model accuracy')
plt.ylabel('acc & loss')
plt.xlabel('epoch')
plt.legend(['accuracy', 'loss'], loc='upper left')
plt.show()

# second pic
# show predictions in a 9*9 picture
nrow=3
ncol=3
fig, axs = plt.subplots(nrows=nrow,ncols=ncol, figsize=(nrow,ncol))
for row in range(nrow):
    for col in range(ncol):
        i = (row * ncol) + col
        pic = x_test[i].reshape(28,28)
        axs[row,col].imshow(pic)
        ans = np.argmax(predict[i], axis=-1)
        str1 = "prediction:"+str(ans)+" , "+str(labels[ans])
        axs[row,col].set_title(str1)

plt.show()

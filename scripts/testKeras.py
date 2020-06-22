import numpy as np
import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

train_images = mnist.train_images()
train_labels = mnist.train_labels()

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

train_images = train_images.reshape((-1,784))
test_images = test_images.reshape((-1,784))
# print(len(test_images),len(test_images[0]))
# print(len(test_labels))
# print(type(test_labels))
print(test_labels)
print(len(test_labels))
# exit()

model = Sequential([
    Dense(64,activation='relu',input_shape=(784,)),
    Dense(64,activation='relu'),
    Dense(10,activation='softmax'),
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'],)
model.fit(train_images,to_categorical(train_labels),epochs=5,batch_size=28)
model.evaluate(test_images,test_labels,batch_size=28)
# print(train_images.shape)
# print(test_images.shape)
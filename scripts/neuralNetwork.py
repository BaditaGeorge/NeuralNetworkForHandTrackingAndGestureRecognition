import imageProcessor
import numpy as np
import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical


class NeuralNetwork:
    def __init__(self):
        print('created!')

    def createModels(self, path):
        ip = imageProcessor.ImageProcessor()
        ip.loadImage(path)
        ip.resizeImages()
        ip.alterInRange()
        self.processDataTrain(ip.imageArray)
        self.train_labels = np.asarray(ip.labels)
        for i in range(0,len(self.train_labels)):
            if self.train_labels[i] == 'hello':
                self.train_labels[i] = 1
            else:
                self.train_labels[i] = 0
        self.train_data = np.asarray(self.train_data)
        ip = imageProcessor.ImageProcessor()
        ip.loadImage('test')
        ip.resizeImages()
        ip.alterInRange()
        self.processDataTest(ip.imageArray)
        self.test_labels = np.asarray(ip.labels)
        for i in range(0,len(self.test_labels)):
            if self.test_labels[i] == 'hello':
                self.test_labels[i] = 1
            else:
                self.test_labels[i] = 0
        self.test_data = np.asarray(self.test_data)
        # print(self.train_data)
        # print(self.train_labels)
        # print(type(self.train_labels))
        # ip.showAt(163)
        # print(ip.labels[163])

    def processDataTrain(self,imgs):
        self.train_data = []
        for i in range(0,len(imgs)):
            self.train_data.append([])
            pix = imgs[i].load()
            for t in range(0,imgs[i].size[0]):
                for j in range(0,imgs[i].size[1]):
                    self.train_data[i].append(0.3*pix[t,j][0] + 0.59*pix[t,j][1] + 0.11*pix[t,j][2])

    def processDataTest(self,imgs):
        self.test_data = []
        for i in range(0,len(imgs)):
            self.test_data.append([])
            pix = imgs[i].load()
            for t in range(0,imgs[i].size[0]):
                for j in range(0,imgs[i].size[1]):
                    self.test_data[i].append(0.3*pix[t,j][0] + 0.59*pix[t,j][1] + 0.11*pix[t,j][2])

    def trainNetwork(self):
        print(self.train_data.shape)
        print(self.train_labels.shape)
        print(self.test_labels.shape)
        print(self.test_data.shape)
        model = Sequential([
            Dense(64, activation='relu', input_shape=(2500,)),
            Dense(64, activation='relu'),
            Dense(2, activation='softmax'),
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'],)
        model.fit(self.train_data, to_categorical(self.train_labels), epochs=5, batch_size=50)
        print(model.predict(self.test_data[:25]))
        print(self.test_labels[39])
        # results = model.evaluate(self.test_data,self.test_labels,batch_size=50)
        # print(results)
        # 

rn = NeuralNetwork()
rn.createModels('train')
rn.trainNetwork()
# print(len(rn.train_data[0]))

# ip = imageProcessor.ImageProcessor()
# ip.loadImage('./train')
# ip.resizeImages(0,len(ip.imageArray)-1)
# ip.resizeImages(0,10)
# ip.alterPixels()
# cv2.imwrite('picture1.jpg',ip.alterSkinPixels(1))
# ip.resizeImages()
# ip.alterInRange()
# ip.saveImages('poze')
# print(ip.imageArray[11].size[0],ip.imageArray[11].size[1])
# ip.showAt(163)
# ip.showAt(120)

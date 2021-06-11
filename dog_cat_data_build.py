import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import pickle

#load dataset
datasetPath = "/home/henry-etech/tensorflow/dog_cat_dataset/PetImages"
categories = ["Dog", "Cat"]
image_size = 100

#build training data
training_data = []
def create_training_data():
    for category in categories: #load dogs and cats
        
        folderPath = os.path.join(datasetPath,category)
        class_num = categories.index(category)

        for imgPath in tqdm(os.listdir(folderPath)): #list all image path in the folders
            try:
                image = cv2.imread(os.path.join(folderPath,imgPath), cv2.IMREAD_GRAYSCALE)
                resized_image = cv2.resize(image, (image_size, image_size))
                training_data.append([resized_image, class_num])
            except Exception as e:
                pass

create_training_data() #get all training data

random.shuffle(training_data) #shuffle the training data (dogs and cats)

#store the training data
X = []
y = []

for features, label in training_data:
    X.append(features) #pixel value
    y.append(label)    #class number

print(X[0].reshape(-1, image_size, image_size, 1))
X = np.array(X).reshape(-1, image_size, image_size, 1)
# np.array(X).reshape()

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
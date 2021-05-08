#%% importing things
from tensorflow import keras as kr
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from keras.layers.convolutional import Conv2D, MaxPooling2D
import pickle
from keras import load_model

#%% read in history

# load model
model = load_model('model')

# load history
history_file = r'C:\Users\oyina\Documents\src\measurement_principles\modeling-project-oyin_marcos\deep learning\history.pkl'
with open(history_file, 'rb') as pkl_file:
        history = pickle.load(pkl_file)

# load data
data_file = r'C:\Users\oyina\Documents\src\measurement_principles\modeling-project-oyin_marcos\deep learning\data.pkl'
with open(data_file, 'rb') as pkl_file:
        data = pickle.load(pkl_file)


#%% plot history
pd.DataFrame(history.history).plot()
plt.grid(True)
# plt.gca().set_ylim(0, 1)
plt.show()

# model.evaluate(valid_data, valid_class)
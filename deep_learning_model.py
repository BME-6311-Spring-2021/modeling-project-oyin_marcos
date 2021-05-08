"""
Deep Learning Project

Neural Net - Classifier
    - Sample images taken from -
    - Images divided into 3 classes:
        - DAPI-stained cell nuclei
        - Tubulin
        - Actin
"""

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

#%% read in files
# src_dir = "D:\\ThirdPartyPrograms\\Python\\PyCharm\\PyCharmProjects\\ExpDsgnClass\\Week1_22123"
src_dir = r"C:\Users\oyina\Documents\src\measurement_principles\BBBC021_v1_images_Week1_22123\Week1_22123"
img_lst = os.listdir(src_dir)
# src_classifications_file = "D:\\ThirdPartyPrograms\\Python\\PyCharm\\PyCharmProjects\\ExpDsgnClass\\class_matched_data.csv"
src_classifications_file = r'C:\Users\oyina\Documents\src\measurement_principles\modeling-project-oyin_marcos\deep learning\class_matched_data.csv'
src_df = pd.read_csv(src_classifications_file)

#%% create data and labels
class_categories = ["dapi", "tubulin", "actin"]     # [0,1,2]
img_lst.sort()
src_df = src_df.sort_values(by="file_name")
src_df = src_df.loc[:499, :]

class_encoding = []
for row_ind in range(0, src_df.shape[0]):
    curr_class = src_df.loc[row_ind, "class"]
    class_encoding.append([1 if i == curr_class else 0 for i in class_categories])
src_df[class_categories] = np.asarray(class_encoding[:500])

data = []
bit_depth = 16
for filename in img_lst[:501]:
    img_path = os.path.join(src_dir, filename)
    img = Image.open(img_path)
    img_data = np.array(img) / (2**bit_depth - 1)
    data.append(img_data)
src_df["img_data"] = data[:500]

img_height = len(data[0])
img_width = len(data[0][0])

classifications = src_df.loc[:, class_categories]
classifications = np.asarray(classifications)
input_data = np.asarray(src_df["img_data"].to_list())
input_data = np.reshape(input_data, (input_data.shape[0], img_width, img_height, 1))

rand = 628
img_data, test_imgs, class_data, test_class = train_test_split(input_data, classifications,
                                                               random_state=rand, test_size=0.2, train_size=0.8)
train_data, valid_data, train_class, valid_class = train_test_split(img_data, class_data,
                                                                    random_state=rand, test_size=0.15, train_size=0.85)

#%% create model and train
act_fun = "relu"
model = kr.models.Sequential([Conv2D(4, kernel_size=(3,3), activation=act_fun, input_shape=input_data.shape[1:]),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(32, kernel_size=(3,3), activation=act_fun),
            MaxPooling2D(pool_size=(2,2)),
            kr.layers.Flatten(),
            kr.layers.Dense(9, activation=act_fun),
            kr.layers.Dense(len(class_categories), activation="softmax")
            ])

model.summary()
opt = "sgd"
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
history = model.fit(train_data, train_class, epochs=50)

# save model
model_file = r'C:\Users\oyina\Documents\src\measurement_principles\modeling-project-oyin_marcos\deep learning\model'
model.save(model_file)

# save history
history_file = r'C:\Users\oyina\Documents\src\measurement_principles\modeling-project-oyin_marcos\deep learning\history.pkl'
with open(history_file, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

# save data
train = (train_data, train_class)
val = (valid_data, valid_class)
test = (test_imgs, test_class)
data = [train, val, test]
data_file = r'C:\Users\oyina\Documents\src\measurement_principles\modeling-project-oyin_marcos\deep learning\data.pkl'
with open(data_file, 'wb') as file_pi:
        pickle.dump(data, file_pi)

#%% plot model eval
# pd.DataFrame(history.history).plot()
# plt.grid(True)
# # plt.gca().set_ylim(0, 1)
# plt.show()

# model.evaluate(valid_data, valid_class)
# %%

# %% importing things
import os, sys
import numpy as np
import pandas as pd

#%% read in directory as list
file_dir = r"C:\Users\oyina\Documents\src\measurement_principles\BBBC021_v1_images_Week1_22123\Week1_22123"
file_names = os.listdir(file_dir)

#%% read in csv as dataframe
raw_data_dir = r"C:\Users\oyina\Documents\src\measurement_principles\modeling-project-oyin_marcos\deep learning\BBBC021_v1_image.csv"
raw_data = pd.read_csv(raw_data_dir)

#%% get dapi, tubulin, and actin columns
dapi = raw_data["Image_FileName_DAPI"].values
tubulin = raw_data["Image_FileName_Tubulin"].values	
actin = raw_data["Image_FileName_Actin"].values

#%% find files names in dapi, tubulin, and actin
in_dapi = list(map(lambda file_name: file_name in dapi, file_names))
in_tubulin = list(map(lambda file_name: file_name in tubulin, file_names))
in_actin = list(map(lambda file_name: file_name in actin, file_names))

# %% match files to names
file_class = np.zeros((len(file_names)), dtype=object)
file_class[in_dapi] = "dapi"
file_class[in_tubulin] = "tubulin"
file_class[in_actin] = "actin"

# %% read out to csv
class_matched_data = pd.DataFrame()
class_matched_data["file_name"] = file_names
class_matched_data["class"] = file_class

file_name = r'C:\Users\oyina\Documents\src\measurement_principles\modeling-project-oyin_marcos\deep learning\class_matched_data.csv'
class_matched_data.to_csv(file_name, index=False)

#%%
matched = r"C:\Users\oyina\Documents\src\measurement_principles\modeling-project-oyin_marcos\deep learning\class_matched_data.csv"
classes_df = pd.read_csv(matched)

#%% 
num_examples = 12 # number of test files
predictions = np.random.rand(num_examples,3) # output from predict_generator()

maxes = np.argmax(predictions, axis=1) # get model prediction by finding max
num_to_str_dict = {0: "dapi", 1: "tubulin", 2: "actin"} # dictonary match up number with stain
maxes_to_str = list(map(num_to_str_dict.get, maxes)) # stain for each prediction

file_names = os.listdir(file_dir) # get list of test files names
accurate = np.zeros((num_examples)) # holding whether prediction is right
for idx, file_name in enumerate(file_names):
    print(idx)
    actual_val = classes_df[classes_df["file_name"]==file_name]["class"].values[0]
    predicted_val = maxes_to_str[idx] # replace df with matched class dataframe
    accurate[idx] = (actual_val == predicted_val)

test_accuracy = np.sum(accurate)/num_examples

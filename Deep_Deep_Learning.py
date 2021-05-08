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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import load_model
from tensorflow import keras as kr
import matplotlib.pyplot as plt
import pandas as pd
import os

#%% functions
class GlobalVar:
    """
    This is just a placeholder object for commonly-needed items.
    """

    def __init__(self):
        self.classes = ["dapi", "actin", "tubulin"]
        self.train_dir = "C:\\Users\\marcos\\Desktop\\deepLearning\\train"
        self.valid_dir = "C:\\Users\\marcos\\Desktop\\deepLearning\\valid"
        self.test_dir = "C:\\Users\\marcos\\Desktop\\deepLearning\\test"
        self.scf = "C:\\Users\\marcos\\Desktop\\deepLearning\\class_matched_data.csv"
        self.histo = "C:\\Users\\marcos\\Desktop\\deepLearning\\histo.tf"
        self.bit_depth = 16
        self.height = 1024
        self.width = 1280
        self.num_epochs = 15
        self.batch = 15
        self.rscale = 1. / (2 ** self.bit_depth - 1)


def format_source(glob):
    src_df = pd.read_csv(glob.scf)
    src_dict = {}
    for i in range(0, len(src_df)):
        key = src_df.loc[i, "file_name"]
        src_dict[key] = src_df.loc[i, "class"]
    train_lst = os.listdir(glob.train_dir)
    valid_lst = os.listdir(glob.valid_dir)

    for category in glob.classes:
        os.mkdir(os.path.join(glob.train_dir, category))
        os.mkdir(os.path.join(glob.valid_dir, category))
    for i in range(0, len(train_lst)):
        filename = train_lst[i]
        class_label = src_dict[filename]
        print(filename, class_label)
        src_path = os.path.join(glob.train_dir, filename)
        dst_path = os.path.join(glob.train_dir, class_label, filename)
        os.rename(src_path, dst_path)
    for i in range(0, len(valid_lst)):
        filename = valid_lst[i]
        class_label = src_dict[filename]
        src_path = os.path.join(glob.valid_dir, filename)
        dst_path = os.path.join(glob.valid_dir, class_label, filename)
        os.rename(src_path, dst_path)


def build_generators(glob):
    train_aug = ImageDataGenerator(rescale=glob.rscale, rotation_range=30, zoom_range=0.5,
                                   width_shift_range=0.25, height_shift_range=0.3, shear_range=0.2,
                                   horizontal_flip=True, fill_mode="nearest"
                                   )
    valid_aug = ImageDataGenerator(rescale=glob.rscale, rotation_range=20, zoom_range=0.1, shear_range=0.05)
    test_aug = ImageDataGenerator(rescale=glob.rscale)

    train_gen = train_aug.flow_from_directory(glob.train_dir, target_size=(glob.height, glob.width), batch_size=glob.batch,
                                              class_mode="categorical", shuffle=True, color_mode="grayscale"
                                              )
    valid_gen = valid_aug.flow_from_directory(glob.valid_dir, target_size=(glob.height, glob.width), batch_size=glob.batch,
                                              class_mode="categorical", shuffle=True, color_mode="grayscale"
                                              )
    test_gen = test_aug.flow_from_directory(glob.test_dir, target_size=(glob.height, glob.width), batch_size=glob.batch,
                                            shuffle=True, color_mode="grayscale", class_mode="categorical"
                                            )
    return (train_gen, valid_gen, test_gen)


def build_model(glob):
    neural_net = kr.models.Sequential([kr.layers.InputLayer(input_shape=(glob.height, glob.width, 1)),
                Conv2D(8, kernel_size=(3,3), activation="tanh"),
                MaxPooling2D(pool_size=(2,2)),
                Conv2D(16, kernel_size=(3,3), activation="relu"),
                MaxPooling2D(pool_size=(2,2)),
                Conv2D(32, kernel_size=(3,3), activation="relu"),
                MaxPooling2D(pool_size=(2,2)),
                kr.layers.Flatten(),
                kr.layers.Dense(100, activation="tanh"),
                kr.layers.Dense(60, activation="relu"),
                kr.layers.Dense(30, activation="relu"),
                kr.layers.Dense(6, activation="relu"),
                kr.layers.Dense(len(glob.classes), activation="softmax")
                ])
    return neural_net


def train_model(neural_net, training_generator, validation_generator, glob):
    neural_net.summary()
    opt = "adam"
    neural_net.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    history = neural_net.fit(training_generator, steps_per_epoch=training_generator.n//glob.batch, epochs=glob.num_epochs,
                             validation_steps=validation_generator.n//glob.batch, validation_data=validation_generator
                             )
    histodf = pd.DataFrame(history.history)
    neural_net.save(glob.histo)
    histodf.plot()
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


def test_model(glob, testing_generator):
    model = load_model(glob.histo)
    model.summary()
    prediction = model.predict_generator(testing_generator)
    print(prediction)


def main():
    global_vars = GlobalVar()
    format_source(global_vars)
    generator_tuple = build_generators(global_vars)
    training_gen = generator_tuple[0]
    validation_gen = generator_tuple[1]
    testing_gen = generator_tuple[2]
    model = build_model(global_vars)
    train_model(model, training_gen, validation_gen, global_vars)
    test_model(global_vars, testing_gen)


#%% run
if __name__ == "__main__":
    main()

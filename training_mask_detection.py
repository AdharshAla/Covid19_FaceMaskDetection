# -*- coding: utf-8 -*-

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mv2_preprocess
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Input, Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse
import seaborn as sns


# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', required=True, help="path to input dataset")
parser.add_argument('-o', '--output', required=True, help="path to output each epoch")
parser.add_argument("-p1", "--confusion_matrix", type=str, default="confusion_matrix.png", help="path to output loss plot")
parser.add_argument("-p2", "--accuracy_loss_plot", type=str, default="accuracy_loss_plot.png", help="path to output accuracy plot")
parser.add_argument("-vgg19", "--model1", type=str, default="vgg19_mask_detection.hdf5", help="path to output face mask detector VGG19 model")
parser.add_argument("-mv2", "--model2", type=str, default="mv2_mask_detection.hdf5", help="path to output face mask detector MobileNetV2 model")
args = vars(parser.parse_args())


# Provide the path of all the images and the categories it needs to be categorized
data_directory = args["dataset"]
#data_directory = list(paths.list_images(args["dataset"]))
output_directory = args["output"]
#output_directory = list(paths.list_images(args["output"]))
# Gets the folder names inside the given directory
categories = os.listdir(data_directory)
# Store the categories as labels with 0,1,2
labels = [i for i in range(len(categories))]
# create a dictonary variable and storing category and label information as key-value pair
#label_dict = dict(zip(categories, labels))
label_dict = {}
for c in categories:
    if c == "No Mask":
        label_dict[c] = 0
    elif c == "Wrong Mask":
        label_dict[c] = 1   
    elif c == "Mask":
        label_dict[c] = 2
# Store number of categories based on number of labels
noc = len(labels)
print ("Categories: ", categories)
print ("Labels: ", labels )
print ("Category and its Label information: ", label_dict)

# Number of images in each CATEGORY
for category in categories:
    path = os.path.join(data_directory, category)
    print("Number of images in category " + category + " are " + str(len(os.listdir(path))))

# Converting the image into array and normalizing the image using preprocess block
def preprocessing_image(preprocess_type, directory, category):
  image_size = 224
  mv2_data = []
  mv2_labels = []
  vgg19_data = []
  vgg19_labels = []
  print("[INFO] loading images...")
  if preprocess_type.strip().upper() == "MV2":
    for category in categories:
        path = os.path.join(data_directory, category)
      # load the input image (224*224) and preprocess it
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = load_img(img_path, target_size=(image_size, image_size))
            image = img_to_array(image) # converted to (224,224,3) dimensions
            #print("image: ", image)
            #print("image shape: ", image.shape)
            image = mv2_preprocess(image) # Resizing scaled to -1 to 1 
            #print("preprocessed image: ", image)
            #print("preprocessed image shape: ", image.shape)      
      # update the data and labels lists, respectively
            mv2_data.append(image)
            mv2_labels.append(label_dict[category])
    # Saving the image data and target data using numpy for backup
    np.save(os.path.join(output_directory, "mv2_data"), mv2_data)
    np.save(os.path.join(output_directory, "mv2_labels"), mv2_labels)
  elif preprocess_type.strip().upper() == "VGG19":
    for category in categories:
        path = os.path.join(data_directory, category)
      # load the input image (224*224) and preprocess it
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = load_img(img_path, target_size=(image_size, image_size))
            image = img_to_array(image) # converted to (224,224,3) dimensions
            #print("image: ", image)
            #print("image shape: ", image.shape)
            image = vgg19_preprocess(image) # Resizing scaled to -1 to 1 
            #print("preprocessed image: ", image)
            #print("preprocessed image shape: ", image.shape)      
      # update the data and labels lists, respectively
            vgg19_data.append(image)
            vgg19_labels.append(label_dict[category])
    # Saving the image data and target data using numpy for backup
    np.save(os.path.join(output_directory, "vgg19_data"), vgg19_data)
    np.save(os.path.join(output_directory, "vgg19_labels"), vgg19_labels)
  print("Images loaded and saved Successfully ")
  

# preprocessing the images using Mobilenetv2 and save it
preprocessing_image("MV2", data_directory, categories)

# preprocessing the images using VGG19 and save it
preprocessing_image("VGG19", data_directory, categories)

# loading the saved numpy arrays of mobilenetv2 preprocessed images
mv2_data = np.load(os.path.join(output_directory, "mv2_data.npy"))
mv2_labels = np.load(os.path.join(output_directory, "mv2_labels.npy"))


# Check the data shape and labels after preprocessing of mobilenetv2 image data
print("first 10 target values: ", mv2_labels[1:10])
print("shape of data", mv2_data[0].shape)
print("first image data information in array", mv2_data[0])

# Bar plot to see the count of images in various categories
unique, counts = np.unique(mv2_labels, return_counts=True)
category_count = dict(zip(unique, counts))
temp_df = pd.DataFrame(category_count.values(), columns = ['Count'])
temp_df['Categories'] = pd.DataFrame(categories)
plt.barh(temp_df.Categories, temp_df.Count, color='rgbkymc')
plt.ylabel("Various Categories")
plt.xlabel("Count")
plt.title("Bar plot to see the count of images based on categories")
for index, value in enumerate(temp_df.Count):
    plt.text(value, index, str(value))
plt.show()

# convert output array of labeled data(from 0 to nb_classes - 1) to one-hot vector
mv2_labels = to_categorical(mv2_labels)
print ("Shape of mv2 data: ", mv2_data.shape)
print ("Shape of mv2 labels: ", mv2_labels.shape)

#pip install keras-tuner

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=10,
	zoom_range=0.05,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.1,
	horizontal_flip=True,
	fill_mode="nearest")

# Build Hyper Tunning model for MobileNetV2
def mv2_build_model(hp):
    baseModel = MobileNetV2(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))
    headModel = baseModel.output
    headModel = keras.layers.AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = keras.layers.Flatten()(headModel)
    headModel = keras.layers.Dense(units = hp.Int(name = 'dense_units', min_value=64, max_value=256, step=16), activation = 'relu')(headModel)
    headModel = keras.layers.Dropout(hp.Float(name = 'dropout', min_value = 0.1, max_value = 0.5, step=0.1, default=0.5))(headModel)
    headModel = keras.layers.Dense(3, activation = 'softmax')(headModel)
    model = Model(inputs=baseModel.input, outputs=headModel)
    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    for layer in baseModel.layers:
      layer.trainable = False
    model.compile(optimizer = keras.optimizers.Adam(hp.Choice('learning_rate', values = [1e-2,5e-3,1e-3,5e-4,1e-4])),
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])
    return model

# Do Randomsearch for 50 combinations
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
# Stratified Split the data into 20% Test and 80% Traininng data
X_train,X_test,y_train, y_test = train_test_split(mv2_data, mv2_labels, test_size = 0.2, stratify = mv2_labels, random_state = 42)
print ("Search for best model fit for mobilenetv2...")
mv2_tuner_search = RandomSearch(mv2_build_model, objective = 'val_accuracy', max_trials =30, directory = output_directory, project_name = "MobileNetV2")
mv2_tuner_search.search(X_train, y_train, epochs = 3, validation_split = 0.2)
# Show a summary of the hyper parameter search output for 100 combination parameters
mv2_tuner_search.results_summary()
mv2_model = mv2_tuner_search.get_best_models(num_models=1)[0]
mv2_model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# Set callback functions to early stop training and save the best model so far
callbacks = [EarlyStopping(monitor='val_loss', patience=10),
             ModelCheckpoint(filepath=args["model2"], monitor='val_loss', save_best_only=True)]

# Train neural network
mv2_history = mv2_model.fit(aug.flow(X_train, y_train, batch_size=32),
                      epochs=30, # Number of epochs
                      callbacks=callbacks, # Early stopping
                      verbose=2, # Print description after each epoch
                      validation_data=(X_test, y_test)) # Data for evaluation

# loading the MobileNetV2 model back
mv2_model = keras.models.load_model(args["model2"])

np.save(os.path.join(output_directory, 'mv2_history.npy'),mv2_history.history)
# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(mv2_history.history) 
# or save to csv: 
hist_csv_file = os.path.join(output_directory, 'mv2_history.csv')
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

# loading the saved numpy arrays of vgg19 preprocessed images
vgg19_data = np.load(os.path.join(output_directory, "vgg19_data.npy"))
vgg19_labels = np.load(os.path.join(output_directory, "vgg19_labels.npy"))

# Check the data shape and labels after preprocessing of vgg19 image data
print("first 10 target values: ", vgg19_labels[1:10])
print("shape of data", vgg19_data[0].shape)
print("first image data information in array", vgg19_data[0])

# convert output array of labeled data(from 0 to nb_classes - 1) to one-hot vector
vgg19_labels = to_categorical(vgg19_labels)
print ("Shape of vgg19 data: ", vgg19_data.shape)
print ("Shape of vgg19 labels: ", vgg19_labels.shape)

# Build Hyper Tunning model for VGG19
def vgg19_build_model(hp):
    baseModel = VGG19(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))
    headModel = baseModel.output
    headModel = keras.layers.MaxPooling2D(pool_size=(5, 5))(headModel)
    headModel = keras.layers.Flatten()(headModel)
    headModel = keras.layers.Dense(units = hp.Int(name = 'dense_units', min_value=64, max_value=256, step=16), activation = 'relu')(headModel)
    headModel = keras.layers.Dropout(hp.Float(name = 'dropout', min_value = 0.1, max_value = 0.5, step=0.1, default=0.5))(headModel)
    headModel = keras.layers.Dense(3, activation = 'softmax')(headModel)
    model = Model(inputs=baseModel.input, outputs=headModel)
    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    for layer in baseModel.layers:
      layer.trainable = False
    model.compile(optimizer = keras.optimizers.Adam(hp.Choice('learning_rate', values = [1e-2,5e-3,1e-3,5e-4,1e-4])),
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])
    return model

print ("Search for best model fit for vgg19...")
vgg19_tuner_search = RandomSearch(vgg19_build_model, objective = 'val_accuracy', max_trials =30, directory = output_directory, project_name = "VGGNET19")
vgg19_tuner_search.search(X_train, y_train, epochs = 3, validation_split = 0.2)
# Show a summary of the hyper parameter search output for 100 combination parameters
vgg19_tuner_search.results_summary()
vgg19_model = vgg19_tuner_search.get_best_models(num_models=1)[0]
vgg19_model.summary()

# Set callback functions to early stop training and save the best model so far
callbacks = [EarlyStopping(monitor='val_loss', patience=10),
             ModelCheckpoint(filepath=args["model1"], monitor='val_loss', save_best_only=True)]

# Train neural network
vgg19_history = vgg19_model.fit(aug.flow(X_train, y_train, batch_size=32),
                      epochs=30, # Number of epochs
                      callbacks=callbacks, # Early stopping
                      verbose=2, # Print description after each epoch
                      validation_data=(X_test, y_test)) # Data for evaluation


# loading the VGG19 model back
vgg19_model = keras.models.load_model(args["model1"])

np.save(os.path.join(output_directory, 'vgg19_history.npy'),vgg19_history.history)
# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(vgg19_history.history) 
# or save to csv: 
hist_csv_file = os.path.join(output_directory, 'vgg19_history.csv')
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

# Plot the loss and accuracy for both VGG19 and MobileNetV2 models
mv2_history_csv = pd.read_csv(os.path.join(output_directory, 'mv2_history.csv'))
vgg19_history_csv = pd.read_csv(os.path.join(output_directory, 'vgg19_history.csv'))
#mv2_history_csv.rename(columns={ mv2_history_csv.columns[1]: "epoch" })
#mv2_history_csv.rename({'Unnamed: 0':'epoch'}, axis='columns')
fig = plt.figure(figsize = (15,15))
ax1 = fig.add_subplot(2, 2, 1) # row, column, position
ax2 = fig.add_subplot(2, 2, 2) # row, column, position
ax3 = fig.add_subplot(2, 2, 3) # row, column, position
ax4 = fig.add_subplot(2, 2, 4) # row, column, position
mv2_history_csv.rename( columns={'Unnamed: 0':'epoch'}, inplace=True)
vgg19_history_csv.rename( columns={'Unnamed: 0':'epoch'}, inplace=True)
vgg19_history_csv.plot(x = "epoch", y=['loss','val_loss'], ax=ax1, title = 'Training vs Val loss for VGG19 model')
ax1.set_xlabel("epoch")
ax1.set_ylabel("loss value")
ax1.set_ylim(min(vgg19_history_csv.loss.min(), vgg19_history_csv.val_loss.min()),max(vgg19_history_csv.loss.max(), vgg19_history_csv.val_loss.max()))
mv2_history_csv.plot(x = "epoch", y=['loss','val_loss'], ax=ax2, title = 'Training vs Val loss for MobileNetV2 model')
ax2.set_xlabel("epoch")
ax2.set_ylabel("loss value")
ax2.set_ylim(min(mv2_history_csv.loss.min(), mv2_history_csv.val_loss.min()),max(mv2_history_csv.loss.max(), mv2_history_csv.val_loss.max()))
vgg19_history_csv.plot(x = "epoch", y=['accuracy','val_accuracy'], ax=ax3, title = 'Training vs Val accuracy for VGG19 model')
ax3.set_xlabel("epoch")
ax3.set_ylabel("accuracy value")
ax3.set_ylim(min(vgg19_history_csv.accuracy.min(), vgg19_history_csv.val_accuracy.min()),max(vgg19_history_csv.accuracy.max(), vgg19_history_csv.val_accuracy.max()))
mv2_history_csv.plot(x = "epoch", y=['accuracy','val_accuracy'], ax=ax4, title = 'Training vs Val accuracy for MobileNetV2 model')
ax4.set_xlabel("epoch")
ax4.set_ylabel("accuracy value")
ax4.set_ylim(min(mv2_history_csv.accuracy.min(), mv2_history_csv.val_accuracy.min()),max(mv2_history_csv.accuracy.max(), mv2_history_csv.val_accuracy.max()))
plt.savefig(args["accuracy_loss_plot"])
plt.show()


# make predictions on the testing set using vgg19
print("[INFO] evaluating VGG19 model predicted values...")
vgg19_predIdxs = vgg19_model.predict(X_test, batch_size=32).round()
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
vgg19_predIdxs = np.argmax(vgg19_predIdxs, axis=1)
print("[INFO] evaluating MobileNetV2 model predicted values...")
mv2_predIdxs = mv2_model.predict(X_test, batch_size=32).round()
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
mv2_predIdxs = np.argmax(mv2_predIdxs, axis=1)
print("Difference betwween the predicted values of VGG19 and MobilenetV2 are: " + str(y_test.shape[0]-np.count_nonzero(vgg19_predIdxs == mv2_predIdxs)) + " in "+ str(y_test.shape[0]) + " records")

# Model Prediction for VGG19
vgg19_y_pred = vgg19_model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score: ",accuracy_score(y_test, vgg19_y_pred.round())*100)

from sklearn.metrics import classification_report, confusion_matrix
print("Classification Report: \n", classification_report(vgg19_y_pred.round(), y_test))

# Model Prediction for MobileNetV2
mv2_y_pred = mv2_model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score: ",accuracy_score(y_test, mv2_y_pred.round())*100)

from sklearn.metrics import classification_report, confusion_matrix
print("Classification Report: \n", classification_report(mv2_y_pred.round(), y_test))

# Evaluate VGG19 model
print(vgg19_model.evaluate(X_test, y_test, batch_size = 128))

# Evaluate MobileNetV2 model
print(mv2_model.evaluate(X_test, y_test, batch_size = 128))

# Confusion Matrix - {'No Mask': 0, 'Wrong Mask': 1, 'Mask': 2}
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
fig = plt.figure(figsize = (20,8))
ax1 = fig.add_subplot(1, 2, 1) # row, column, position
ax2 = fig.add_subplot(1, 2, 2) # row, column, position
cm1 =confusion_matrix(y_test.argmax(axis=1), vgg19_y_pred.round().argmax(axis=1))  
index = ['No Mask','Wrong Mask','Mask']  
columns = ['No Mask','Wrong Mask','Mask']  
cm_df1 = pd.DataFrame(cm1,columns,index) 
cm_df1.index.name = "vgg19_Actual"
cm_df1.columns.name = "vgg19_Predicted"
ax1.set_title("VGG19 Model")

cm2 =confusion_matrix(y_test.argmax(axis=1), mv2_y_pred.round().argmax(axis=1))  
index = ['No Mask','Wrong Mask','Mask']  
columns = ['No Mask','Wrong Mask','Mask']  
cm_df2 = pd.DataFrame(cm2,columns,index) 
cm_df2.index.name = "mv2_Actual"
cm_df2.columns.name = "mv2_Predicted"
ax2.set_title("MobileNetV2 Model")
sns.heatmap(cm_df1, annot=True, fmt = ".0f", ax=ax1)
sns.heatmap(cm_df2, annot=True, fmt = ".0f", ax=ax2)
plt.savefig(args["confusion_matrix"])
plt.show()

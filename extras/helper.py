import tensorflow as tf
import matplotlib.image as mpimg



# Convert and image to desired shape and scale it so that it can be passed as input to our Model
def prep_and_load_file(file_location, desired_shape=224, scale=True):
    image = mpimg.imread(file_location)
    image = tf.image.resize(image, [desired_shape, desired_shape])
    if scale:
      return image/255.
    else:
      return image


# img = prep_and_load_file(file_location='/content/flower-image.jpg')
# print(img)

import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


#Plotting Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, categories=2):
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(categories))
    disp.plot(cmap='Blues')
    plt.title('Çonfusion Matrix')
    plt.show()


# Predict and plot the prediction made by a model on a particular image
def pred_and_plot(model, file_location, class_names):
    img = prep_and_load_file(file_location=file_location)

    pred = model.predict(tf.expand_dims(img, axis=0))

    if len(pred[0]) > 1:  #multi-class
        pred_class = class_names[pred.argmax()]
    else: #binary
        pred_class = class_names[int(tf.round(pred)[0][0])]

    plt.imshow(img)
    plt.title(f"Predicted class: {pred_class}")
    plt.axis(False)
    plt.show()



# plot loss and accuracy curves based on history of a model
def plot_loss_curves(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = np.arange(len(loss))

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    plt.plot(epochs, loss, label = 'training loss')
    plt.plot(epochs, val_loss, label = 'validation loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.legends()
    plt.show()

    plt.plot(epochs, accuracy, label = 'Training accuracy')
    plt.plot(epochs, val_accuracy, label = 'Validation accuracy')
    plt.title('Accuracy curves')
    plt.xlabel('Epochs')
    plt.legends()
    plt.show()



import zipfile
import os


# Unzip folder in case the dataset in in the form of Zipped folder
def unzip_folder(folder_path):
    zip_ref = zipfile.ZipFile(folder_path)
    zip_ref.extractall()
    zip_ref.close()



# Check folder structure of a particular folder, mostly dataset folder
def walk_through_directory(dir_path):
    for subdirpath, dirnames, filenames in os.walk(dir_path):
      print(f"{len(dirnames)} directories and {len(filenames)} files in {subdirpath}")

from __future__ import absolute_import, division, print_function, unicode_literals

import time
import numpy as np

import tensorflow as tf#.compat.v2 as tf
tf.enable_v2_behavior()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

from tensorflow.python.client import device_lib
from keras.utils.training_utils import multi_gpu_model

def check_tensor_core_gpu_present():
    local_device_protos = device_lib.list_local_devices()
    for line in local_device_protos:
        if "compute capability" in str(line):
            compute_capability = float(line.physical_device_desc.split("compute capability: ")[-1])
            if compute_capability>=7.0:
                return True

print("TensorFlow version is", tf.__version__)

try:
    # check and assert TensorFlow >= 1.14
    tf_version_list = tf.__version__.split(".")
    if int(tf_version_list[0]) < 2:
        assert int(tf_version_list[1]) >= 14
except:
    print("TensorFlow 1.14.0 or newer is required.")
    
print("Tensor Core GPU Present:", check_tensor_core_gpu_present())
assert check_tensor_core_gpu_present() == True

# The data, split between train and test sets

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

num_classes = np.max(y_train) + 1

# Convert class vectors to binary class matrices

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(height_shift_range=0.05,
                                                          horizontal_flip=True)


def normalize(ndarray):
    ndarray = ndarray.astype("float32")
    ndarray = ndarray/255.0
    return ndarray

x_train = normalize(x_train)
x_test = normalize(x_test)

def create_model(num_classes=10):
    """
    Returns a simple CNN suitable for classifiying images from CIFAR10
    """
    # model parameters
    act = "relu"
    pad = "same"
    ini = "he_uniform"
    
    model = tf.keras.models.Sequential([
        Conv2D(128, (3, 3), activation=act, padding=pad, kernel_initializer=ini,
               input_shape=(32,32,3)),
        Conv2D(256, (3, 3), activation=act, padding=pad, kernel_initializer=ini),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(256, (3, 3), activation=act, padding=pad, kernel_initializer=ini),
        Conv2D(256, (3, 3), activation=act, padding=pad, kernel_initializer=ini),
        Conv2D(256, (3, 3), activation=act, padding=pad, kernel_initializer=ini),
        Conv2D(256, (3, 3), activation=act, padding=pad, kernel_initializer=ini),
        Conv2D(256, (3, 3), activation=act, padding=pad, kernel_initializer=ini),
        Conv2D(256, (3, 3), activation=act, padding=pad, kernel_initializer=ini),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(256, (3, 3), activation=act, padding=pad, kernel_initializer=ini),
        Conv2D(128, (3, 3), activation=act, padding=pad, kernel_initializer=ini),
        MaxPooling2D(pool_size=(4,4)),
        Flatten(),
        BatchNormalization(),
        Dense(512, activation='relu'),
        Dense(num_classes, activation="softmax")
    ])

    return model

model = create_model(num_classes)
model.summary()

# training parameters
BATCH_SIZE = 160
N_EPOCHS = 10
opt = tf.keras.optimizers.SGD(lr=0.003, momentum=0.7)

def train_model(mixed_precision, optimizer):
    """
    Trains a CNN to classify images on CIFAR10,
    and returns the training and classification performance
    
    Args:
        mixed_precision: `True` or `False`
        optimizer: An instance of `tf.keras.optimizers.Optimizer`
    """
    model = create_model(num_classes)
    #model = multi_gpu_model(model, gpus=2)


    if mixed_precision:
        optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])
    
    train_start = time.time()

    train_log = model.fit_generator(datagen.flow(x_train, y_train,
                                                 batch_size=BATCH_SIZE),
                                    epochs=N_EPOCHS,
                                    workers=2, use_multiprocessing=True)

    train_end = time.time()

    score = model.evaluate(x_test, y_test)
    
    results = {"test_loss": score[0],
               "test_acc": score[1],
               "train_time": train_end-train_start,
               "train_log": train_log}
    
    return results

#Train without mixed precision
fp32_results = train_model(mixed_precision=False, optimizer=opt)

test_acc = round(fp32_results["test_acc"]*100, 1)
train_time = round(fp32_results["train_time"], 1)
print(test_acc, "% achieved in", train_time, "seconds")

tf.keras.backend.clear_session()
time.sleep(30)

#Train with mixed precision
mp_results = train_model(mixed_precision=True, optimizer=opt)

test_acc = round(mp_results["test_acc"]*100, 1)
train_time = round(mp_results["train_time"], 1)

print(test_acc, "% achieved in", train_time, "seconds")

###Evaluate performance
##import matplotlib.pyplot as plt
##
##plt.plot(fp32_results["train_log"].history["loss"], label="FP32")
##plt.plot(mp_results["train_log"].history["loss"], label="Mixed Precision")
##plt.title("Performance Comparison")
##plt.ylabel("Training Loss")
##plt.xlabel("Epoch")
##plt.legend()
##plt.show()
##print(test_acc, "% achieved in", train_time, "seconds")

#Get speedup
speed_up = round(100 * fp32_results["train_time"]/mp_results["train_time"], 1)

print("Total speed-up:", speed_up, "%")

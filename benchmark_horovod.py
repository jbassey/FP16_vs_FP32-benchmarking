##from __future__ import absolute_import, division, print_function, unicode_literals
##
##import time
##import numpy as np
##
##import tensorflow as tf#.compat.v2 as tf
###tf.enable_v2_behavior()
##
##from tensorflow.keras.models import Sequential
##from tensorflow.keras.layers import *
##
##from tensorflow.python.client import device_lib
##import horovod.tensorflow.keras as hvd
##
##
##def check_tensor_core_gpu_present():
##    local_device_protos = device_lib.list_local_devices()
##    for line in local_device_protos:
##        if "compute capability" in str(line):
##            compute_capability = float(line.physical_device_desc.split("compute capability: ")[-1])
##            if compute_capability>=7.0:
##                return True
##
##print("TensorFlow version is", tf.__version__)
##
##try:
##    # check and assert TensorFlow >= 1.14
##    tf_version_list = tf.__version__.split(".")
##    if int(tf_version_list[0]) < 2:
##        assert int(tf_version_list[1]) >= 14
##except:
##    print("TensorFlow 1.14.0 or newer is required.")
##    
##print("Tensor Core GPU Present:", check_tensor_core_gpu_present())
##assert check_tensor_core_gpu_present() == True
##
##
##
##hvd.init()
##gpus = tf.config.experimental.list_physical_devices('GPU')
##for gpu in gpus:
##    tf.config.experimental.set_memory_growth(gpu, True)
##if gpus:
##    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
##    
### The data, split between train and test sets
##
##(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
##
##num_classes = np.max(y_train) + 1
##
### Convert class vectors to binary class matrices
##
##y_train = tf.keras.utils.to_categorical(y_train, num_classes)
##y_test = tf.keras.utils.to_categorical(y_test, num_classes)
##
##datagen = tf.keras.preprocessing.image.ImageDataGenerator(height_shift_range=0.05,
##                                                          horizontal_flip=True)
##
##
##def normalize(ndarray):
##    ndarray = ndarray.astype("float32")
##    ndarray = ndarray/255.0
##    return ndarray
##
##x_train = normalize(x_train)
##x_test = normalize(x_test)
##
##def create_model(num_classes=10):
##    """
##    Returns a simple CNN suitable for classifiying images from CIFAR10
##    """
##    # model parameters
##    act = "relu"
##    pad = "same"
##    ini = "he_uniform"
##    
##    model = tf.keras.models.Sequential([
##        Conv2D(128, (3, 3), activation=act, padding=pad, kernel_initializer=ini,
##               input_shape=(32,32,3)),
##        Conv2D(256, (3, 3), activation=act, padding=pad, kernel_initializer=ini),
##        MaxPooling2D(pool_size=(2,2)),
##        Conv2D(256, (3, 3), activation=act, padding=pad, kernel_initializer=ini),
##        Conv2D(256, (3, 3), activation=act, padding=pad, kernel_initializer=ini),
##        Conv2D(256, (3, 3), activation=act, padding=pad, kernel_initializer=ini),
##        Conv2D(256, (3, 3), activation=act, padding=pad, kernel_initializer=ini),
##        Conv2D(256, (3, 3), activation=act, padding=pad, kernel_initializer=ini),
##        Conv2D(256, (3, 3), activation=act, padding=pad, kernel_initializer=ini),
##        MaxPooling2D(pool_size=(2,2)),
##        Conv2D(256, (3, 3), activation=act, padding=pad, kernel_initializer=ini),
##        Conv2D(128, (3, 3), activation=act, padding=pad, kernel_initializer=ini),
##        MaxPooling2D(pool_size=(4,4)),
##        Flatten(),
##        BatchNormalization(),
##        Dense(512, activation='relu'),
##        Dense(num_classes, activation="softmax")
##    ])
##
##    return model
##
##model = create_model(num_classes)
##model.summary()
##
### training parameters
##BATCH_SIZE = 160
##N_EPOCHS = 10
##opt = tf.optimizers.Adam(0.003*hvd.size())#lr=0.003, momentum=0.7)#
###opt = hvd.DistributedOptimizer(opt)
####opt = hvd.DistributedOptimizer(opt)#,compression=compression
##
##def train_model(mixed_precision, optimizer):
##    """
##    Trains a CNN to classify images on CIFAR10,
##    and returns the training and classification performance
##    
##    Args:
##        mixed_precision: `True` or `False`
##        optimizer: An instance of `tf.keras.optimizers.Optimizer`
##    """
##    model = create_model(num_classes)
##
##    if mixed_precision:
##        #optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
##        optimizer = hvd.DistributedOptimizer(optimizer,compression=hvd.Compression.fp16)
##    else:
##        optimizer = hvd.DistributedOptimizer(optimizer)
##        
##    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),#"categorical_crossentropy",
##                  optimizer=optimizer,
##                  metrics=["accuracy"])
##    
##    verbose = 1 if hvd.rank() == 0 else 0
##    train_start = time.time()
##
####    train_log = model.fit_generator(datagen.flow(x_train, y_train,
####                                                 batch_size=BATCH_SIZE)),
####                                    #steps_per_epoch=BATCH_SIZE//hvd.size())#,#epochs=N_EPOCHS,
####                                    #workers=2)
##    train_log = model.fit((x_train, y_train),batch_size=BATCH_SIZE)
##    train_end = time.time()
##
##    score = model.evaluate(x_test, y_test)
##    
##    results = {"test_loss": score[0],
##               "test_acc": score[1],
##               "train_time": train_end-train_start,
##               "train_log": train_log}
##    
##    return results
##
###Train without mixed precision
##fp32_results = train_model(mixed_precision=False, optimizer=opt)
##
##test_acc = round(fp32_results["test_acc"]*100, 1)
##train_time = round(fp32_results["train_time"], 1)
##print(test_acc, "% achieved in", train_time, "seconds")
##
##tf.keras.backend.clear_session()
##time.sleep(30)
##
###Train with mixed precision
##mp_results = train_model(mixed_precision=True, optimizer=opt)
##
##test_acc = round(mp_results["test_acc"]*100, 1)
##train_time = round(mp_results["train_time"], 1)
##
##print(test_acc, "% achieved in", train_time, "seconds")
##
#####Evaluate performance
####import matplotlib.pyplot as plt
####
####plt.plot(fp32_results["train_log"].history["loss"], label="FP32")
####plt.plot(mp_results["train_log"].history["loss"], label="Mixed Precision")
####plt.title("Performance Comparison")
####plt.ylabel("Training Loss")
####plt.xlabel("Epoch")
####plt.legend()
####plt.show()
####print(test_acc, "% achieved in", train_time, "seconds")
##
###Get speedup
##speed_up = round(100 * fp32_results["train_time"]/mp_results["train_time"], 1)
##
##print("Total speed-up:", speed_up, "%")

# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
import horovod.tensorflow.keras as hvd
import time
import argparse
ap = argparse.ArgumentParser()

ap.add_argument("-g", "--gpus", type=int, default=2,
	help="# of GPUs to use for training")
args = vars(ap.parse_args())
ap = argparse.ArgumentParser()
G = args["gpus"]

# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

(mnist_images, mnist_labels), _ = \
    tf.keras.datasets.mnist.load_data(path='mnist-%d.npz' % hvd.rank())

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
             tf.cast(mnist_labels, tf.int64))
)
dataset = dataset.repeat().shuffle(10000).batch(128)

mnist_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
    tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Horovod: adjust learning rate based on number of GPUs.
opt = tf.keras.optimizers.Adam(0.001 * hvd.size())

# Horovod: add Horovod DistributedOptimizer.
opt = hvd.DistributedOptimizer(opt)#,compression=compression=hvd.Compression.fp16)

mnist_model.compile(loss="categorical_crossentropy",
                    optimizer=opt,
                    metrics=['accuracy'])

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3, verbose=1),
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

# Horovod: write logs on worker 0.
verbose = 1 if hvd.rank() == 0 else 0
startTime = time.time()
# Train the model.
# Horovod: adjust number of steps based on number of GPUs.
mnist_model.fit(dataset, steps_per_epoch=500 // hvd.size(), callbacks=callbacks, epochs=24, verbose=verbose)
endTime = time.time()
print('Training time: ', endTime - startTime)

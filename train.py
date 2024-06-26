import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sn
import skimage.io
import keras.backend as K
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, VGG16, convnext, densenet, efficientnet, efficientnet_v2, imagenet_utils, inception_resnet_v2, inception_v3, mobilenet, mobilenet_v2, mobilenet_v3, nasnet, resnet, resnet_v2, VGG19, xception
from tensorflow.keras.layers import Dense, Flatten, Dropout,BatchNormalization ,Activation
from tensorflow.keras.models import Model, Sequential
from keras.applications.nasnet import NASNetLarge
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

import wandb
from wandb.integration.keras import WandbCallback

import os
import glob
from collections import Counter
import math

# 폴더 경로 설정
base_path = '...' # 폴더 경로 가림

classes = ['Happy', 'Neutral', 'Notgood', 'Sad', 'Surpring']

class_frequencies = np.array([1331, 1147, 239, 99, 300])

total_samples = np.sum(class_frequencies)
class_weights = total_samples / (len(class_frequencies) * class_frequencies)

def weighted_cross_entropy(y_true, y_pred):
    loss = tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=class_weights)
    return tf.reduce_mean(loss)

def f1_score(y_true, y_pred): 
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


sweep_config = {
    'method': 'bayes',
    'metric': {
        'goal': 'minimize',
        'name': 'val_loss'
    },
    'parameters': {
        'batch_size': {
            'values': [16, 32, 64, 128]
        },
        'epochs': {
            'values': [500]
        },
        'learning_rate': {  # learning rate
            'distribution': 'uniform',
            'min': 1e-7,
            'max': 1e-2
        }
    },
    'name': '240608_4_VGG16_32_64'
}

sweep_id = wandb.sweep(sweep_config, project='aigazzang')


def train():
    wandb.init()

    config = wandb.config
        

    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2,
                                       rotation_range=15, width_shift_range=0.2, height_shift_range=0.2,
                                       shear_range=0.2, horizontal_flip=True, vertical_flip=True,
                                       brightness_range=[0.8, 1.2], fill_mode='nearest')

    valid_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)


    train_dataset = train_datagen.flow_from_directory(directory='/home/stratio/Desktop/deep_man/cropped_faces',
                                                      target_size=(48, 48), class_mode='categorical',
                                                      subset='training', batch_size=config.batch_size)

    valid_dataset = valid_datagen.flow_from_directory(directory='/home/stratio/Desktop/deep_man/cropped_faces',
                                                      target_size=(48, 48), class_mode='categorical',
                                                      subset='validation', batch_size=config.batch_size)
    

    base_model = VGG16(input_shape=(48, 48, 3), include_top=False, weights="imagenet")
    
    for layer in base_model.layers[:-4]:
        layer.trainable = False

    model = Sequential([
        base_model,
        Dropout(0.5),
        Flatten(),
        BatchNormalization(),
        Dense(32, kernel_initializer='he_uniform'),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        Dense(32, kernel_initializer='he_uniform'),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        Dense(32, kernel_initializer='he_uniform'),
        BatchNormalization(),
        Activation('relu'),
        Dense(5, activation='softmax')
    ])
    
    METRICS = [
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),  
            tf.keras.metrics.AUC(name='auc'),
                f1_score,
        ]
    
    
    initial_learning_rate = config.learning_rate
    lr_schedule = tf.keras.experimental.CosineDecayRestarts(
        initial_learning_rate,
        first_decay_steps=1000,
        t_mul=2.0,
        m_mul=1.0,
        alpha=0.0
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)  
    
    
    lrd = ReduceLROnPlateau(monitor = 'val_accuracy', patience = 20, verbose = 1, factor = 0.50)
    mcp = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    es = EarlyStopping(verbose=1, patience=20)
    
    callbacks = [lrd, es, WandbCallback(), mcp]

    model.compile(optimizer=optimizer, loss=weighted_cross_entropy, metrics=METRICS)
    
    model.fit(train_dataset, validation_data=valid_dataset, epochs=config.epochs, verbose=2, callbacks=callbacks)
    
wandb.agent(sweep_id, function=train)

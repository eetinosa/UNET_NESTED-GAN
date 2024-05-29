"""Pre Generator training script.
"""

from __future__ import print_function

import numpy as np
import os
from keras import layers
import tensorflow as tf
from PIL import Image
from skimage.transform import resize

import matplotlib.pyplot as plt

from skimage.transform import resize
from keras.losses import binary_crossentropy
from keras.models import Model, Sequential
from keras.layers import (
    Input,
    concatenate,
    Conv2D,
    ZeroPadding2D,
    Convolution2D,
    Conv2DTranspose,
    MaxPooling2D,
    add,
    UpSampling2D,
    multiply,
    Dropout,
    BatchNormalization,
    LeakyReLU,
    Dense,
    Flatten,
)
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    Dense,
    GlobalAveragePooling2D,
    Dropout,
)
from keras.layers import Input, GaussianNoise
from keras.optimizers import Adam
from keras.layers.core import Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras import backend as K, models
from skimage.io import imsave
from gan_utils import imgs2discr, imgs2gan



K.set_image_data_format("channels_last")  # TF dimension ordering in this code
smooth = 1.0

img_rows = 128
img_cols = 128

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_path = "dataset\\file\\"
path_to_save_results = data_path + "predictions\\"
path_to_trained_generator_weights = "dataset\\trained_models"


def load_train_data():
    imgs_train = np.load(data_path + "train.npy")
    imgs_mask_train = np.load(data_path + "train_mask.npy")
    return imgs_train, imgs_mask_train


def load_validation_data():
    imgs_valid = np.load(data_path + "validation.npy")
    imgs_mask_valid = np.load(data_path + "validation_mask.npy")
    return imgs_valid, imgs_mask_valid


def load_test_data():
    imgs_test = np.load(data_path + "test.npy")
    imgs_mask_test = np.load(data_path + "test_mask.npy")
    return imgs_test, imgs_mask_test

def true_positives(y_true, y_pred):
    tp=tf.reduce_sum(tf.round(K.clip(y_true*y_pred, 0, 1)))
    return tp

def true_negatives(y_true, y_pred):
    tn=tf.reduce_sum(tf.round(K.clip((1-y_true)*(1-y_pred), 0, 1)))
    return tn

def false_positives(y_true, y_pred):
    fp=tf.reduce_sum(tf.round(K.clip((1-y_true)*(y_pred), 0, 1)))
    return fp

def false_negatives(y_true, y_pred):
    fn=tf.reduce_sum(tf.round(K.clip((y_true)*(1-y_pred), 0, 1)))
    return fn

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefilterLoss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def mean_iou(y_true, y_pred):
    true_positives=tf.reduce_sum(tf.round(K.clip(y_true*y_pred, 0, 1)))
    true_negatives=tf.reduce_sum(K.round(K.clip((1-y_true)*(1-y_pred), 0, 1)))
    possible_negatives=tf.reduce_sum(K.round(K.clip((1-y_true), 0, 1)))
    possible_positives=tf.reduce_sum(tf.round(K.clip(y_true, 0, 1)))
    iou = (true_positives)/(possible_negatives - true_negatives + possible_positives + K.epsilon())
    return K.mean(iou)  


def sensitivity(y_true, y_pred):
    tp= true_positives(y_true, y_pred)
    fn= false_negatives(y_true, y_pred)
    return tp / (tp + fn + K.epsilon())


def specificity(y_true, y_pred):
    tn = true_negatives(y_true, y_pred)
    fp = false_positives(y_true, y_pred)
    return tn / (tn + fp+ K.epsilon())


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    p = true_positives / (predicted_positives + K.epsilon())
    return p


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    r = true_positives / (possible_positives + K.epsilon())
    return r


def f1score(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        r = true_positives / (possible_positives + K.epsilon())
        return r

    def precision(y_true, y_pred):
       
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        p = true_positives / (predicted_positives + K.epsilon())
        return p

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    return 2 * ((p * r) / (p + r + K.epsilon()))


def attnGate(InATT, InATT2, filterG, filterL, filterInt):

    up = Conv2D(filterG, (1, 1), activation="relu", padding="same")(InATT)
    up = BatchNormalization()(up)

    down = Conv2D(filterL, (1, 1), activation="relu", padding="same")(InATT2)
    down = BatchNormalization()(down)
    
    sumadd = add([up, down])
    sumadd = Activation(activation="relu")(sumadd)

    path = Conv2D(filterInt, (1, 1), activation="relu", padding="same")(sumadd)
    sumhalf = BatchNormalization()(path)

    sum_1 = Conv2D(1, (1, 1), activation="sigmoid", padding="same")(sumhalf)
    sum_1 = BatchNormalization()(sum_1)

    attnOutput = multiply([InATT2, sum_1]) # attention gate output is the multiplication of the attention coefficients with the encoder layer output
    
    # attnOutput = add([up, attnOutput]) ############
    
    return attnOutput


def DilatedConv(x,filters_bottleneck,mode="dilation1", depth=6,kernel_size=(3, 3),activation="relu"):
    
    dilated_layers = []
    if mode == "dilation1":  
        for i in range(depth):
            x = Conv2D(filters_bottleneck,kernel_size,activation=activation,padding="same", dilation_rate=2 ** i)(x)
            dilated_layers.append(x)
        return add(dilated_layers)
    
    elif mode == "dilation2":  
        for i in range(depth):
            dilated_layers.append(
                Conv2D(filters_bottleneck,kernel_size,activation=activation,padding="same", dilation_rate=2 ** (2*i))(x))
        return add(dilated_layers)

def DilatedConv2 (x,filters_bottleneck,mode="dilation1", depth=3,kernel_size=(3, 3),activation="relu"):
    
    dilated_layers = []
    if mode == "dilation1":  
        for i in range(depth):
            x = Conv2D(filters_bottleneck,kernel_size,activation=activation,padding="same", dilation_rate=2 ** i)(x)
            dilated_layers.append(x)
        return add(dilated_layers)
    
    elif mode == "dilation2":  
        for i in range(depth):
            dilated_layers.append(
                Conv2D(filters_bottleneck,kernel_size,activation=activation,padding="same", dilation_rate=2 ** (2*i))(x))
        return add(dilated_layers)

def res_block(x, nb_filters, strides):
    res_path = BatchNormalization()(x)
    res_path = Activation(activation="relu")(res_path)
    
    res_path = Conv2D(filters=nb_filters, kernel_size=(3, 3), padding="same", strides=strides[0])(res_path)
    
    res_path = BatchNormalization()(res_path)
    res_path = Activation(activation="relu")(res_path)
    
    res_path = Conv2D(filters=nb_filters, kernel_size=(3, 3), padding="same", strides=strides[1])(res_path)

    link = Conv2D(nb_filters, kernel_size=(1, 1), strides=strides[0])(x)
    
    link = BatchNormalization()(link)

    res_path = add([link, res_path])
    return res_path


def encoder(x):
    to_decoder = []
    
    #first residual block
    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding="same", strides=(1, 1))(x)
    main_path = BatchNormalization()(main_path)
    main_path = Activation(activation="relu")(main_path)

    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding="same", strides=(1, 1))(main_path)

    shortcut = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1))(x)
    shortcut = BatchNormalization()(shortcut)

    main_path = add([shortcut, main_path])
    
    # first branching to decoder
    to_decoder.append(main_path)

    # main_path = DilatedConv2(main_path, filters_bottleneck=64, mode="dilation1", depth=2)
    main_path = res_block(main_path, 64, [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    # main_path = DilatedConv2(main_path, filters_bottleneck=128, mode="dilation1", depth=2)
    main_path = res_block(main_path, 128, [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    # main_path = DilatedConv2(main_path, filters_bottleneck=256, mode="dilation1", depth=2)
    main_path = res_block(main_path, 256, [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    # main_path = DilatedConv2(main_path, filters_bottleneck=512, mode="dilation1", depth=2)
    main_path = res_block(main_path, 512, [(2, 2), (1, 1)])
    to_decoder.append(main_path)
    return to_decoder


def decoder(x, EncoderOutput):
    main_path = UpSampling2D(size=(2, 2))(x)
    xin_encoder_1 = attnGate(main_path, EncoderOutput[4], 256, 256, 128)
    main_path = concatenate([main_path, xin_encoder_1], axis=3)
    main_path = res_block(main_path, 512, [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    xin_encoder_2 = attnGate(main_path, EncoderOutput[3], 128, 128, 64)
    main_path = concatenate([main_path, xin_encoder_2], axis=3)
    main_path = res_block(main_path, 256, [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    xin_encoder_3 = attnGate(main_path, EncoderOutput[2], 64, 64, 32)
    main_path = concatenate([main_path, xin_encoder_3], axis=3)
    main_path = res_block(main_path, 128, [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    xin_encoder_4 = attnGate(main_path, EncoderOutput[1], 32, 32, 16)
    main_path = concatenate([main_path, xin_encoder_4], axis=3)
    main_path = res_block(main_path, 64, [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    xin_encoder_5 = attnGate(main_path, EncoderOutput[0], 16, 16, 8)
    main_path = concatenate([main_path, xin_encoder_5], axis=3)
    main_path = res_block(main_path, 32, [(1, 1), (1, 1)])

    return main_path


def build_RDA_UNET():
    # inputs = Input(shape=input_shape)
    img_rows = 128
    img_cols = 128
    inputs = Input(shape=(img_rows, img_cols, 1))

    to_decoder = encoder(inputs)

    path = res_block(to_decoder[4], 512, [(2, 2), (1, 1)])

    bottle = DilatedConv(path, filters_bottleneck=256, mode="dilation1", depth=6) #Dilation convolution between encoder and decoder

    path = decoder(bottle, EncoderOutput=to_decoder)
    
    path = DilatedConv2(path, filters_bottleneck=32, mode="dilation1",depth=3)  #Dilation Convolution after the decoder

    path = Conv2D(filters=1, kernel_size=(1, 1), activation="hard_sigmoid", padding="same")(path)    
    
    model = Model(inputs, path)

    model.compile(
        optimizer="adam",
        loss=dice_coefilterLoss,
        metrics=["accuracy", dice_coef,sensitivity,specificity,f1score,precision,recall,mean_iou])

    return model #generator model


def make_trainable(network, value):
    """
    If False, it fixes the network and it is not trainable (the weights are frozen)
    If True, the network is trainable (the weights can be updated)
    :param net: network
    :param val: boolean to make the network trainable or not
    """
    network.trainable = value
    for l in network.layers:
        l.trainable = value


def build_discriminator():

    k = 3  # kernel size
    s = 2  # stride
    n_filters = 32  # number of filters
    inputs = Input(shape=(img_rows, img_cols, 2))

    conv1 = Conv2D(n_filters, kernel_size=(k, k), strides=(s, s), padding="same")(inputs)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    
    conv1 = Conv2D(n_filters, kernel_size=(k, k), padding="same")(conv1)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    
    pool1 = MaxPooling2D(pool_size=(s, s))(conv1)

    conv2 = Conv2D(2 * n_filters, kernel_size=(k, k), strides=(s, s), padding="same")(pool1)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    
    conv2 = Conv2D(2 * n_filters, kernel_size=(k, k), padding="same")(conv2)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    
    pool2 = MaxPooling2D(pool_size=(s, s))(conv2)

    conv3 = Conv2D(4 * n_filters, kernel_size=(k, k), padding="same")(pool2)
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    conv3 = Conv2D(4 * n_filters, kernel_size=(k, k), padding="same")(conv3)
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    pool3 = MaxPooling2D(pool_size=(s, s))(conv3)

    conv4 = Conv2D(8 * n_filters, kernel_size=(k, k), padding="same")(pool3)
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    conv4 = Conv2D(8 * n_filters, kernel_size=(k, k), padding="same")(conv4)
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPooling2D(pool_size=(s, s))(conv4)

    conv5 = Conv2D(16 * n_filters, kernel_size=(k, k), padding="same")(pool4)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = LeakyReLU(alpha=0.1)(conv5)
    conv5 = Conv2D(16 * n_filters, kernel_size=(k, k), padding="same")(conv5)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = LeakyReLU(alpha=0.1)(conv5)

    gap = GlobalAveragePooling2D()(conv5)
    outputs = Dense(1, activation="sigmoid")(gap)

    model = Model(inputs, outputs)

    # loss of the discriminator. it is a binary loss
    def discriminator_loss(y_true, y_pred):
        loss = binary_crossentropy(K.batch_flatten(y_true), K.batch_flatten(y_pred))
        return loss

    model.compile(
        optimizer=Adam(lr=1e-4),
        loss=discriminator_loss,
        metrics=["accuracy",dice_coef,sensitivity,specificity,f1score,precision,recall,mean_iou])

    return model

###################################################################################

def build_gan(generator, discriminator):

    image = Input(shape=(img_rows, img_cols, 1))
    mask = Input(shape=(img_rows, img_cols, 1))

    fake_mask = generator(image)
    fake_pair = concatenate([image, fake_mask]) #(axis=3) 

    gan = Model([image, mask], discriminator(fake_pair))

    def wasserstein_loss(y_true, y_pred):
        return K.mean(y_true * y_pred)

    # gan.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy',dice_coef, sensitivity,specificity,f1score,precision,recall,mean_iou])
    gan.compile(
        optimizer=Adam(lr=1e-4),
        loss=wasserstein_loss,
        metrics=["accuracy",dice_coef,sensitivity, specificity, f1score, precision, recall, mean_iou])
    
    return gan


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def train():

    # get data
    print("-" * 30)
    print("Loading and preprocessing train data...")
    print("-" * 30)
    imgs_train, imgs_mask_train = load_train_data()
    imgs_valid, imgs_mask_valid = load_validation_data()

    imgs_train = preprocess(imgs_train)
    print(imgs_train.shape)
    imgs_mask_train = preprocess(imgs_mask_train)
    print(imgs_mask_train.shape)
    imgs_valid = preprocess(imgs_valid)
    print(imgs_valid.shape)
    imgs_mask_valid = preprocess(imgs_mask_valid)
    print(imgs_mask_valid.shape)

    imgs_train = imgs_train.astype("float32")
    imgs_valid = imgs_valid.astype("float32")

    # print(imgs_train.shape)

    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    val_mean = np.mean(imgs_valid)
    val_std = np.std(imgs_valid)

    imgs_train -= mean
    imgs_train /= std

    imgs_valid -= val_mean
    imgs_valid /= val_std

    imgs_mask_train = imgs_mask_train.astype("float32")
    imgs_mask_train /= 255.0  # scale masks to [0, 1]

    imgs_mask_valid = imgs_mask_valid.astype("float32")
    imgs_mask_valid /= 255.0

    # preparing test data
    imgs_test, imgs_test_mask = load_test_data()
    imgs_test = preprocess(imgs_test)
    imgs_test_mask = preprocess(imgs_test_mask)
    imgs_test_source = imgs_test.astype("float32")
    imgs_test_source -= mean
    imgs_test_source /= std
    imgs_test_mask = imgs_test_mask.astype("float32")
    imgs_test_mask /= 255.0  # scale masks to [0, 1]

    # Get the generator model i.e UNet
    print("-" * 30)
    print("Creating and compiling the generator model...")
    print("-" * 30)
    model_generator = build_RDA_UNET()
    print(model_generator.summary())
    
    def get_batch_train():
        idx = np.random.randint(0, imgs_train.shape[0], batch_size)
        imgs = imgs_train[idx]
        masks = imgs_mask_train[idx]
        return imgs, masks

    def get_batch_valid():
        idx = np.random.randint(0, imgs_valid.shape[0], batch_size)
        imgs = imgs_valid[idx]
        masks = imgs_mask_valid[idx]
        return imgs, masks

    n_rounds = 20 # number of rounds to apply adversarial training
    batch_size = 32

    # Getting data and its shape
    print("Getting train and validation data...")

    steps_per_epoch_g = 1
    

    

    gen_loss = np.zeros(n_rounds)
    gen_acc = np.zeros(n_rounds)
    gen_sensitivity = np.zeros(n_rounds)
    gen_specificity = np.zeros(n_rounds)
    gen_f1score = np.zeros(n_rounds)
    gen_precision = np.zeros(n_rounds)
    gen_recall = np.zeros(n_rounds)
    gen_mean_iou = np.zeros(n_rounds)
    gen_dice_coeff = np.zeros(n_rounds)

    imgs_valid, imgs_mask_valid = get_batch_valid()
    # model_checkpoint = ModelCheckpoint('dataset/pre_unet.hdf5', monitor='val_loss',
                                    #    save_best_only=True)

    print("-" * 30)
    print("GEN training...")
    print("-" * 30)
    # earlystopper=EarlyStopping(monitor='val_loss',patience=10,verbose=1)
    # model_generator.fit(imgs_train, imgs_mask_train, batch_size=32, epochs=2, verbose=1, shuffle=True,
    #                 validation_data=(imgs_valid, imgs_mask_valid), callbacks=[model_checkpoint])

    for n_round in range(n_rounds):
        print("Training Generator...")
        # train Discriminator
        # make_trainable(model_discriminator, True)
        for i in range(steps_per_epoch_g):
            image_batch, labels_batch = get_batch_train()
            # pred = model_generator.predict(image_batch)
            # img_discr_batch, lab_discr_batch = imgs2discr(image_batch, labels_batch, pred) 
            (loss,acc,dice_coef,sensitivity,specificity,f1score,precision,recall,mean_iou) = model_generator.train_on_batch(image_batch, labels_batch)
            
        gen_loss[n_round] = loss
        gen_acc[n_round] = acc
        gen_dice_coeff[n_round] = dice_coef
        gen_sensitivity[n_round] = sensitivity
        gen_specificity[n_round] = specificity
        gen_f1score[n_round] = f1score
        gen_precision[n_round] = precision
        gen_recall[n_round] = recall
        gen_mean_iou[n_round] = mean_iou
        print("Generator Round: {0} -> Loss {1}".format((n_round + 1), loss))



        

        # save the weights of the unet
        if not os.path.exists(path_to_trained_generator_weights):
            os.makedirs(path_to_trained_generator_weights, exist_ok=True)
        model_generator.save_weights(
            os.path.join(
                (path_to_trained_generator_weights + "\pre_gen\\"),
                ("pre_gen_weights_{}.hdf5".format(n_round))
            )
        )
       
    

if __name__ == "__main__":
    train()

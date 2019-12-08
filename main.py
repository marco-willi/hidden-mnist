"""
    Simplified Implementation of HiDDeN - MNIST Dataset & Simple Noise Addition
    Reference: HiDDeN: Hiding data with deep networks (Zhu et. al, 2018)
"""
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

# Get Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

data = x_train
data = np.expand_dims(data, -1)
input_shape = data.shape[1:]
data = data.astype(np.float32)
np.min(data)
# Restrict to values between 0 and 1
data = (data / 255)

# Parameters
msg_length = 8

# msg = np.array([x for x in range(0, 32)])
# msg = np.expand_dims(msg, 0)
# xx = tf.keras.layers.RepeatVector(4*4)(msg)
# xx.shape
# xx[0,1,:]
# yy = tf.keras.layers.Reshape((4, 4, 32))(xx)
# yy.shape
# yy[0, 0, 1,:]

####################
# EncoderDecoder

def encoder_net(cover_image, message, input_shape, msg_length):

    # Message Block
    mb = tf.keras.layers.RepeatVector(
        input_shape[0] * input_shape[1])(message)
    mb = tf.keras.layers.Reshape((input_shape[0:2]) + (msg_length, ))(mb)


    # Image Processing Block
    x = cover_image

    for _ in range(0, 3):
        x = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)

    # Concatenate Message Block with Image Processing Block and original image
    x = tf.keras.layers.Concatenate(axis=-1)([mb, x, cover_image])

    # Encoded image
    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    encoded_img = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        activation='linear')(x)
    return encoded_img


def decoder_net(encoded_image, msg_length):

    x = encoded_image

    for _ in range(0, 3):
        x = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2D(
        filters=msg_length,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(msg_length)(x)
    return x


def encoder_decoder_net(input_shape, msg_length):
    # inputs
    message = tf.keras.Input(shape=(msg_length, ), name='message')
    cover_image = tf.keras.Input(shape=input_shape, name='cover_image')

    encoded_image = encoder_net(cover_image, message, input_shape, msg_length)
    decoded_message = decoder_net(encoded_image, msg_length)

    encoder_decoder = tf.keras.Model(
        {'cover_image': cover_image, 'message': message},
        outputs={
            'encoded_image': encoded_image,
            'decoded_message': decoded_message})
    return encoder_decoder



####################
# Adversary

def adversary_net(input_shape):

    image = tf.keras.Input(shape=input_shape)
    x = image

    for _ in range(0, 2):
        x = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    logits = tf.keras.layers.Dense(1)(x)

    adversary = tf.keras.Model(inputs=image, outputs=logits)

    return adversary


#################
# input data
def create_dataset(data, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.map(lambda x: tf.cast(x, dtype=tf.float32))
    dataset = dataset.shuffle(60000)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat(1)
    return dataset

def reconstruction_loss(y, y_hat):
    return tf.reduce_mean(tf.math.pow(y - y_hat, 2))

def adversary_loss_cover(logits_cover):
    """ Adversary Loss on Cover Images """
    return tf.keras.losses.binary_crossentropy(
        y_true=tf.zeros_like(logits_cover),
        y_pred=logits_cover, from_logits=True)

def adversary_loss_encoded(logits_encoded)
    """ Adversary Loss on Encoded Images """:
    return tf.keras.losses.binary_crossentropy(
        y_true=tf.ones_like(logits_encoded),
        y_pred=logits_encoded, from_logits=True)

def decoding_error_rate(y, y_hat):
    return tf.reduce_mean(tf.math.abs(y - tf.math.round(y_hat)), axis=0)


#################
# Create Architectures + optimizers
encoder_decoder = encoder_decoder_net(input_shape, msg_length)
adversary = adversary_net(input_shape)

optimizer_encoder_decoder = tf.keras.optimizers.Adam()
optimizer_adversary = tf.keras.optimizers.Adam()

#################
# Training Step

@tf.function
def train_step(cover_images, messages, optimizer_encoder_decoder, optimizer_adversary):
    with tf.GradientTape() as tape_ed, tf.GradientTape() as tape_adv:

        encoder_decoder_output = encoder_decoder(
            inputs={'cover_image': cover_images, 'message': messages},
            training=True)

        adverse_cover = adversary(
            inputs={'image': cover_images},
            training=True)

        adverse_encoded = adversary(
            inputs={'image': encoder_decoder_output['encoded_image']},
            training=True)

        # Loss of Encoder Decoder
        loss_encoder = reconstruction_loss(
            cover_images, encoder_decoder_output['encoded_image'])

        loss_decoder = reconstruction_loss(
            messages, encoder_decoder_output['decoded_message'])
        
        loss_detectability = tf.reduce_mean(adversary_loss_cover(adverse_encoded))

        # loss of adversary
        loss_adversary_cover = adversary_loss_cover(adverse_cover)
        loss_adversary_encoded = adversary_loss_encoded(adverse_encoded)

        # total loss adversary
        loss_adversary = tf.reduce_mean(
            loss_adversary_cover + loss_adversary_encoded)

        # total loss encoder decoder
        loss_encoder_decoder = loss_encoder + loss_decoder + loss_detectability

    # gradient updates
    grads_encoder_decoder = tape_ed.gradient(
        loss_encoder_decoder, encoder_decoder.trainable_variables)
    optimizer_encoder_decoder.apply_gradients(
        zip(grads_encoder_decoder, encoder_decoder.trainable_variables))

    grads_adversary = tape_adv.gradient(
        loss_adversary, adversary.trainable_variables)
    optimizer_adversary.apply_gradients(
        zip(grads_adversary, adversary.trainable_variables))
    
    # calculate message recovery error rate
    mean_error_per_sample = decoding_error_rate(
        messages, encoder_decoder_output['decoded_message'])
    mean_error = tf.reduce_mean(mean_error_per_sample)
        
    return loss_encoder_decoder, loss_adversary


#################
# Training Loop

batch_size = 32
n_steps = 100
n_epochs = 20

data_train = create_dataset(data, batch_size)

for e in range(0, n_epochs):
    data_train = create_dataset(data, batch_size)

    for step, cover_images in enumerate(data_train):

        messages = tf.random.uniform(
            [batch_size, msg_length], minval=0, maxval=2, dtype=tf.dtypes.int32)
        messages = tf.cast(messages, dtype=tf.dtypes.float32)

        loss_de, loss_adv = train_step(
            cover_images, messages,
            optimizer_encoder_decoder, optimizer_adversary)

    # Plot Examples
    encoder_decoder_output = encoder_decoder(
        inputs={'cover_image': cover_images, 'message': messages},
        training=False)

    mean_error_per_sample = decoding_error_rate(
        messages, encoder_decoder_output['decoded_message'])
    mean_error = tf.reduce_mean(mean_error_per_sample)

    print("==========================================")
    print("Epoch: {} Loss EncoderDecoder: {:.5f}".format(e, loss_de))
    print("Epoch: {} Loss Adversary: {:.5f}".format(e, loss_adv))
    print("Epoch: {} Decoder Error Rate (1 batch): {:.2f}".format(e, mean_error))

    for j in range(0, 2):
        img_cover = cover_images[j:j+1, :, :, :]
        img_encoded = encoder_decoder_output['encoded_image'][j]
        img_encoded = tf.squeeze(img_encoded, -1)
        img_cover = np.squeeze(img_cover)
        img_diff = np.abs(img_cover - img_encoded)

        fig = plt.figure(figsize=(8, 8))

        plt.subplot(1, 3, 1)
        plt.imshow(img_encoded, cmap='gray')
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(img_cover, cmap='gray')
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(img_diff, cmap='gray')
        plt.axis('off')
        plt.show()



messages
tf.round(encoder_decoder_output['decoded_message'])

adverse_encoded = adversary(
    inputs={'image': encoder_decoder_output['encoded_image']},
    training=False)

tf.sigmoid(adverse_encoded)


adverse_cover = adversary(
    inputs={'image': cover_images},
    training=False)

adverse_cover
tf.sigmoid(adverse_cover)

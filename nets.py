"""
    Network Architectures
"""
import tensorflow as tf

import noise


def encoder(cover_image, message, input_shape, msg_length):
    """ Create Encoder Net """

    # Message Block
    m = tf.keras.layers.RepeatVector(
        input_shape[0] * input_shape[1])(message)
    m = tf.keras.layers.Reshape((input_shape[0:2]) + (msg_length, ))(m)

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

    # Concatenate Message Block with Image Processing Block and Cover Image
    x = tf.keras.layers.Concatenate(axis=-1)([m, x, cover_image])

    # Encode Image
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


def decoder(encoded_image, msg_length):
    """ Decoder Net """

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
    decoded_message = tf.keras.layers.Dense(msg_length)(x)
    return decoded_message


def encoder_decoder(input_shape, msg_length, noise_type):
    """ EncoderDecoder Net """
    message = tf.keras.Input(shape=(msg_length, ), name='message')
    cover_image = tf.keras.Input(shape=input_shape, name='cover_image')

    encoded_image = encoder(cover_image, message, input_shape, msg_length)

    if noise_type == 'identity':
        noised_image = encoded_image
    elif noise_type == 'gaussian':
        noised_image = noise.GaussianBlurring2D(
            kernel_size=(3, 3), padding='same')(encoded_image)
    elif noise_type == 'dropout':
        noised_image = noise.DropOutLayer()(encoded_image, cover_image)
    else:
        raise NotImplementedError(
            "noise_type {} not implemented".format(noise_type))

    decoded_message = decoder(noised_image, msg_length)

    model = tf.keras.Model(
        inputs={
            'cover_image': cover_image,
            'message': message},
        outputs={
            'encoded_image': encoded_image,
            'noised_image': noised_image,
            'decoded_message': decoded_message})
    return model


####################
# Discriminator

def discriminator(input_shape):
    """ Discriminator Net: Identify Encoded Images """

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

    model = tf.keras.Model(inputs=image, outputs=logits)

    return model

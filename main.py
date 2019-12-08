"""
    Simplified Implementation of HiDDeN - MNIST Dataset & Simple Noise Addition
    Reference: HiDDeN: Hiding data with deep networks (Zhu et. al, 2018)
"""
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

# Get Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Transform Dataset
data = x_train
data = np.expand_dims(data, -1)
input_shape = data.shape[1:]
data = data.astype(np.float32)

# Restrict to values between 0 and 1
data = (data / 255.0)

####################
# EncoderDecoder

def encoder_net(cover_image, message, input_shape, msg_length):
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


def decoder_net(encoded_image, msg_length):
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


def encoder_decoder_net(input_shape, msg_length):
    """ EncoderDecoder Net """
    message = tf.keras.Input(shape=(msg_length, ), name='message')
    cover_image = tf.keras.Input(shape=input_shape, name='cover_image')

    encoded_image = encoder_net(cover_image, message, input_shape, msg_length)
    decoded_message = decoder_net(encoded_image, msg_length)

    model = tf.keras.Model(
        inputs={
            'cover_image': cover_image,
            'message': message},
        outputs={
            'encoded_image': encoded_image,
            'decoded_message': decoded_message})
    return model


####################
# Discriminator

def discriminator_net(input_shape):
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


#################
# input data

def create_dataset(data, batch_size, shuffle_buffer=60000):
    """ Dataset Iterator """
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.map(lambda x: tf.cast(x, dtype=tf.float32))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat(1)
    return dataset

def distortion_loss(y, y_hat):
    """ Image Distortion Loss - L2 Norm """
    return tf.reduce_mean(tf.math.pow(y - y_hat, 2))

def recovery_loss(y, y_hat):
    """ Message Recovery Loss - L2 Norm """
    per_message = tf.reduce_mean(tf.math.pow(y - y_hat, 2), axis=0)
    return tf.reduce_mean(per_message)

def discriminator_loss(logits, y_expected):
    """ Discriminator Loss """
    y_true = tf.fill(logits.shape, y_expected)
    return tf.keras.losses.binary_crossentropy(
        y_true=y_true,
        y_pred=logits,
        from_logits=True)

def decoding_error_rate(y, y_hat):
    """ Mean Error Rate across Batch """
    return tf.reduce_mean(tf.math.abs(y - tf.math.round(y_hat)), axis=0)


#################
# Training Step

@tf.function
def train_step(
    cover_images,
    messages,
    optimizer_encoder_decoder,
    optimizer_discriminator,
    loss_weight_recover=1.0,
    loss_weight_distortion=1.0,
    loss_weight_adversarial=1.0):
    with tf.GradientTape() as tape_ed, tf.GradientTape() as tape_adv:

        encoder_decoder_output = encoder_decoder(
            inputs={'cover_image': cover_images, 'message': messages},
            training=True)

        discriminator_on_cover = discriminator(
            inputs={'image': cover_images},
            training=True)

        discriminator_on_encoded = discriminator(
            inputs={'image': encoder_decoder_output['encoded_image']},
            training=True)

        # Loss of Encoder Decoder
        loss_distortion = distortion_loss(
            cover_images, encoder_decoder_output['encoded_image'])

        loss_recover = recovery_loss(
            messages, encoder_decoder_output['decoded_message'])
        
        loss_adversarial = tf.reduce_mean(
            discriminator_loss(discriminator_on_encoded, 0.0))

        # loss of Discriminator
        loss_discriminator_cover = discriminator_loss(
            discriminator_on_cover, 0.0)
        loss_discriminator_encoded = discriminator_loss(
            discriminator_on_encoded, 1.0)

        # total loss discriminator
        loss_discriminator = tf.reduce_mean(
            loss_discriminator_cover + loss_discriminator_encoded)

        # total loss encoder decoder
        loss_encoder_decoder = loss_weight_recover * loss_recover + \
            loss_weight_distortion * loss_distortion + \
            loss_weight_adversarial * loss_adversarial

    # gradient updates
    grads_encoder_decoder = tape_ed.gradient(
        loss_encoder_decoder, encoder_decoder.trainable_variables)
    optimizer_encoder_decoder.apply_gradients(
        zip(grads_encoder_decoder, encoder_decoder.trainable_variables))

    grads_discriminator = tape_adv.gradient(
        loss_discriminator, discriminator.trainable_variables)
    optimizer_discriminator.apply_gradients(
        zip(grads_discriminator, discriminator.trainable_variables))
    
    # Record losses
    losses = {
        'encoder_decoder': loss_encoder_decoder,
        'discriminator': loss_discriminator,
        'recover': loss_recover,
        'distortion': loss_distortion,
        'adversarial': loss_adversarial,
        'discriminator_cover': tf.reduce_mean(loss_discriminator_cover),
        'discriminator_encoded': tf.reduce_mean(loss_discriminator_encoded)}
            
    return losses


#################
# Training Loop

batch_size = 32
n_steps = 100
n_epochs = 20
msg_length = 8

loss_weight_recover=1.0,
loss_weight_distortion=0.7,
loss_weight_adversarial=1e-3

# Create Nets
encoder_decoder = encoder_decoder_net(input_shape, msg_length)
discriminator = discriminator_net(input_shape)

optimizer_encoder_decoder = tf.keras.optimizers.Adam(1e-3)
optimizer_discriminator = tf.keras.optimizers.Adam(1e-3)

data_train = create_dataset(data, batch_size)

for e in range(0, n_epochs):
    data_train = create_dataset(data, batch_size)

    for step, cover_images in enumerate(data_train):

        messages = tf.random.uniform(
            [batch_size, msg_length], minval=0, maxval=2, dtype=tf.dtypes.int32)
        messages = tf.cast(messages, dtype=tf.dtypes.float32)

        losses = train_step(
            cover_images, messages,
            optimizer_encoder_decoder, optimizer_discriminator,
            loss_weight_recover=loss_weight_recover,
            loss_weight_distortion=loss_weight_distortion,
            loss_weight_adversarial=loss_weight_adversarial)

    # Plot Examples
    encoder_decoder_output = encoder_decoder(
        inputs={'cover_image': cover_images, 'message': messages},
        training=False)

    mean_error_per_sample = decoding_error_rate(
        messages, encoder_decoder_output['decoded_message'])
    mean_error = tf.reduce_mean(mean_error_per_sample)

    print("==========================================")
    for loss, loss_value in losses.items():
        print("Epoch: {:<2} Loss {:<20}: {:.5f}".format(
            e, loss, float(loss_value.numpy())))
    print("Epoch: {:<2} Decoder Error Rate (1 batch): {:.2f}".format(
        e, mean_error))

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


#################
# Plot some figures

encoder_decoder_output = encoder_decoder(
    inputs={'cover_image': cover_images, 'message': messages},
    training=False)

n_rows = 6
n_cols = 3
index = 0
fig = plt.figure(figsize=(4, 8))
for j in range(0, n_rows):
    img_cover = cover_images[j:j+1, :, :, :]
    img_encoded = encoder_decoder_output['encoded_image'][j]
    img_encoded = tf.squeeze(img_encoded, -1)
    img_cover = np.squeeze(img_cover)
    img_diff = np.abs(img_cover - img_encoded)  
    plt.subplot(n_rows, n_cols, index + 1)
    plt.imshow(img_encoded, cmap='gray')
    plt.axis('off')
    plt.subplot(n_rows, n_cols, index + 2)
    plt.imshow(img_cover, cmap='gray')
    plt.axis('off')
    plt.subplot(n_rows, n_cols, index + 3)
    plt.imshow(img_diff, cmap='gray')
    plt.axis('off')
    index += 3
plt.tight_layout()
plt.savefig('./examples.png', bbox_inches='tight',
            pad_inches=0)
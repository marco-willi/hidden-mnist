"""
    Implementation of HiDDeN - Hiding Data in the MNIST Dataset
    Reference: "HiDDeN: Hiding data with deep networks (Zhu et. al, 2018)"
"""
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import utils
import nets

# Get Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# split train into train and validation
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.05, random_state=123)

# transform data
def transform_data(x):
    x = np.expand_dims(x, -1)
    x = x.astype(np.float32)
    x /= 255.0
    return x

data_train = transform_data(x_train)
data_test = transform_data(x_test)
data_val = transform_data(x_val)

input_shape = data_train.shape[1:]

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

def binarize(x):
    return tf.math.round(
                tf.clip_by_value(x, clip_value_min=0, clip_value_max=1))

def decoding_error_rate(y, y_hat):
    """ Mean Error Rate across Batch """
    return tf.reduce_mean(tf.math.abs(y - binarize(y_hat)), axis=0)


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
        loss_encoder_decoder = \
            loss_weight_recover * loss_recover + \
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
    
    # Record losses (only one batch atm)
    # TODO: add tf.summary
    losses = {
        'encoder_decoder': loss_encoder_decoder,
        'discriminator': loss_discriminator,
        'recover': loss_recover,
        'distortion': loss_distortion,
        'adversarial': loss_adversarial,
        'discriminator_cover': tf.reduce_mean(loss_discriminator_cover),
        'discriminator_encoded': tf.reduce_mean(loss_discriminator_encoded)}
            
    return losses


def create_messages(batch_size, msg_length):
    messages = tf.random.uniform(
    [batch_size, msg_length], minval=0, maxval=2, dtype=tf.dtypes.int32)
    messages = tf.cast(messages, dtype=tf.dtypes.float32)
    return messages


def evaluate_batches(n_batches, dataset, messages, encoder_decoder, discriminator):

    dataset_iter = iter(dataset)

    for step in range(0, n_batches):

        cover_images = next(dataset_iter)

        encoder_decoder_output = encoder_decoder(
            inputs={'cover_image': cover_images, 'message': messages},
            training=False)

        discriminator_on_cover = discriminator(
            inputs={'image': cover_images},
            training=False)

        discriminator_on_encoded = discriminator(
            inputs={'image': encoder_decoder_output['encoded_image']},
            training=False)

        decoded_msgs = encoder_decoder_output['decoded_message']

        bit_error_rate = tf.reduce_mean(tf.math.abs(
            messages - binarize(decoded_msgs)))

        false_positive_rate = tf.reduce_mean(
            binarize(tf.sigmoid(discriminator_on_cover)))

        true_positive_rate = tf.reduce_mean(
            binarize(tf.sigmoid(discriminator_on_encoded)))

        print("Step: {} ------------------------".format(step))
        print("Bit error rate: {:.2f}".format(bit_error_rate))
        print("Discriminator: False positive rate (cover images): {:.2f}".format(
            false_positive_rate))
        print("Discriminator: True positive rate (encoded images): {:.2f}".format(
            true_positive_rate))

#################
# Training Loop

batch_size = 32
batch_size_val = 256
n_epochs = 20
msg_length = 8

# one of: identity, gaussian, dropout
noise_type = 'gaussian'

loss_weight_recover=1.0,
loss_weight_distortion=0.7,
loss_weight_adversarial=1e-3

# Create Nets
encoder_decoder = nets.encoder_decoder(input_shape, msg_length, noise_type)
discriminator = nets.discriminator(input_shape)

optimizer_encoder_decoder = tf.keras.optimizers.Adam(1e-3)
optimizer_discriminator = tf.keras.optimizers.Adam(1e-3)

for e in range(0, n_epochs):
    dataset_train = create_dataset(data_train, batch_size)

    for step, cover_images in enumerate(dataset_train):

        #messages = create_messages(batch_size, msg_length)
        # TODO: verify tf.function speed-up
        # suspicion that messages = create_me... is way slower
        messages = tf.random.uniform(
             [batch_size, msg_length], minval=0, maxval=2, dtype=tf.dtypes.int32)
        messages = tf.cast(messages, dtype=tf.dtypes.float32)

        losses = train_step(
            cover_images, messages,
            optimizer_encoder_decoder, optimizer_discriminator,
            loss_weight_recover=loss_weight_recover,
            loss_weight_distortion=loss_weight_distortion,
            loss_weight_adversarial=loss_weight_adversarial)

    # evaluate on validation batch
    print("==========================================")
    print("Epoch: {} Validation Metrics".format(e))
    messages_val = create_messages(batch_size_val, msg_length)
    dataset_val = create_dataset(data_val, batch_size_val)
    evaluate_batches(1, dataset_val, messages_val, encoder_decoder, discriminator)

    # print loss values
    for loss, loss_value in losses.items():
        print("Epoch: {:<2} Training Loss {:<20}: {:.5f}".format(
            e, loss, float(loss_value.numpy())))

    # Plot Examples
    # TODO: switch from training to val examples
    encoder_decoder_output = encoder_decoder(
        inputs={'cover_image': cover_images, 'message': messages},
        training=False)

    fig = plt.figure(figsize=(4, 4))
    images_to_plot = [
        cover_images,
        encoder_decoder_output['encoded_image'],
        (cover_images - encoder_decoder_output['encoded_image']) * 10,
        encoder_decoder_output['noised_image'], ]
    utils.plot_examples(4, images_to_plot)
    plt.tight_layout()
    plt.show()

#################
# Plot to Disk

encoder_decoder_output = encoder_decoder(
    inputs={'cover_image': cover_images, 'message': messages},
    training=False)

fig = plt.figure(figsize=(4, 8))
images_to_plot = [
    cover_images,
    encoder_decoder_output['encoded_image'],
    (cover_images - encoder_decoder_output['encoded_image']) * 10,
    encoder_decoder_output['noised_image']]
utils.plot_examples(8, images_to_plot)
plt.tight_layout()
plt.savefig('./examples.png', bbox_inches='tight',
            pad_inches=0)


#################
# Testing

batch_size_test = 16
messages_test = create_messages(batch_size_test, msg_length)
dataset_test= create_dataset(data_test, batch_size_test)
evaluate_batches(99, dataset_test, messages_test,
                 encoder_decoder, discriminator)
"""Constructs AlexNet neural network architecture."""
import tensorflow as tf

def build_model():
    """ Build model with AlexNet architecture.

    Returns:
        Tensorflow model.
    """
    input_layer = tf.keras.layers.Input(shape=(224,224,3))

    # First convolutional layer.
    conv_1 = tf.keras.layers.Conv2D(
        filters=96,
        kernel_size=(11,11),
        strides=(4,4),
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer='zeros',
        activation='relu',

        padding='same',

        name='first_convolutional'
    )(input_layer)
    # Overlapping maxpooling layer.
    conv_1_maxpool = tf.keras.layers.MaxPooling2D(
        pool_size=(3,3),
        strides=(2,2),

        padding='same',

        name='first_convolution_maxpool'
    )(conv_1)
    # Local response normalization layer.
    conv_1_lrnorm = tf.keras.layers.Lambda(
        lr_normalization,

        name='first_convolution_lrnorm'
    )(conv_1_maxpool)

    # Second convolutional layer.
    conv_2 = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=(5, 5),
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer='ones',
        activation='relu',

        strides=(1,1),
        padding='same',

        name='second_convolutional'
    )(conv_1_lrnorm)
    # Overlapping maxpool layer.
    conv_2_maxpool = tf.keras.layers.MaxPooling2D(
        pool_size=(3,3),
        strides=(2,2),

        padding='same',

        name='second_convolution_maxpool'
    )(conv_2)
    # Local response normalization.
    conv_2_lrnorm = tf.keras.layers.Lambda(
        lr_normalization,

        name='second_convolutional_lrnorm'
    )(conv_2_maxpool)

    # Third convolutional layer.
    conv_3 = tf.keras.layers.Conv2D(
        filters=384,
        kernel_size=(3,3),
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer='ones',
        activation='relu',

        strides=(1,1),
        padding='same',

        name='third_convolutional'
    )(conv_2_lrnorm)

    # Fourth convolutional layer.
    conv_4 = tf.keras.layers.Conv2D(
        filters=384,
        kernel_size=(3,3),
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer='zeros',
        activation='relu',

        strides=(1,1),
        padding='same',

        name='fourth_convolutional'
    )(conv_3)

    # Fifth convolutional layer.
    conv_5 = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=(3,3),
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer='ones',
        activation='relu',

        strides=(1,1),
        padding='same',

        name='fifth_convolutional'
    )(conv_4)
    # Overlapping maxpool layer.
    conv_5_maxpool = tf.keras.layers.MaxPooling2D(
        pool_size=(3,3),
        strides=(2,2),

        padding='same',

        name='fifth_convolutional_maxpool'
    )(conv_5)

    flattening_layer = tf.keras.layers.Flatten()(conv_5_maxpool)

    # Dense, dropout, and output layers.
    dense_1 = tf.keras.layers.Dense(
        units=4096,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer='ones',
        activation='relu',

        name='first_dense'
    )(flattening_layer)
    dense_1_dropout = tf.keras.layers.Dropout(rate=0.5, name='first_dense_dropout')(dense_1)

    dense_2 = tf.keras.layers.Dense(
        units=4096,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer='ones',
        activation='relu',

        name='second_dense'
    )(dense_1_dropout)
    dense_2_dropout = tf.keras.layers.Dropout(rate=0.5, name='second_dense_dropout')(dense_2)

    output_layer = tf.keras.layers.Dense(
        units=1000,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        activation='softmax',

        name='output_layer'
    )(dense_2_dropout)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer, name='AlexNet')

    return model

def lr_normalization(layer_output):
    """ Local response normalization function with AlexNet specific parameters."""
    output_normalized = tf.nn.local_response_normalization(
        layer_output,
        depth_radius=5,
        bias=2,
        alpha=10e-4,
        beta=0.75
    )

    return output_normalized

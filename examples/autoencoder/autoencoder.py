import tensorflow as tf

from layers import *
from layer_utils import *

def encoder(input):
    # Create a conv network with 3 conv layers and 1 FC layer
    # input image is 28 x 28
    # Conv 1: filter: [3, 3, 1], stride: [2, 2], relu
    with tf.variable_scope('conv1') as scope:
        kernels = tf.get_variable('weights', [3, 3, 1, 1],
                                  initializer=tf.truncated_normal_initializer())
        biases = tf.get_variable('biases', [1],
                                 initializer=tf.random_normal_initializer())
        conv = tf.nn.conv2d(input, kernels, padding='SAME', strides=[1, 2, 2, 1])
        conv1 = tf.nn.relu(conv + biases, name=scope.name)
    
    # Conv 2: filter: [3, 3, 8], stride: [2, 2], relu
    # input size is 14 x 14
    with tf.variable_scope('conv2') as scope:
        kernels = tf.get_variable('weights', [3, 3, 1, 8],
                                  initializer=tf.truncated_normal_initializer())
        biases = tf.get_variable('biases', [8],
                                 initializer=tf.random_normal_initializer())
        conv = tf.nn.conv2d(conv1, kernels, padding='SAME', strides=[1, 2, 2, 1])
        conv2 = tf.nn.relu(conv + biases, name=scope.name)
    
    # Conv 3: filter: [3, 3, 8], stride: [2, 2], relu
    # input size is 7 x 7
    with tf.variable_scope('conv3') as scope:
        kernels = tf.get_variable('weights', [3, 3, 8, 8],
                                  initializer=tf.truncated_normal_initializer())
        biases = tf.get_variable('biases', [8],
                                 initializer=tf.random_normal_initializer())
        conv = tf.nn.conv2d(conv2, kernels, padding='SAME', strides=[1, 2, 2, 1])
        conv3 = tf.nn.relu(conv + biases, name=scope.name)

    # FC: output_dim: 100, no non-linearity
    with tf.variable_scope('fc') as scope:
        input_dim = 4 * 4 * 8
        output_dim = 100
        conv3 = tf.reshape(conv3, shape=[-1, input_dim])
        w = tf.get_variable('weights', [input_dim, output_dim],
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable('biases', [output_dim],
                            initializer=tf.random_normal_initializer())
        fc = tf.matmul(conv3, w) + b

    return fc

def decoder(input):
    # Create a deconv network with 1 FC layer and 3 deconv layers
    # FC: output dim: 128, relu
    with tf.variable_scope('fc_deconv') as scope:
        w = tf.get_variable('weights', [100, 4 * 4 * 8],
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable('biases', [4 * 4 * 8],
                            initializer=tf.random_normal_initializer())
        fc = tf.matmul(input, w) + b
        fc = tf.reshape(fc, shape=[-1, 4, 4, 8])

    # Reshape to [batch_size, 4, 4, 8]

    # Deconv 1: filter: [3, 3, 8], stride: [2, 2], relu
    with tf.variable_scope('deconv1') as scope:
        output_shape = get_deconv2d_output_dims([fc.get_shape()[0].value, 4, 4, 8], [3, 3, 8], [2, 2], padding='SAME')
        kernels = tf.get_variable('weights', [3, 3, 8, 8],
                                  initializer=tf.truncated_normal_initializer())
        biases = tf.get_variable('biases', [8],
                                 initializer=tf.random_normal_initializer())
        deconv = tf.nn.conv2d_transpose(fc, kernels, output_shape=output_shape,
                                        strides=[1, 2, 2, 1], padding='SAME')
        deconv1 = tf.nn.relu(deconv + biases, name=scope.name)
    
    # Deconv 2: filter: [8, 8, 1], stride: [2, 2], padding: valid, relu
    with tf.variable_scope('deconv2') as scope:
        output_shape = get_deconv2d_output_dims(output_shape, [8, 8, 1], [2, 2], padding='VALID')
        kernels = tf.get_variable('weights', [8, 8, 1, 8],
                                  initializer=tf.truncated_normal_initializer())
        biases = tf.get_variable('biases', [1],
                                 initializer=tf.random_normal_initializer())
        deconv = tf.nn.conv2d_transpose(deconv1, kernels, output_shape=output_shape,
                                        strides=[1, 2, 2, 1], padding='VALID')
        deconv2 = tf.nn.relu(deconv + biases, name=scope.name)
    
    # Deconv 3: filter: [7, 7, 1], stride: [1, 1], padding: valid, sigmoid
    with tf.variable_scope('deconv3') as scope:
        output_shape = get_deconv2d_output_dims(output_shape, [7, 7, 1], [1, 1], padding='VALID')
        kernels = tf.get_variable('weights', [7, 7, 1, 1],
                                  initializer=tf.truncated_normal_initializer())
        biasis = tf.get_variable('biases', [1],
                                 initializer=tf.random_normal_initializer())
        deconv = tf.nn.conv2d_transpose(deconv2, kernels, output_shape=output_shape,
                                        strides=[1, 1, 1, 1], padding='VALID')
        deconv3 = tf.nn.sigmoid(deconv + biases, name=scope.name)

    return deconv3

def autoencoder(input_shape):
    # Define place holder with input shape
    X = tf.placeholder(tf.float32, shape=input_shape)

    # Define variable scope for autoencoder
    with tf.variable_scope('autoencoder') as scope:
        # Pass input to encoder to obtain encoding
        encoding = encoder(X)
        
        # Pass encoding into decoder to obtain reconstructed image
        decoding = decoder(encoding)
        
        # Return input image (placeholder) and reconstructed image
        return X, decoding

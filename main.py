"""
Created by Calum Bruton, August 12, 2017

A convolutional neural network created using TensorFlow to classify, cars, bikes, and people
Trained on the the GRAZ 02 data set of cars, bikes, people images
"""

# Imports
import numpy as np
import tensorflow as tf
import subprocess
from helper_functions import *



def main():

    # A line that stops a mac from sleeping during script execution
    subprocess.Popen("caffeinate")

    # If true the a model will be pre-loaded
    LOAD_MODEL = False

    
    """======== LOAD DATASET ========"""

    # Directories
    train_path = 'training_data/images'
    test_path = 'test_data/images'
    results_path = 'results'

    #Training and Test Data
    num_train_imgs = 1064
    num_test_imgs = 187
    num_classes = 3
    img_px = 22500   #150x150

    training_data, training_labels, test_data, test_labels = load_data(train_path, test_path, num_train_imgs,
                                                                       num_test_imgs, num_classes, img_px)                                                                       

    display_image([300,200], training_data)

    """======== CREATE MODEL ========"""

    # Model Features
    input_nodes = 22500
    hidden_nodes = 5000
    drop_rate = 0.4

    
    # Inputs
    x = tf.placeholder('float', [None, img_px])

    # Reshape input to a 4D tensor 
    input_layer = tf.reshape(x, [-1, 150, 150, 1])

    # Convolutional layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[3, 3],
        padding="valid",
        activation=tf.nn.relu)

    # Pooling layer #2
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=32,
      kernel_size=[3, 3],
      padding="valid",
      activation=tf.nn.relu)

    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Convolutional Layer #3 
    conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=32,
      kernel_size=[3, 3],
      padding="valid",
      activation=tf.nn.relu)

    # Pooling Layer #3
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    # Convolutional Layer #4
    conv4 = tf.layers.conv2d(
      inputs=pool3,
      filters=64,
      kernel_size=[2, 2],
      padding="valid",
      activation=tf.nn.relu)

    # Pooling Layer #4
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool4_flat = tf.reshape(pool4, [-1, 8*8*64])
    dense = tf.layers.dense(inputs=pool4_flat, units=hidden_nodes, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=drop_rate)

    # Logits Layer
    pred = tf.layers.dense(inputs=dropout, units=num_classes)


    # The placeholder for labels of the classes - i.e what the correct class is for training
    y_ = tf.placeholder(tf.float32, [None, num_classes])



    """======== TRAIN MODEL ========"""

    # Parameters
    hm_epochs = 20
    batch_size = 100

    
    # Define loss function and optimizer
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y_) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Create a session and initialize variables
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()

    # Load Trained Model
    if LOAD_MODEL:
        #restore saved model
        saver.restore(sess, results_path + '/trained_convolutional_NN')

    
    for epoch in range(hm_epochs):
        epoch_loss = 0
        for _ in range(int(num_train_imgs/batch_size)):
            epoch_x, epoch_y = next_batch(batch_size, training_data, training_labels)
            _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y_: epoch_y})
            epoch_loss += c

        # Print epoch, loss, and accuracy
        print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:test_data, y_:test_labels})*100, "%\n========================")

        # Save predictions and labels to a csv file for analysis
        prediction = tf.argmax(pred,1)
        p =  prediction.eval(feed_dict={x: test_data}, session=sess)
        np.savetxt(results_path + "/results.csv", np.r_['0,2',np.argmax(test_labels, axis=1), p], delimiter=",")

    # Save the trained model 
    saver.save(sess, results_path + "/trained_convolutional_NN")




if __name__ == '__main__':
    main()

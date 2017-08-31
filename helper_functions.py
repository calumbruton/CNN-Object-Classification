"""
Fucntions to assist in CNN training and testing
"""


# Imports
import numpy as np
import tensorflow as tf
import scipy.ndimage
import os
import scipy.misc as smp



def next_batch(num, data, labels):
    """ Return a total of `num` random samples and labels. """
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

    num_train_imgs = 1064
    num_test_imgs = 187
    num_classes = 3
    image_pixels = 22500            #150x150


def display_image(indices, data):
    """display_image takes an array of inices and a set of flattened pixel data and uses it
    to render the images in each index of the data. used for debugging and experimentation."""
    for i in indices:
        im_data = np.reshape(data[i],[150,150])
        img = smp.toimage(im_data)      # Create a PIL image
        img.show()                      # View in default viewer
                                                                       
def load_data(train_path, test_path, num_train_imgs, num_test_imgs, num_classes, image_px):
    """Loads the training and test data images. Takes the directory of the training images and test images as a parameter.
    Also takes the number of training images, the number of test images, the number of classes, and the pixels in each image.
    Returns numpy array with the flattened pixel image data and the one hot encoded label data """
                                                                   
    training_data = np.zeros((num_train_imgs,image_px))
    training_labels = np.zeros((num_train_imgs,num_classes))

    test_data = np.zeros((num_test_imgs,image_px))
    test_labels = np.zeros((num_test_imgs,num_classes))

    # Classes and one hot encoding
    classes = ['bikes', 'cars', 'persons']
    bike = [1,0,0]
    car = [0,1,0]
    person = [0,0,1]

    # Index counters
    train_index = 0
    test_index = 0
        
    # for each folder of images in the training data
    for folder in os.listdir(train_path):
        if not folder.startswith('.'):
            
            # for each image in each folder
            for file in os.listdir(train_path + "/" + folder):
                if not file.startswith('.'):
                    # create a grey scale numpy array of the image 
                    image = scipy.ndimage.imread(train_path + "/" + folder + "/" + file, True)

                    # Make the image a one dimensional vector
                    image = image.flatten()

                    # Add the pixel data to the next column of the training_data numpy array
                    training_data[train_index] = image
                
                    # Store the classification of the image
                    class_ind = classes.index(folder)
                    training_labels[train_index] = [bike,car,person][class_ind]

                    # increase the index position
                    train_index += 1
                    
            print("Training image folder of", folder, "succesfully loaded")


           
    # Save the data using np save so this can be loaded without this script
    # np.save('training_data/pixel-data', training_data)
    # np.savetxt("training_data/pixel-data.csv", training_data[1], delimiter=",")
    # np.savetxt("training_data/classification-data.csv", training_labels, delimiter=",")

    # for each folder of images in the test data
    for folder in os.listdir(test_path):
        if not folder.startswith('.'):
            
            # for each image in each folder
            for file in os.listdir(test_path + "/" + folder):
                if not file.startswith('.'):
                    # create a grey scale numpy array of the image 
                    image = scipy.ndimage.imread(test_path + "/" + folder + "/" + file, True)

                    # Make the image a one dimensional vector
                    image = image.flatten()
                    
                    # Add the pixel data to the next column of the training_data numpy array
                    test_data[test_index] = image
                
                    # Store the classification of the image
                    class_ind = classes.index(folder)
                    test_labels[test_index] = [bike,car,person][class_ind]

                    # increase the index position
                    test_index += 1
            
            print("Test image folder of", folder, "succesfully loaded")

    return training_data, training_labels, test_data, test_labels
    

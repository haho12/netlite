# reference: https://www.cs.toronto.edu/~kriz/
import numpy as np
import matplotlib.pyplot as plt

import os.path
import tarfile 
import urllib.request
import pickle

def load_train(input_path = 'cifar10_raw', num_images = 50000):
    images = np.zeros((num_images, 32, 32, 3), dtype=np.float32)
    labels = np.zeros(num_images, dtype=np.int32)
    num_batches = int(np.ceil(num_images / 10000))
    for i in range(num_batches):
        images_file = 'data_batch_' + str(i+1)
        if i == num_batches - 1:
            num_images_i = num_images - i * 10000
        else:
            num_images_i = 10000
        images_i, labels_i = load(input_path, images_file, num_images_i)
        
        images[i*10000:(i+1)*10000,:,:,:] = images_i
        labels[i*10000:(i+1)*10000]       = labels_i
        
    return images, labels

def load_valid(input_path = 'cifar10_raw', num_images = 10000):
    images_file = 'test_batch'
    return load(input_path, images_file, num_images)

def download(download_dir, filename):
    if not os.path.isdir(download_dir):
        os.mkdir(download_dir)
    url = "https://www.cs.toronto.edu/~kriz/" + filename
    targetfile = os.path.join(download_dir, filename)
    print("Downloading: " + filename + " ...")
    urllib.request.urlretrieve(url, targetfile)

def extract(download_dir, filename):
    targetfile = os.path.join(download_dir, filename)
    tar_file = tarfile.open(targetfile) 
    tar_file.extractall(download_dir)
    tar_file.close() 

def load(input_path, images_file, num_images_max):
    gz_filename = 'cifar-10-python.tar.gz'
    gz_path = os.path.join(input_path, gz_filename)
    raw_path = os.path.join(input_path, 'cifar-10-batches-py', images_file)

    if not os.path.isfile(gz_path):
        download(input_path, gz_filename)
    if not os.path.isfile(raw_path):
        extract(input_path, gz_filename)

    with open(raw_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')


    data   = dict[b'data']
    labels = np.array(dict[b'labels'])
    num_images = len(labels) # Todo: currently only loading batch 1 of 5
    num_images_max = min(num_images, num_images_max)

    images = np.zeros((num_images_max, 32, 32, 3), dtype=np.float32)
    for i in range(num_images_max):
        data_np = data[i].reshape(3,32,32) / 255.

        images[i,:,:,0] = data_np[0,:,:]
        images[i,:,:,1] = data_np[1,:,:]
        images[i,:,:,2] = data_np[2,:,:]

    return images, labels[:num_images_max]

def show_images(images, labels):
    cols = 5
    rows = 2

    plt.figure(figsize=(10,7))

    class_names = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')     
    for i in range(cols*rows):
        plt.subplot(rows, cols, i+1)        
        plt.imshow(images[i,:,:,:])
        plt.title("label: " + str(labels[i]) + " (" + class_names[labels[i]] + ")")
        plt.axis('off')

if __name__ == '__main__':
    images, labels = load_train()
    show_images(images, labels)

    #count = 0
    #for i in range(images.shape[0]):
    #    if labels[i] == 7: # horse
    #        plt.imsave("cifar10_horses\\horse_" + str(count) + '.png', images[i,:,:,:])
    #        count += 1
    #        if count >= 10:
    #            break
            
 
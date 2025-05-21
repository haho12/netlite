# reference: http://yann.lecun.com/exdb/mnist/
import numpy as np
import matplotlib.pyplot as plt

import os.path
import gzip
import urllib.request

def load_train(input_path = 'mnist_raw', num_images = 60000):
    images_file = 'train-images-idx3-ubyte.gz'
    labels_file = 'train-labels-idx1-ubyte.gz'
    return load(input_path, images_file, labels_file, num_images)

def load_valid(input_path = 'mnist_raw', num_images = 10000):
    images_file = 't10k-images-idx3-ubyte.gz'
    labels_file = 't10k-labels-idx1-ubyte.gz'
    return load(input_path, images_file, labels_file, num_images)

def download(download_dir, filename):
    if not os.path.isdir(download_dir):
        os.mkdir(download_dir)
    urls = ["http://yann.lecun.com/exdb/mnist/",
            "https://huggingface.co/datasets/dmitva/the-mnist-database/blob/main/",
            "https://wuecampus.uni-wuerzburg.de/moodle/pluginfile.php/4698779/mod_folder/content/0/",
            "https://huggingface.co/spaces/chrisjay/mnist-adversarial/raw/603879aac618aca69749a8a9172daec23a9dd2c4/files/MNIST/raw/",
           ] 
    target_file = os.path.join(download_dir, filename)
    print("Downloading '" + filename + "' : ")
    for url in urls:
        source_file = url + filename
        try:
            #print(f"Attempting to download: {source_file}")
            urllib.request.urlretrieve(source_file, target_file)
            #print(f"Successfully downloaded: {target_file}")
            print('+')
            break
        except Exception as e:
            #print(f"Failed to download {source_file}: {e}")
            print('-')
            continue

def load(input_path, images_file, labels_file, num_images_max):
    images_path = os.path.join(input_path, images_file)
    labels_path = os.path.join(input_path, labels_file)
    if not os.path.isfile(images_path):
        download(input_path, images_file)
    if not os.path.isfile(labels_path):
        download(input_path, labels_file)

    images_byte = gzip.open(images_path, 'r')
    labels_byte = gzip.open(labels_path, 'r')

    # read images
    magic_num  = int.from_bytes(images_byte.read(4),  byteorder='big')
    num_images = int.from_bytes(images_byte.read(4),  byteorder='big')
    img_height = int.from_bytes(images_byte.read(4),  byteorder='big')
    img_width  = int.from_bytes(images_byte.read(4),  byteorder='big')
    assert(magic_num == 2051)
    assert(img_height == 28)
    assert(img_width == 28)
    
    num_images_max = min(num_images, num_images_max)

    images = np.zeros((num_images_max, 32, 32, 1), dtype=np.float32)
    for i in range(num_images_max):
        data_raw = images_byte.read(img_height * img_width)
        data_np  = np.frombuffer(data_raw, dtype=np.uint8)
        images[i,2:-2,2:-2,0] = data_np.reshape(28,28) / 255.

    # read labels
    magic_num  = int.from_bytes(labels_byte.read(4),  byteorder='big')
    num_labels = int.from_bytes(labels_byte.read(4),  byteorder='big')
    assert(magic_num == 2049)
    assert(num_labels == num_images)

    buf = labels_byte.read(num_images_max)    
    labels = np.frombuffer(buf, dtype=np.uint8)

    return images, labels

def show_images(images, labels):
    cols = 5
    rows = 2

    plt.figure(figsize=(10,7))
     
    for i in range(cols*rows):
        plt.subplot(rows, cols, i+1)        
        plt.imshow(images[i,:,:,0], cmap="gray")
        plt.title("label: " + str(labels[i]))
        plt.axis('off')

if __name__ == '__main__':
    images, labels = load_train()
    show_images(images, labels)

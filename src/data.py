from src.utils import data_config
from pathlib import Path
import numpy as np
import zipfile
import random
import h5py
    
def prepare_points(tensor, data_config, threshold = False):
    if threshold:
        tensor = np.where(
            tensor > data_config.threshold, 
            data_config.lower, 
            data_config.upper
        )
    tensor = tensor.reshape((
            tensor.shape[0], 
            data_config.y_shape, 
            data_config.x_shape, 
            data_config.z_shape
        ))
    return tensor

def get_random_colored_images(images, seed = 0x000000, color='random'):
    np.random.seed(seed)
    size = images.shape[0]
    
    # here we use HSV representation
    colored_images = []
    Hues_arr = np.array([60, 120, 240, 360])
    if color == 'random':
        #hues = 360*np.random.rand(size)
        hues = np.random.choice(Hues_arr, size)
    elif color == 'red':
        hues = 360*np.ones(size)
    elif color == 'blue':
        hues = 240*np.ones(size)
    elif color == 'green':
        hues = 120*np.ones(size)
    elif color == 'yellow':
        hues = 60*np.ones(size)      
        
    for V, H in zip(images, hues):
      
        colored_image = np.zeros((3, V.shape[0], V.shape[1], V.shape[2]))
        H_i = round(H/60) % 6
        colored_image[0] = V
        colored_image[1] = V
        colored_image[2] = V
        for x in range(data_config.x_shape):
            for y in range(data_config.y_shape):
                for z in range(data_config.z_shape):
                    if H_i == 0:              # red
                        colored_image[0, x, y, z] = int(V[x, y, z] > 0) * 255
                        colored_image[1, x, y, z] = 0
                        colored_image[2, x, y, z] = 0
                    elif H_i == 1:            # yellow
                        colored_image[0, x, y, z] = int(V[x, y, z] > 0) * 255
                        colored_image[1, x, y, z] = int(V[x, y, z] > 0) * 255
                        colored_image[2, x, y, z] = 0
                    elif H_i == 2:            # green
                        colored_image[0, x, y, z] = 0
                        colored_image[1, x, y, z] = int(V[x , y, z] > 0) * 255
                        colored_image[2, x, y, z] = 0
                    elif H_i == 3:
                        colored_image[0, x, y, z] = 0
                        colored_image[1, x, y, z] = 0
                        colored_image[2, x, y, z] = 0
                    elif H_i == 4:            # blue
                        colored_image[0, x, y, z] = 0
                        colored_image[1, x, y, z] = 0
                        colored_image[2, x, y, z] = int(V[x , y, z] > 0) * 255
                    elif H_i == 5:
                        colored_image[0, x, y, z] = 0
                        colored_image[1, x, y, z] = 0
                        colored_image[2, x, y, z] = 0

        colored_images.append(colored_image)
    
    colored_images = np.stack(colored_images, axis = 0)
    
    return colored_images
       
def get_dataset():
    X_train_A = []
    X_train_B = []
    y_train_A = []
    y_train_B = []
    X_test = []
    y_test = []
    with h5py.File('data/full_dataset_vectors.h5', 'r') as dataset:
        for i in range(dataset["y_train"].shape[0]):
            if dataset["y_train"][i] == 3:
                X_train_A.append(dataset["X_train"][i])
                y_train_A.append(dataset["y_train"][i])
            elif dataset["y_train"][i] == 5:
                X_train_B.append(dataset["X_train"][i])
                y_train_B.append(dataset["y_train"][i])

        for i in range(dataset["y_test"].shape[0]):
            if dataset["y_test"][i] == 3:
                X_test.append(dataset["X_test"][i])
                y_test.append(dataset["y_test"][i])

    X_train_A = np.array(X_train_A)
    X_train_B = np.array(X_train_B)
    y_train_A = np.array(y_train_A)
    y_train_B = np.array(y_train_B)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    X_Shape_train_A = prepare_points(X_train_A, data_config, threshold = False)
    X_Shape_train_B = prepare_points(X_train_B, data_config, threshold = False)   
    X_Shape_test = prepare_points(X_test, data_config, threshold = False)
    
    X_RGB_train_A = get_random_colored_images(X_Shape_train_A, color='random')
    X_RGB_train_B = get_random_colored_images(X_Shape_train_B, color='random')
    X_RGB_test = get_random_colored_images(X_Shape_test, color='random')
    
    
    print("Train A shape:", X_RGB_train_A.shape)
    print("Train B shape:", X_RGB_train_B.shape)
    print("Test shape:", X_RGB_test.shape)
    
    return X_RGB_train_A, X_RGB_train_B, X_RGB_test

def save_data(name, arr):
    with open(name, "wb") as f:
        np.save(f, arr)
        
def load_data(name):
    return np.load(name)

if __name__ == "__main__":
    X_RGB_train_A, X_RGB_train_B, X_RGB_test = get_dataset()

    save_data("../data/x_train_a.npy", X_RGB_train_A)
    save_data("../data/x_train_b.npy", X_RGB_train_B)
    save_data("../data/x_test.npy", X_RGB_test)
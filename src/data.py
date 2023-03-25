import numpy as np
import random
import h5py
from src.utils import data_config
from src.coloring import get_random_colored_images

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

def get_dataset(path='../data/full_dataset_vectors.h5', colored=True):
    with h5py.File(path, 'r') as dataset:
        print((dataset["y_train"][:] == 3).squeeze().shape)
        X_train_A = dataset["X_train"][np.where(dataset["y_train"][:] == 3)]
        X_train_B = dataset["X_train"][np.where(dataset["y_train"][:] == 5)]
        X_test = dataset["X_test"][np.where(dataset["y_test"][:] == 3)]
    
    X_RGB_train_A = prepare_points(X_train_A, data_config, threshold = False)
    X_RGB_train_B = prepare_points(X_train_B, data_config, threshold = False)   
    X_RGB_test = prepare_points(X_test, data_config, threshold = False)

    if colored:
        X_RGB_train_A = get_random_colored_images(X_RGB_train_A, color='random')
        X_RGB_train_B = get_random_colored_images(X_RGB_train_B, color='random')
        X_RGB_test = get_random_colored_images(X_RGB_test, color='random')
    
    
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
    X_train_A, X_train_B, X_test = get_dataset()

    save_data("../data/x_train_a.npy", X_train_A)
    save_data("../data/x_train_b.npy", X_train_B)
    save_data("../data/x_test.npy", X_test)

    X_RGB_train_A, X_RGB_train_B, X_RGB_test = get_dataset(colored=True)
    save_data("../data/x_train_a_c.npy", X_RGB_train_A)
    save_data("../data/x_train_b_c.npy", X_RGB_train_B)
    save_data("../data/x_test_c.npy", X_RGB_test)
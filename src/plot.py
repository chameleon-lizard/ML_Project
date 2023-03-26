from src.utils import data_config
from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import torch

def plot_img_3D(image, plot_only_figure=True):
    color = np.zeros(3)
    plot_data = []
    for x in range(data_config.x_shape):
        for y in range(data_config.y_shape):
            for z in range(data_config.z_shape):
                isVoxel = (image[0,x, y, z] > 0) or (image[1,x, y, z] > 0) or (image[2,x, y, z] > 0)
                color[0] = int(image[0, x, y, z]*255)
                color[1] = int(image[1, x, y, z]*255)
                color[2] = int(image[2, x, y, z]*255)
    
                plot_data.append([x, y, z, isVoxel, f'rgb({color[0]}, {color[1]}, {color[2]})'])
    plot_df = pd.DataFrame(plot_data, columns=["x", "y", "z", "isVoxel", "color"])

    if plot_only_figure:
        plot_df = plot_df.loc[plot_df["isVoxel"] == True]

    fig = go.Figure(data=[go.Scatter3d(x=plot_df['x'], y=plot_df['y'], z=plot_df['z'], 
                                       mode='markers',
                                       marker=dict(
                                       color = plot_df["color"],
                                       size=4,       
                                       colorscale='Viridis',
                                       opacity= 0.8 ))])
    fig.show()

def plot_img_3D_gray(image, plot_only_figure=True):
    plot_data = []
    for x in range(data_config.x_shape):
        for y in range(data_config.y_shape):
            for z in range(data_config.z_shape):
                isVoxel = image[x, y, z] > 0
                plot_data.append([x, y, z, isVoxel])
    plot_df = pd.DataFrame(plot_data, columns=["x", "y", "z", "isVoxel"])

    if plot_only_figure:
        plot_df = plot_df.loc[plot_df["isVoxel"] == True]

    fig = go.Figure(data=[go.Scatter3d(x=plot_df['x'], y=plot_df['y'], z=plot_df['z'], 
                                       mode='markers',
                                       marker=dict(
                                       size=4,       
                                       colorscale='Viridis',
                                       opacity= 0.8 ))])
    fig.show()

def plot_img_gray(img):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    
    ax.voxels(img[0])
    plt.show()

def plot_img(img):
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    color = np.array([rgb2hex(rgb) for rgb in img.numpy().reshape(3, -1).T])
    color = color.reshape(*img.shape[1:])
    
    if torch.any(img[0, ...]):
        img = img[0, ...]
    elif torch.any(img[1, ...]):
        img = img[1, ...]
    else:
        img = img[2, ...]
    
    ax.voxels(img, facecolors=color)
    plt.show()

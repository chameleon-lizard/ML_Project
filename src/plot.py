from src.utils import data_config
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import pandas as pd
import numpy as np

def plot_img2(image, plot_only_figure=True):
    for x in range(data_config.x_shape):
        for y in range(data_config.y_shape):
            for z in range(data_config.z_shape):
                # find first voxel and get it's color
                if (image[0, x, y, z] > 0):
                    plot_img_color = np.squeeze(image[:, x, y, z])
                    break
                if (image[1, x, y, z] > 0):
                    plot_img_color = np.squeeze(image[:, x, y, z])
                    break
                if (image[2, x, y, z] > 0):
                    plot_img_color = np.squeeze(image[:, x, y, z])
                    break
    # 0 - red, 1 - yellow and green, 2 - blue
    if plot_img_color[0] == 255:
        color_ind = 0
    elif plot_img_color[1] == 255:
        color_ind = 1
    elif plot_img_color[2] == 255:
        color_ind = 2
    else:
        raise ValueError("Incorrect color encountered.")
        
    plot_img_3d = image[color_ind]
    plot_label = "3"
    color = np.zeros(3)
    plot_data = []
    for x in range(data_config.x_shape):
        for y in range(data_config.y_shape):
            for z in range(data_config.z_shape):
                #val = int(plot_img_3d[x, y, z] * 255)
                isVoxel = (plot_img_3d[x, y, z] > 0)
                if isVoxel:                   
                    color[0] = int(image[0, x, y, z])
                    color[1] = int(image[1, x, y, z])
                    color[2] = int(image[2, x, y, z])
                else:
                    color[0] = 255
                    color[1] = 255
                    color[2] = 255             
                plot_data.append([x, y, z, isVoxel, f'rgb({color[0]}, {color[1]}, {color[2]})'])
    plot_df = pd.DataFrame(plot_data, columns=["x", "y", "z", "isVoxel", "color"])
    if plot_only_figure:
        plot_df = plot_df.loc[plot_df["isVoxel"] == True]

    fig = go.Figure(data=[go.Scatter3d(x=plot_df['x'], y=plot_df['y'], z=plot_df['z'], 
                                       mode='markers',
                                       text=f"current label: {plot_label}",
                                       marker=dict(
                                       color = plot_df["color"],
                                       size=4,       
                                       colorscale='Viridis',
                                       opacity= 0.8 ))])
    fig.show()
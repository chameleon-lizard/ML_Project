# Unpaired learning with Optimal Transport application to 3D MNIST images

Computational Optimal Transport (OT) is a set of tools, which provide a way of transforming one distribution into another with the minimal effort. Recently scalable neural OT based methods, which do not require paired training dataset, have been developed and applied to a wide range of tasks, including transfer learning, generative modeling and super-resolution.

In this project, we are to implement and train OT approach on 3D MNIST digits dataset, analyze the applicability of OT method to 3D image-to-image translation. Specifically we want to learn a model to transform the 3D digits '3' into 3D digits 5. 

## Presentations
[Project presentation](http://) - official report on this project by our team.

## Related repositories

 - [Repository](https://github.com/iamalexkorotin/NeuralOptimalTransport) for [Neural Optimal Transform](https://arxiv.org/abs/2201.12220)

## Experiments


## Repository structure
``` bash

ML_Project
├── src 
│   ├── data.py  # dataset preprocessing, paints 3D MNIST digits
│   ├── dataset.py # dataloader
│   ├── nn_gray.py # models for the first approach and their auxiliary functions
│   ├── nn.py # models for the second approach
│   ├── plot.py # contains functions for plotting
│   └── utils.py # configuration file
├── README.md
└── project.ipynb # main file for running experiments
```

## Datasets

[3D MNIST dataset](https://www.kaggle.com/datasets/daavoo/3d-mnist) - the dataset should is pre-processed with ```src/data.py```
The dataloader can be found in ```scr/dataset.py```





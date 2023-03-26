import numpy as np
import torch

def get_random_colored_images(images, seed = 42, color='random'):
    np.random.seed(seed)
    
    # here we use RGB representation of red, green, blue and yellow
    colors = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0))
    colored_images = []
    
    for img in images:
        if color == 'random':
            col = colors[np.random.randint(0, 4)]
        elif color == 'red':
            col = colors[0]
        elif color == 'blue':
            col = colors[2]
        elif color == 'green':
            col = colors[1]
        elif color == 'yellow':
            col = colors[3]
   
        colored_image = np.stack((img, img, img))
        for i in range(3):
            colored_image[i] *= col[i]
        colored_images.append(colored_image)
    return np.asarray(colored_images)

def random_color(im):
    hue = np.random.choice([60, 120, 240, 280])
    d = (im *(hue%60)/60)
    im_min, im_inc, im_dec = torch.zeros_like(im), d, im - d
    c_im = torch.zeros((3, im.shape[1], im.shape[2]))
    H = round(hue/60) % 6    
    cmap = [[0, 3, 2], [2, 0, 3], [1, 0, 3], [1, 2, 0], [3, 1, 0], [0, 1, 2]]
    return torch.cat((im, im_min, im_dec, im_inc), dim=0)[cmap[H]]

def add_dimension(im):
    img_3D = torch.zeros(im.shape[0], im.shape[1], im.shape[2], 16)
    for i in range(im.shape[1]):
      if i < 4 or i > 12:
        img_3D[:, :, :, i] = torch.zeros_like(im)
      else:
        img_3D[:, :, :, i] = im
    return img_3D 
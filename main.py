import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

import tensorflow as tf

from PIL import Image
import os
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('encoder', help='Encoder model file')
parser.add_argument('decoder', help='Decoder model file')
parser.add_argument('dataset', help='Dataset path')

parser.add_argument('--dims', help='Latent space dimensions', default=16, type=int)
parser.add_argument('--size', help='Image size', default=128, type=int)
args = parser.parse_args()

encoder = tf.keras.models.load_model(args.encoder)
decoder = tf.keras.models.load_model(args.decoder)

def predict(dim):
    return decoder.predict(np.array([dim])).reshape((args.size, args.size)) * 10

def lspace(dim):
    path = args.dataset
    data = []

    for img in tqdm(os.listdir(path), desc="Loading dataset"):
        data.append(np.array(Image.open(os.path.join(path, img)).resize((args.size, args.size))))
    data = np.average(np.array(data).astype("float32") / 255.0, axis=3)
    x_test = np.reshape(data, newshape=(data.shape[0], np.prod(data.shape[1:])))
    del data
    
    space = []
    for i in tqdm(range(x_test.shape[0]), desc="Encoding"):
        space.append(encoder.predict(np.array([x_test[i]])))
        
    return np.array(space).reshape((x_test.shape[0], dim))

latent_space = lspace(args.dims)

fig, ax = plt.subplots(2, 2, figsize=(128,128))

global_dim = np.zeros(args.dims)
curr_dim = [0, 1]

ax[0, 0].set_title("Latent space")
ax[0, 0].scatter(latent_space[:, 0], latent_space[:, 1], s=1)
ax[0, 0].scatter([[global_dim[curr_dim[0]]]], [[global_dim[curr_dim[1]]]], s=2, c='r')

ax[0, 1].set_title("Prediction")
img = ax[0, 1].imshow(np.zeros((128, 128)), cmap='gray')
cb = plt.colorbar(img, ax=ax[0, 1])

ax[1, 0].set_title("Cooridnates")
ax[1, 0].bar(np.arange(args.dims), global_dim)

ax[1, 1].remove()

dimx_ax = fig.add_axes([0.6, 0.35, 0.3, 0.1])
dimx = Slider(
    ax=dimx_ax,
    label='Dimention x',
    valmin=0,
    valmax=args.dims-1,
    valinit=0,
    valstep=1,
)

dimy_ax = fig.add_axes([0.6, 0.25, 0.3, 0.1])
dimy = Slider(
    ax=dimy_ax,
    label='Dimention y',
    valmin=0,
    valmax=args.dims-1,
    valinit=1,
    valstep=1,
)

def update(val):
    global curr_dim
    curr_dim = [dimx.val, dimy.val]
    ax[0, 0].clear()
    ax[0, 0].set_title("Latent space")
    ax[0, 0].scatter(latent_space[:, curr_dim[0]], latent_space[:, curr_dim[1]], s=1)
    ax[0, 0].scatter([[global_dim[curr_dim[0]]]], [[global_dim[curr_dim[1]]]], s=2, c='r')

dimx.on_changed(update)
dimy.on_changed(update)

def onclick(event):
    global flag, cb, curr_dim
    ix, iy = event.xdata, event.ydata
    
    if not event.inaxes in [ax[0, 0]]:
        return
    
    global_dim[curr_dim[0]] = ix
    global_dim[curr_dim[1]] = iy
    
    ax[0, 1].clear()
    cb.remove()
    
    pred = predict(global_dim)
    img = ax[0, 1].imshow(pred, cmap='gray')
    cb = plt.colorbar(img, ax=ax[0, 1])
    
    ax[1, 0].clear()
    ax[1, 0].bar(np.arange(args.dims), global_dim)
    
    ax[0, 0].clear()
    ax[0, 0].set_title("Latent space")
    ax[0, 0].scatter(latent_space[:, curr_dim[0]], latent_space[:, curr_dim[1]], s=1)
    ax[0, 0].scatter([[global_dim[curr_dim[0]]]], [[global_dim[curr_dim[1]]]], s=2, c='r')
    
    ax[0, 0].set_title("Latent space")
    ax[0, 1].set_title("Prediction")
    ax[1, 0].set_title("Cooridnates")
    
    plt.draw()

cid = fig.canvas.mpl_connect('button_press_event', onclick)

ax_random = plt.axes([0.6, 0.15, 0.3, 0.1])
random_button = Button(ax_random, 'Randomize', hovercolor='0.975')
def random(event):
    global global_dim, cb
    global_dim = np.random.rand(args.dims)
    
    ax[0, 1].clear()
    cb.remove()
    
    pred = predict(global_dim)
    img = ax[0, 1].imshow(pred, cmap='gray')
    cb = plt.colorbar(img, ax=ax[0, 1])
    
    ax[1, 0].clear()
    ax[1, 0].bar(np.arange(args.dims), global_dim)
    
    ax[0, 0].clear()
    ax[0, 0].set_title("Latent space")
    ax[0, 0].scatter(latent_space[:, curr_dim[0]], latent_space[:, curr_dim[1]], s=1)
    ax[0, 0].scatter([[global_dim[curr_dim[0]]]], [[global_dim[curr_dim[1]]]], s=2, c='r')
    
    ax[0, 0].set_title("Latent space")
    ax[0, 1].set_title("Prediction")
    ax[1, 0].set_title("Cooridnates")
    
    plt.draw()
    
random_button.on_clicked(random)

ax_reset = plt.axes([0.6, 0.05, 0.3, 0.1])
reset_button = Button(ax_reset, 'Reset', hovercolor='0.975')
def reset(event):
    global global_dim, cb
    global_dim = np.zeros(args.dims)
    
    ax[0, 1].clear()
    cb.remove()
    
    pred = predict(global_dim)
    img = ax[0, 1].imshow(pred, cmap='gray')
    cb = plt.colorbar(img, ax=ax[0, 1])
    
    ax[1, 0].clear()
    ax[1, 0].bar(np.arange(args.dims), global_dim)
    
    ax[0, 0].clear()
    ax[0, 0].set_title("Latent space")
    ax[0, 0].scatter(latent_space[:, curr_dim[0]], latent_space[:, curr_dim[1]], s=1)
    ax[0, 0].scatter([[global_dim[curr_dim[0]]]], [[global_dim[curr_dim[1]]]], s=2, c='r')
    
    ax[0, 0].set_title("Latent space")
    ax[0, 1].set_title("Prediction")
    ax[1, 0].set_title("Cooridnates")
    
    plt.draw()

reset_button.on_clicked(reset)

reset(0)

plt.show()
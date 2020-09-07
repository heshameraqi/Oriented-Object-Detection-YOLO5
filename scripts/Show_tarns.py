from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from skimage.transform import resize

data = ListDataset("../val.txt", "../../Test_Resized", val=True)
classes = load_classes("/media/user/76E6B044E6B00701/OmarElezaby/data/classes.txt")
print(len(classes))

def visualize_data_image(imgs, targets):
    image = imgs
    labels = targets

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.collections as collections
    from matplotlib.path import Path
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(image)

    p1_x = labels[:, 1] + labels[:, 4] * np.cos(np.radians(labels[:, 5])) / 2.0 + \
           labels[:, 3] * np.cos(np.radians(90 + labels[:, 5])) / 2.0
    p1_y = labels[:, 2] - labels[:, 4] * np.sin(np.radians(labels[:, 5])) / 2.0 - \
           labels[:, 3] * np.sin(np.radians(90 + labels[:, 5])) / 2.0

    p2_x = labels[:, 1] - labels[:, 4] * np.cos(np.radians(labels[:, 5])) / 2.0 + \
           labels[:, 3] * np.cos(np.radians(90 + labels[:, 5])) / 2.0
    p2_y = labels[:, 2] + labels[:, 4] * np.sin(np.radians(labels[:, 5])) / 2.0 - \
           labels[:, 3] * np.sin(np.radians(90 + labels[:, 5])) / 2.0

    p3_x = labels[:, 1] - labels[:, 4] * np.cos(np.radians(labels[:, 5])) / 2.0 - \
           labels[:, 3] * np.cos(np.radians(90 + labels[:, 5])) / 2.0
    p3_y = labels[:, 2] + labels[:, 4] * np.sin(np.radians(labels[:, 5])) / 2.0 + \
           labels[:, 3] * np.sin(np.radians(90 + labels[:, 5])) / 2.0

    p4_x = labels[:, 1] + labels[:, 4] * np.cos(np.radians(labels[:, 5])) / 2.0 - \
           labels[:, 3] * np.cos(np.radians(90 + labels[:, 5])) / 2.0
    p4_y = labels[:, 2] - labels[:, 4] * np.sin(np.radians(labels[:, 5])) / 2.0 + \
           labels[:, 3] * np.sin(np.radians(90 + labels[:, 5])) / 2.0

    patches = []
    for i in range(labels.shape[0]):
        if not np.any(labels[i]):  # objects in image finished before max_objects
            break
        verts = [(p1_x[i], p1_y[i]), (p2_x[i], p2_y[i]), (p3_x[i], p3_y[i]), (p4_x[i], p4_y[i]), (0., 0.), ]
        codes = [Path.MOVETO,        Path.LINETO,        Path.LINETO,        Path.LINETO,        Path.CLOSEPOLY, ]
        path = Path(verts, codes)
        patches.append(mpl.patches.PathPatch(path, linewidth=1, edgecolor='r', facecolor='none'))
        ax.text(verts[0][0], verts[0][1], classes[int(labels[i][0])], fontsize=6,
                            bbox=dict(edgecolor='none', facecolor='white', alpha=0.8, pad=0.))
    ax.add_collection(collections.PatchCollection(patches, match_original=True))
    # plt.show(block=False)
    plt.show()


def visualize_data_batch(imgs, targets):
    for sample_id in range(imgs.shape[0]):
        image = np.transpose(imgs[sample_id].numpy(), (1, 2, 0))
        labels = targets[sample_id].numpy()

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.collections as collections
        from matplotlib.path import Path
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.imshow(image)

        # denormalize x,y
        labels[:, 1] *= image.shape[0]
        labels[:, 2] *= image.shape[1]

        # denormalize w,l
        diagonal_length = np.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2)
        labels[:, 3] *= diagonal_length
        labels[:, 4] *= diagonal_length

        # denormalize theta
        labels[:, 5] *= 180.

        p1_x = labels[:, 1] + labels[:, 4] * np.cos(np.radians(labels[:, 5])) / 2.0 + \
               labels[:, 3] * np.cos(np.radians(90 + labels[:, 5])) / 2.0
        p1_y = labels[:, 2] - labels[:, 4] * np.sin(np.radians(labels[:, 5])) / 2.0 - \
               labels[:, 3] * np.sin(np.radians(90 + labels[:, 5])) / 2.0

        p2_x = labels[:, 1] - labels[:, 4] * np.cos(np.radians(labels[:, 5])) / 2.0 + \
               labels[:, 3] * np.cos(np.radians(90 + labels[:, 5])) / 2.0
        p2_y = labels[:, 2] + labels[:, 4] * np.sin(np.radians(labels[:, 5])) / 2.0 - \
               labels[:, 3] * np.sin(np.radians(90 + labels[:, 5])) / 2.0

        p3_x = labels[:, 1] - labels[:, 4] * np.cos(np.radians(labels[:, 5])) / 2.0 - \
               labels[:, 3] * np.cos(np.radians(90 + labels[:, 5])) / 2.0
        p3_y = labels[:, 2] + labels[:, 4] * np.sin(np.radians(labels[:, 5])) / 2.0 + \
               labels[:, 3] * np.sin(np.radians(90 + labels[:, 5])) / 2.0

        p4_x = labels[:, 1] + labels[:, 4] * np.cos(np.radians(labels[:, 5])) / 2.0 - \
               labels[:, 3] * np.cos(np.radians(90 + labels[:, 5])) / 2.0
        p4_y = labels[:, 2] - labels[:, 4] * np.sin(np.radians(labels[:, 5])) / 2.0 + \
               labels[:, 3] * np.sin(np.radians(90 + labels[:, 5])) / 2.0

        patches = []
        for i in range(labels.shape[0]):
            if not np.any(labels[i]):  # objects in image finished before max_objects
                break
            verts = [(p1_x[i], p1_y[i]), (p2_x[i], p2_y[i]), (p3_x[i], p3_y[i]), (p4_x[i], p4_y[i]), (0., 0.), ]
            codes = [Path.MOVETO,        Path.LINETO,        Path.LINETO,        Path.LINETO,        Path.CLOSEPOLY, ]
            path = Path(verts, codes)
            patches.append(mpl.patches.PathPatch(path, linewidth=1, edgecolor='r', facecolor='none'))
            ax.text(verts[0][0], verts[0][1], classes[int(labels[i][0])], fontsize=6,
                    bbox=dict(edgecolor='none', facecolor='white', alpha=0.8, pad=0.))
        ax.add_collection(collections.PatchCollection(patches, match_original=True))
        # plt.show(block=False)
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig("output/%d_sin2&cos2.png", bbox_inches="tight", pad_inches=0.0)
        plt.show()

data_loder = DataLoader(data, batch_size=8,)
for _, img, labels in data_loder:
    '''
    # for one image
    img = img.squeeze()
    labels = labels.squeeze()
    img = np.transpose(img, (1, 2, 0))
    #plt.imshow(img)
    #plt.show()

    labels[:, 1] *= img.shape[0]
    labels[:, 2] *= img.shape[1]

    # denormalize w,l
    diagonal_length = np.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2)
    labels[:, 3] *= diagonal_length
    labels[:, 4] *= diagonal_length

    # denormalize theta
    labels[:, 5] *= 90.
    #visualize_data_image(img.numpy(), labels.numpy())



    #img_new = np.array(transforms.functional.rotate(Image.fromarray(np.uint8(img)), 15, expand=True))
    #t = HorizinatFlib(p=1)
    #img, labels =
    #t = Image_augmentation()
    #img = t(np.array(Image.open(img_file), dtype =np.float32) / 255)
    print(labels[:,3:-1])
    #print(labels)
    break
    '''
    visualize_data_batch(img, labels)

import numpy as np
from collections import defaultdict
import os
from utils.utils import load_classes
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

def parse_anno(annotation_path, file_name):
    """
    Read annotations from text file and convert it ti a numpy array
    :param annotation_path: Text
    :return: result: (N, 6)
    """
    result = np.array([[0,0,0,0,0,0]
                       ,[0,0,0,0,0,0]])
    with open(file_name, "r")as file:
        annotations_pathes = [annotation_path +'/'+ s.replace("\n", "") for s in file.readlines() ]

    for annotation in annotations_pathes:
        if annotation == annotation_path +'/'+ "classes.txt" : continue
        labels =  np.loadtxt(annotation, delimiter=' ', skiprows=1)
        if(len(labels.shape) == 1): labels = np.expand_dims(labels, axis = 0)
        # delete ADV, tuk tuk, tyckel
        labels = labels[labels[:, 0] != 18, :]
        labels = labels[labels[:, 0] != 16, :]
        labels = labels[labels[:, 0] != 14, :]

        # correct the order of the classes
        labels[labels[:, 0] == 15, 0] = 14
        labels[labels[:, 0] == 17, 0] = 15
        labels[labels[:, 0] == 19, 0] = 16
        
        result = np.concatenate([result, labels], axis = 0)

    return result[2:]


def draw_hist(x, value_name,  bins):
    plt.hist(x, bins = bins)
    plt.ylabel('No of times')
    plt.xlabel(value_name)
    plt.show()

if __name__ == '__main__':
    print("angel Statistics\n")
    labels_train = parse_anno('/media/user/76E6B044E6B00701/OmarElezaby/data', "val_new_final.txt")
    labels_data = parse_anno('/media/user/76E6B044E6B00701/OmarElezaby/data', "data.txt")
    #draw_hist(labels[:, 5], "theta", 18)
    theta_appers = defaultdict(int)
    for theta in labels_train[:,5]:
        theta_appers[int(theta)] += 1
    sort =  sorted(theta_appers.items(), key = lambda x:(x[1],x[0]), reverse= True)
    #for ang, val in sort[:]:
        #print(f"{ang} appeard {val}")

    print("\n\n classes Statistics\n")

    # for the classes
    classes = np.array(load_classes("/media/user/76E6B044E6B00701/OmarElezaby/data/classes.txt"))
    classes_appears_data = defaultdict(int)
    classes_appears_train = defaultdict(int)
    for cls in labels_train[:,0]:
        classes_appears_train[cls] += 1
    for cls in labels_data[:,0]:
        classes_appears_data[cls] += 1
    x =[]
    for xx in classes_appears_train.keys() : x.append(int(xx))
    #plt.bar(classes[x], classes_appears_train.values())
    #plt.show()
    sort = sorted(classes_appears_train.items(), key = lambda x:(x[0],x[1]), reverse=True)
    for cls, val in sort:
        print(f"{classes[int(cls)]} appears {classes_appears_data[cls]}")
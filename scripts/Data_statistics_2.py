import numpy as np
from collections import defaultdict
import os
from utils.utils import load_classes
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from PIL import Image

def parse_anno(annotation_path, end_with):
    """
    Read annotations from text file and convert it ti a numpy array
    :param annotation_path: Text
    :return: result: (N, 6)
    """

    annotations_files = [s for s in os.listdir(annotation_path) if s.endswith(end_with)]



    return annotations_files


def draw_hist(x, value_name,  bins):
    plt.hist(x, bins=bins)
    plt.ylabel('No of times')
    plt.xlabel(value_name)
    plt.show()





if __name__ == '__main__':

    #imgs_names_train = parse_anno("data/data")[:800]
    #labels_names_data = parse_anno("/media/user/76E6B044E6B00701/OmarElezaby/data", "txt")
    #imgs_names_data = parse_anno("/media/user/76E6B044E6B00701/OmarElezaby/data", "txt")
    #for img in imgs_names_data:
    #    if(img.replace(".jpg", ".txt") not in labels_names_data):print(img)


    #for img in  imgs_names_data:
    #   labels = np.loadtxt("data/data/" + img, delimiter=' ', skiprows=1)
    #   if(any(labels[:,0] == 12)): print(img)
    '''
    file = open("train_new_final.txt", "w")
    file2 = open("val_new_final.txt", "r")
    file3 = open("data.txt", "r")
    val_data = [s.replace("\n","") for s in file2.readlines()]
    data = set([s.replace("\n","") for s in file3.readlines()])
    for name in data:
        if name not in val_data: file.write(name +"\n")
    file.close()
    file2.close()
    file3.close()
    '''
    '''
    file = open("train_new_final.txt", "r")
    file2 = open("val_new_final.txt", "r")
    file3 = open("data.txt", "r")
    train_data = [s.replace("\n","") for s in file.readlines()]
    val_data = [s.replace("\n","") for s in file2.readlines()]
    print(len(set(train_data)))
    train_data.extend(val_data)
    #train_data = set()
    data = set([s.replace("\n","") for s in file3.readlines()])
    file.close()
    file2.close()
    file3.close()
    print(len(set(train_data)))
    print(len(set(val_data)))
    print(len(set(data)))

    '''
    '''
    file = open("train_new_final.txt", "w")
    file2 = open("val_new_final.txt", "r")
    file3 = open("data.txt", "r")
    val_data = [s.replace("\n", "") for s in file2.readlines()]
    data = set([s.replace("\n", "") for s in file3.readlines()])
    for name in data:
        if name not in val_data: file.write(name + "\n")
    #for name in imgs_names_data:
    #    file.write(name + "\n")

    #classes = load_classes("data/data/classes.txt")
'''


    #visualize_data("353292084896568_29982424_31335968_20170410153650_0_I_f07c7778-69b9-4b58-8f2b-72ef0da2fac8.jpg", "353292084896568_29982424_31335968_20170410153650_0_I_f07c7778-69b9-4b58-8f2b-72ef0da2fac8.txt")
    '''

    # split classes
    #val_names = [s for s in os.listdir("/media/user/76E6B044E6B00701/OmarElezaby/New") if s.endswith(".txt") and s != "classes.txt"]
    #data_names = [s for s in os.listdir("/media/user/76E6B044E6B00701/OmarElezaby/test") if s.endswith(".txt") and s != "classes.txt"]
    start = "867363024672796_30037813_31036673_20180823104807_0_M_e524c9ff-3c00-46ca-ac2b-5ad72040e1b7.txt"
    data_old = open("data_old.txt", "r")
    val_file = open("val_old_final.txt", "w")
    cnt = 0
    for name in data_old.readlines():
        if (cnt > 0 or name == start) and cnt <= 200:
            val_file.write(name.replace("\n","") + "\n")
            cnt += 1

    #training_file = open("data_new_final.txt", "w")
    #for name in data_names:
    #    training_file.write(name + "\n")
    '''
    '''
    #find best split
    data_old = open("data_new.txt","r")
    data = [s.replace("\n","") for s in data_old.readlines()]
    best_error = 1000
    best_start = ""
    new_appeard = [163, 1071, 335, 382, 754, 77, 79, 89, 416, 468, 282, 846, 22, 38, 193, 387, 665]

    data_all = open("data.txt","r")
    data_all_names = [s.replace("\n","") for s in data_all.readlines()]
    result = np.array([[0, 0, 0, 0, 0, 0]
                          , [0, 0, 0, 0, 0, 0]])
    for annotation in data_all_names:
        if annotation == "classes.txt": continue
        labels = np.loadtxt("/media/user/76E6B044E6B00701/OmarElezaby/data" + '/' + annotation, delimiter=' ',
                            skiprows=1)
        if (len(labels.shape) == 1): labels = np.expand_dims(labels, axis=0)
        # delete ADV, tuk tuk, tyckel
        labels = labels[labels[:, 0] != 18, :]
        labels = labels[labels[:, 0] != 16, :]
        labels = labels[labels[:, 0] != 14, :]

        # correct the order of the classes
        labels[labels[:, 0] == 15, 0] = 14
        labels[labels[:, 0] == 17, 0] = 15
        labels[labels[:, 0] == 19, 0] = 16

        result = np.concatenate([result, labels], axis=0)

    labels_data = result[2:].copy()


    for i, img in enumerate(data):
        result = np.array([[0, 0, 0, 0, 0, 0]
                              , [0, 0, 0, 0, 0, 0]])
        if(len(data) - i) < 200 : break
        data_chunck = data[i:i+200]
        for annotation in data_chunck:
            if annotation == "classes.txt": continue
            labels = np.loadtxt("/media/user/76E6B044E6B00701/OmarElezaby/data" + '/' + annotation, delimiter=' ', skiprows=1)
            if (len(labels.shape) == 1): labels = np.expand_dims(labels, axis=0)
            # delete ADV, tuk tuk, tyckel
            labels = labels[labels[:, 0] != 18, :]
            labels = labels[labels[:, 0] != 16, :]
            labels = labels[labels[:, 0] != 14, :]

            # correct the order of the classes
            labels[labels[:, 0] == 15, 0] = 14
            labels[labels[:, 0] == 17, 0] = 15
            labels[labels[:, 0] == 19, 0] = 16

            result = np.concatenate([result, labels], axis=0)
        labels_train = result[2:]


        # for the classes
        #classes = np.array(load_classes("data/data/classes.txt"))
        classes_appears_data = defaultdict(int)
        classes_appears_train = defaultdict(int)
        for cls in labels_train[:, 0]:
            classes_appears_train[cls] += 1
        for cls in labels_data[:, 0]:
            classes_appears_data[cls] += 1
        x = []
        for xx in classes_appears_train.keys(): x.append(int(xx))
        # plt.bar(classes[x], classes_appears_train.values())
        # plt.show()
        sort = sorted(classes_appears_train.items(), key=lambda x: (x[1], x[0]), reverse=True)
        error_count = 0
        for cls, val in sort:
            error_count += abs(.2 - (val + new_appeard[int(cls)]) / classes_appears_data[cls]) if (abs(.2 - (val + new_appeard[int(cls)]) / classes_appears_data[cls]) > .03) else 0
        if error_count < best_error:

            print(error_count)
            best_error = error_count
            best_start = img


    print(f"{best_start} have error {best_error}")
    '''

    from shutil import copyfile

    data_file = open("../train.txt", "r")
    data_imgs = [s.replace("\n","").replace(".txt",".jpg") for s in data_file.readlines()]
    data_labels = [s.replace(".jpg",".txt") for s in data_imgs]
    print(data_labels[0])
    for name in data_imgs:
        copyfile("../../data/"+name,"../../Train/"+name)

    for name in data_labels:
        copyfile("../../data/"+name,"../../Train/"+name)


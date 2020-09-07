import numpy as np

val_file_name = "data.txt"
detect_file_name = "detect_imgs.txt"
data_folder = "/media/user/76E6B044E6B00701/OmarElezaby/data"
class_number = 12
sampels_number = 50

with open(val_file_name,"r") as val_file:

    labels_files = [s.replace("\n","") for s in val_file.readlines()]

with open(detect_file_name,"w") as detect_file:
    cnt_sampels = 0
    for file in labels_files:

        labels = np.loadtxt(data_folder + "/" + file, delimiter=' ', skiprows=1)
        if (len(labels.shape) == 1): labels = np.expand_dims(labels, axis=0)
        if any(labels[:,0] == class_number):
            detect_file.write(file+"\n")
            cnt_sampels+=1

        if cnt_sampels >= sampels_number: break

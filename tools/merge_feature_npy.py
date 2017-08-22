import numpy as np
import os
import shutil

new_npy_dir = "up&down_jstl"


parts = ["up","down"]
datasets = ["prid","viper",'3dpes','ilids','cuhk01','cuhk03']
features = ['train_features.npy','val_features.npy','test_probe_features.npy','test_gallery_features.npy']
labels =['test_gallery_labels.npy','test_probe_labels.npy','train_labels.npy','val_labels.npy']
result_dir = "/home/nikoong/Algorithm_test/dgd_person_reid/external/exp/results/"


for dataset in datasets:
    for label in labels:
        up_label = result_dir +  "up_jstl/" + dataset +"_jstl_iter_55000_fc7_bn/"+label
        new_label =  result_dir + "up&down_jstl/" + dataset +"_jstl_iter_55000_fc7_bn/"+label
        shutil.copyfile(up_label,new_label)
    for feature in features:
        up_src_dir = result_dir +  "up_jstl/" + dataset +"_jstl_iter_55000_fc7_bn/"
        down_src_dir = result_dir +  "down_jstl/" + dataset +"_jstl_iter_55000_fc7_bn/"
        new_dir = result_dir + "up&down_jstl/" + dataset +"_jstl_iter_55000_fc7_bn/"
        up = np.load(os.path.join(up_src_dir, feature))
        down = np.load(os.path.join(down_src_dir, feature))
        up = list(up)
        down = list(down)
        new = []    
        if len(up) == len(down):    
            for i in range(len(up)):
                a = list(up[i])
                b = list(down[i])
                a.extend(b)
                new.append(np.array(a))
        new = np.array(new)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        np.save(os.path.join(new_dir, feature),new)


#print TX[1]   
#print type(TX[1])
 
        

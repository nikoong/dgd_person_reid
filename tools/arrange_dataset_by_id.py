import shutil 
import os

def Parsetxt(txt_path):
    file_list=[]
    label_list=[]
    with open(txt_path,'r') as f:
        for line in f:
            file_list.append(line.split(' ')[0])
            label_list.append(int(line.split(' ')[1]))
        return file_list, label_list 

datasets_path = '/home/nikoong/Algorithm_test/dgd_person_reid/external/exp/datasets/'
new_datasets_path = '/home/nikoong/dataset/re-id_new/'

datasets = ['prid','viper','3dpes','ilids','cuhk01','cuhk03']
subsets = ['train','val','test_gallery','test_probe']

for dataset in datasets:
    for subset in subsets:
        txt_path = datasets_path+dataset+'/'+'txt/'+subset+'.txt'
        files, label_list = Parsetxt(txt_path)
        for file in files:
            image = file.split('/')[-2]+'_'+file.split('/')[-1].split(' ')[0]            
            label = file.split('/')[-1].split('_')[0]
            label_dir = new_datasets_path+dataset+'/'+label
            if os.path.exists(label_dir):
                pass
            else:
                os.makedirs(label_dir)
            shutil.copy(file,  label_dir+'/'+image)
            
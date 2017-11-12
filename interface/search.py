import os
import lmdb
import numpy as np
from caffe.proto.caffe_pb2 import Datum
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
from scipy import misc

def Makefilelist(data_path):
    filelist=[]
    files= os.listdir(data_path)
    for filename in files:
        fullfilename = os.path.join(data_path,filename)
        filelist.append(fullfilename)
    return filelist

#list to txt
def list2txt(txt_list,txt_path):
    with open(txt_path,'w') as f :
        for line in txt_list:
            f.write(line+'\n')

#txt to list
def txt2list(txt_path):
    txt_list = []
    with open(txt_path) as f :
        for line in f:
            txt_list.append(line.split('\n')[0])
    return txt_list

def add_zero_label(datalist):
    newdatalist = []
    for line in datalist:
        newline = line + " 0"
        newdatalist.append(newline)
    return newdatalist 


def lmdb2npy(input_lmdb, output_npy, truncate=np.inf):
    datum = Datum()
    data = []
    env = lmdb.open(input_lmdb)
    with env.begin() as txn:
        cursor = txn.cursor()
        for i, (key, value) in enumerate(cursor):
            if i >= truncate: break
            datum.ParseFromString(value)
            data.append(datum.float_data)
    #data = np.squeeze(np.asarray(data))
    data = np.asarray(data)
    np.save(output_npy, data)








def extract_features(probelist, gallerylist, refrash_gallery, trained_model , blob):
    originpath = os.path.abspath('.')
    #prepare datalist    
    probelist_withlabel = add_zero_label(probelist)
    list2txt(probelist_withlabel, originpath + '/txt/probe.txt')
    gallerylist_withlabel = add_zero_label(gallerylist)    
    list2txt(gallerylist_withlabel, originpath + '/txt/gallery.txt')

    #prepare prototxt
    batch_size = 50
    if(refrash_gallery):
        subsets = ['probe', 'gallery']
    else:
        subsets = ['probe']
    for subset in subsets:
        data = txt2list(originpath + '/txt/' +subset +'.txt')
        num_image = len(data)
        num_batch = (num_image + batch_size - 1 )/batch_size 
        proto_temp = open(originpath + '/models/exfeat_template.prototxt')
        content = proto_temp.read()
        content = content.replace('${batch_size}', str(batch_size))
        content = content.replace('${source}', originpath + '/txt/'+subset+'.txt')
        proto_path = originpath + '/models/exfeat_template_'+subset+'.prototxt'
        with open(proto_path,'w') as f:
            f.write(content)
        lmdbpath = originpath +'/feature/'+subset+'_features_lmdb'
        if os.path.isdir(lmdbpath): 
            os.system('rm -rf '+lmdbpath)
        #extract features
        command = originpath + '/../external/caffe/build/tools/extract_features ' \
                + trained_model + ' ' \
                + proto_path +' ' \
                + blob +' ' \
                + lmdbpath +' ' \
                + str(num_batch) +' '\
                + 'lmdb GPU 0'
        print 'extract '+ subset +' features'+'\n'
        os.system(command)
        #convert lmdb to npy
        npypath = originpath +'/feature/'+subset+'_features.npy'
        lmdb2npy(lmdbpath, npypath, num_image)

def compute_distance_matirx():
    originpath = os.path.abspath('.')
    PX = np.load(originpath + '/feature/probe_features.npy')
    GX = np.load(originpath + '/feature/gallery_features.npy')
    print PX.shape
    print GX.shape
    print "probe number: ",len(PX)
    print "gallery number: ",len(GX)
    D = pairwise_distances(GX, PX, metric='mahalanobis',  n_jobs=-2)
    return D

def compute_distance():
    originpath = os.path.abspath('.')
    PX = np.load(originpath + '/feature/probe_features.npy')
    GX = np.load(originpath + '/feature/gallery_features.npy')
    D = euclidean_distances(GX, PX)
    return D
    
    
def show_result(order,probelist,gallerylist):
    #show probe image
    for j in range(len(probelist)):
        print '\n'
        print probelist[j],'\n'
        plt.figure(num='result',figsize=(8,4)) 
        img = mpimg.imread(probelist[j])
        img = misc.imresize(img,(200,65))
        plt.subplot(2,10,1)
        plt.imshow(img)
        plt.axis('off')
        #show top 10 gallery image
        for i in range(10):
            img = mpimg.imread(gallerylist[order[i][j]])
            img = misc.imresize(img,(200,65))
            plt.subplot(2,10,i+11)
            plt.imshow(img)
            plt.axis('off')
            print  gallerylist[order[i][j]] ,'\n'    
        plt.show()

        
def Parsetxt(txt_path):
    file_list=[]
    label_list=[]
    with open(txt_path,'r') as f:
        for line in f:
            file_list.append(line.split(' ')[0])
            label_list.append(int(line.split(' ')[1]))
        return file_list, label_list   

def compute_threshold():
    probe_txt = '/home/nikoong/Algorithm_test/dgd_person_reid/external/exp/datasets/cuhk03/txt/test_probe.txt'
    gallery_txt = '/home/nikoong/Algorithm_test/dgd_person_reid/external/exp/datasets/cuhk03/txt/test_gallery.txt'
    probelist,PY = Parsetxt(probe_txt)
    gallerylist,GY = Parsetxt(gallery_txt)
    extract_features(probelist, gallerylist,  True, os.path.abspath('.')+"/trained/jstl_iter_20000.caffemodel", 'fc7_bn')
    D = compute_distance_matirx()
    
    distance = 0
    count = 0
    for i in range(len(probelist)):
        if i%100 ==0: print "i=",i
        for j in range(len(gallerylist)):
            if PY[i] == GY[j]:
                count = count+1
                distance = distance + D[j][i]
    print count,'\n'
    print distance/count,'\n'
    return distance/count






#######################################  interface  ##################################################  

def is_same(a, b ,threshold,trained_model = os.path.abspath('.')+"/trained/jstl_iter_20000.caffemodel", blob='fc7_bn'):
    lista = [a]
    listb = [b]
    extract_features(lista, listb, True, trained_model, blob)
    D = compute_distance()
    if D[0][0] <= threshold:
        print "Same one"
        return True
    else:
        print D[0][0]
        return False

def search_single(probe, gallery_txt, refrash_gallery=True, trained_model = os.path.abspath('.')+"/trained/jstl_iter_20000.caffemodel", blob='fc7_bn'):
    probelist = [probe]
    gallerylist = txt2list(gallery_txt)
    extract_features(probelist, gallerylist, refrash_gallery, trained_model, blob)
    D = compute_distance_matirx()
    order = np.argsort(D, axis=0)
    show_result(order,probelist,gallerylist)
    

def search_multiprobe(probe_txt, gallery_txt, refrash_gallery=True, trained_model = os.path.abspath('.')+"/trained/jstl_iter_20000.caffemodel", blob='fc7_bn'):
    probelist = txt2list(probe_txt)
    gallerylist = txt2list(gallery_txt)
    extract_features(probelist, gallerylist, refrash_gallery, trained_model, blob)
    D = compute_distance_matirx()
    order = np.argsort(D, axis=0)
    show_result(order,probelist,gallerylist)





######################################################################################### 
'''
search_single(
    '/home/nikoong/Algorithm_test/dgd_person_reid/external/exp/datasets/cuhk03/cam_0/00636_00007.jpg',
    '/home/nikoong/Algorithm_test/dgd_person_reid/interface/txt/gallery(origin).txt',
    True
    )
'''


'''
search_multiprobe(
    '/home/nikoong/Algorithm_test/dgd_person_reid/interface/txt/probe(origin).txt',
    '/home/nikoong/Algorithm_test/dgd_person_reid/interface/txt/gallery(origin).txt',
    True
    )
'''

'''
print is_same(
    '/home/nikoong/Algorithm_test/dgd_person_reid/external/exp/datasets/cuhk03/cam_0/00636_00007.jpg',
    '/home/nikoong/Algorithm_test/dgd_person_reid/external/exp/datasets/cuhk03/cam_0/00636_00006.jpg',
    22.16
    )
'''




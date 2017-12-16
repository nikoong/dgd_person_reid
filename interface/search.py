# -*- coding: utf-8 -*-

import os
import lmdb
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
from scipy import misc

import sys
caffe_root = '/home/nikoong/Algorithm_test/attribute/external/caffe'
sys.path.append(caffe_root + 'python')
import caffe
from caffe.proto.caffe_pb2 import Datum



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
        #remove old features files and extract new features files
        lmdbpath = originpath +'/feature/'+subset+'_features_lmdb'
        if os.path.isdir(lmdbpath): 
            os.system('rm -rf '+lmdbpath) 
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
    D = pairwise_distances(GX, PX, metric='mahalanobis', VI = np.eye(PX.shape[1]), n_jobs=-2)
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


def save_result(order,probelist,gallerylist,txt_path):
    resultlist=[]
    for j in range(len(probelist)):
        for i in range(len(order)):
            resultlist.append(gallerylist[order[i][j]])
    list2txt(resultlist,txt_path)
  


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


def gen_attribute_dic(imagelist):
    #功能:打印图片中行人属性。
    #imagelist：list，输入图片列表
    #输出list，list里每一项对应这输入图片各个属性的一个字典

    originpath = os.path.abspath('.')
    #prepare datalist
    num_image = len(imagelist)
    imagelist_withlabel = add_zero_label(imagelist)
    list2txt(imagelist_withlabel, originpath + '/txt/attri.txt')
    #defing net
    model_def = originpath + '/models/exfeat_attribute.prototxt'
    model_weights = originpath + '/trained/DUCK_10_batch30_3_lr01_loss08_001_iter_30000.caffemodel'
    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)
    caffe.set_device(0)  # if we have multiple GPUs, pick the first one
    caffe.set_mode_gpu()
    batch_size = 50
    attributes_map = {
                'downcolor_prob':['downblack', 'downblue', 'downbrown', 'downgray', 'downgreen', 'downred', 'downwhite'],
                'gender_prob':['male','female'],
                'bag_prob':['no','yes'],
                'boots_prob':['no','yes'],
                'backpack_prob':['no','yes'],
                'handbag_prob':['no','yes'],
                'hat_prob':['no','yes'],
                'shoes_prob':['dark', 'light'],
                'top_prob':['short','long'],
                'upcolor_prob':['upblack', 'upblue', 'upbrown', 'upgray', 'upgreen', 'uppurple', 'upred', 'upwhite']}
    attributes_list=[] 
    num_batch = (num_image + batch_size - 1 )/batch_size
    for i in range(num_batch):
        output = net.forward()  # run once before timing to set up memory
        for j in range(batch_size):
            if i*batch_size + j >= num_image: break
            new_attributes = {}
            for attribute in attributes_map.keys():
                probe = output[attribute][j].argmax()
                new_attributes[attribute] = attributes_map[attribute][probe]
            attributes_list.append(new_attributes)
    return attributes_list

#######################################  interface  ################################################## 


def gen_attribute(imagelist):
    #功能:打印图片中行人属性。
    originpath = os.path.abspath('.')
    #prepare datalist
    num_image = len(imagelist)
    imagelist_withlabel = add_zero_label(imagelist)
    list2txt(imagelist_withlabel, originpath + '/txt/attri.txt')
    #defing net
    model_def = originpath + '/models/exfeat_attribute.prototxt'
    model_weights = originpath + '/trained/DUCK_10_batch30_3_lr01_loss08_001_iter_30000.caffemodel'
    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)
    caffe.set_device(0)  # if we have multiple GPUs, pick the first one
    caffe.set_mode_gpu()
    batch_size = 50
    attributes_map = {
                'downcolor_prob':['downblack', 'downblue', 'downbrown', 'downgray', 'downgreen', 'downred', 'downwhite'],
                'gender_prob':['male','female'],
                'bag_prob':['no','yes'],
                'boots_prob':['no','yes'],
                'backpack_prob':['no','yes'],
                'handbag_prob':['no','yes'],
                'hat_prob':['no','yes'],
                'shoes_prob':['dark', 'light'],
                'top_prob':['short','long'],
                'upcolor_prob':['upblack', 'upblue', 'upbrown', 'upgray', 'upgreen', 'uppurple', 'upred', 'upwhite']}
    num_batch = (num_image + batch_size - 1 )/batch_size
    for i in range(num_batch):
        output = net.forward()  # run once before timing to set up memory
        for j in range(batch_size):
            if i*batch_size + j >= num_image: break
            print imagelist[i*batch_size + j]
            for attribute in attributes_map.keys():
                probe = output[attribute][j].argmax()
                print attribute.ljust(14)+":" , attributes_map[attribute][probe]
                


def is_same(a, b ,threshold = 22.16,trained_model = os.path.abspath('.')+"/trained/jstl_iter_20000.caffemodel", blob='fc7_bn'):
    #功能：判断a，b两幅图片是否是同一个人
    #参数：
    #a，b是图片路径
    #threshold是阈值，默认22.6；阈值越大，判断越严格
    #trianed_model,是训练好的caffemodel文件
    #blob，是特征提取的层的名字
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
    #功能：提供一张图片probe，在图片库gallery中搜索同一个人
    #参数：
    #probe是单张目标人物图片路径, gallery_txt是图片库组成的txt路径,
    #refrash_gallery，bool类型，是否改变了图片库；如果图片库没有改变，就设为False，函数会自动使用上次图片库中图像特征，避免重复计算
    #threshold是阈值，默认22.6；阈值越大，判断越严格
    #trianed_model,是训练好的caffemodel文件
    #blob，是特征提取的层的名字
    probelist = [probe]
    gallerylist = txt2list(gallery_txt)
    extract_features(probelist, gallerylist, refrash_gallery, trained_model, blob)
    D = compute_distance_matirx()
    order = np.argsort(D, axis=0)
    show_result(order,probelist,gallerylist)
    

def search_multiprobe(probe_txt, gallery_txt, refrash_gallery=True, trained_model = os.path.abspath('.')+"/trained/jstl_iter_20000.caffemodel", blob='fc7_bn'):
    #功能：对probe_txt中每张图片进行搜索，因为采取了多线程，所以比循环调用search_single函数快。
    #参数：
    #probe是多张目标人物图片路径, gallery_txt是图片库图片组成的txt路径,
    #refrash_gallery，bool类型，是否改变了图片库；如果图片库没有改变，就设为False，函数会自动使用上次图片库中图像特征，避免重复计算
    #threshold是阈值，默认22.6；阈值越大，判断越严格
    #trianed_model,是训练好的caffemodel文件
    #blob，是特征提取的层的名字
    probelist = txt2list(probe_txt)
    gallerylist = txt2list(gallery_txt)
    extract_features(probelist, gallerylist, refrash_gallery, trained_model, blob)
    D = compute_distance_matirx()
    order = np.argsort(D, axis=0)
    show_result(order,probelist,gallerylist)


def search_single_with_attibutesdic(probe, gallery_txt, attributes_check={}, refrash_gallery=True, trained_reid_model = os.path.abspath('.')+"/trained/jstl_iter_20000.caffemodel", blob='fc7_bn'):
    #功能：提供一张图片probe，在图片库gallery中搜索同一个人，根据提供的属性进一步筛选
    #参数：
    #probe是单张目标人物图片路径, gallery_txt是图片库组成的txt路径,
    #attributes_check,dic类型，是属性字典{'gender_prob':'male','downcolor_prob':'downblack'}
    #refrash_gallery，bool类型，是否改变了图片库；如果图片库没有改变，就设为False，函数会自动使用上次图片库中图像特征，避免重复计算
    #trianed_model,是训练好的caffemodel文件

    probelist = [probe]
    gallerylist = txt2list(gallery_txt)
    extract_features(probelist, gallerylist, refrash_gallery, trained_reid_model, blob)
    D = compute_distance_matirx()
    order = np.argsort(D, axis=0)
    print len(order)
    order = list(order)
    if attributes_check != {}:
        gallery_attributes = gen_attribute_dic(gallerylist)
        for attribute in attributes_check.keys():
            for i in range(len(gallerylist)):
                if (gallery_attributes[i][attribute]!=attributes_check[attribute])&([i] in order):
                    order.remove([i])
        order = np.asarray(order)
        print len(order)      
        #show_result(order,probelist,gallerylist)
        save_result(order,probelist,gallerylist,os.path.abspath('.')+"/result/result.txt")


def search_single_with_attibuteslist(probe, gallery_txt, attributeslist=[], refrash_gallery=True, trained_reid_model = os.path.abspath('.')+"/trained/jstl_iter_20000.caffemodel", blob='fc7_bn'):
    #功能：提供一张图片probe，在图片库gallery中搜索同一个人，根据提供的属性进一步筛选
    #参数：
    #probe是单张目标人物图片路径, gallery_txt是图片库组成的txt路径,
    ##attributes_check,list类型，是属性列表['gender_prob','downcolor_prob']
    #refrash_gallery，bool类型，是否改变了图片库；如果图片库没有改变，就设为False，函数会自动使用上次图片库中图像特征，避免重复计算
    #trianed_model,是训练好的caffemodel文件
    #blob，是特征提取的层的名字
    probelist = [probe]
    gallerylist = txt2list(gallery_txt)
    extract_features(probelist, gallerylist, refrash_gallery, trained_reid_model, blob)
    D = compute_distance_matirx()
    order = np.argsort(D, axis=0)
    print len(order)
    order = list(order)
    if attributeslist != []:
        probe_attributes = gen_attribute_dic(probelist)[0]
        gallery_attributes = gen_attribute_dic(gallerylist)
        for attribute in attributeslist:
            for i in range(len(gallerylist)):
                if (gallery_attributes[i][attribute]!=probe_attributes[attribute]and [i] in order ):
                    order.remove([i])
        order = np.asarray(order)
        print  len(order)      
        #show_result(order,probelist,gallerylist)
        save_result(order,probelist,gallerylist,os.path.abspath('.')+"/result/result.txt")

    




########################################Example################################################# 
'''
search_single(
    '/home/nikoong/Algorithm_test/dgd_person_reid/external/exp/datasets/cuhk03/cam_0/00636_00007.jpg',
    '/home/nikoong/Algorithm_test/dgd_person_reid/interface/txt/gallery(origin).txt',
    True
    )
'''


search_single_with_attibuteslist(
    '/home/nikoong/Algorithm_test/dgd_person_reid/external/exp/datasets/cuhk03/cam_0/00636_00007.jpg',
    '/home/nikoong/Algorithm_test/dgd_person_reid/interface/txt/gallery(origin).txt',
    ['gender_prob'],
    True
    )

'''

search_single_with_attibutesdic(
    '/home/nikoong/Algorithm_test/dgd_person_reid/external/exp/datasets/cuhk03/cam_0/00636_00007.jpg',
    '/home/nikoong/Algorithm_test/dgd_person_reid/interface/txt/gallery(origin).txt',
    {'gender_prob':'male','downcolor_prob':'downblack'},
    False
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


#gen_attribute(txt2list('/home/nikoong/Algorithm_test/dgd_person_reid/interface/txt/probe(origin).txt'))

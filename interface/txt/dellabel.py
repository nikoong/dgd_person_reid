def Parsetxt(txt_path):
    file_list=[]
    label_list=[]
    with open(txt_path,'r') as f:
        for line in f:
            file_list.append(line.split(' ')[0])
            label_list.append(int(line.split(' ')[1]))
        return file_list, label_list 

#list to txt
def list2txt(txt_list,txt_path):
    with open(txt_path,'w') as f :
        for line in txt_list:
            f.write(line+'\n')

file,label = Parsetxt('test_probe.txt')
list2txt(file,'/home/nikoong/Algorithm_test/dgd_person_reid/interface/txt/probe(origin).txt')

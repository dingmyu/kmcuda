#python move.py 2 5 ~/feature_path.txt ~/t.txt
import os
import sys
classes = int(sys.argv[1])
feature_num = int(sys.argv[2])
feature_file = sys.argv[3]
name_file = sys.argv[4]
os.system("rm -rf /home/face/result")
os.mkdir("/home/face/result")
os.system("/home/face/index/workspace/example %s %d %d" % (feature_file, classes, feature_num))
work_dir ="/home/face/result/"
for i in range(classes):
    if not os.path.exists(work_dir + str(i)):
        os.mkdir(work_dir + str(i))
f_label = open("label.txt")
f = f_label.readlines()
f_name = open(name_file)
f1 = f_name.readlines()
for i in range(feature_num):
    label = f[i].strip().split(' ')[1]
    os.system("cp %s %s" % (f1[i].strip(), work_dir + label))

f_label.close()
f_name.close()


from numpy import *
f_label = open("label.txt")
f = f_label.readlines()
f_feature = open(feature_file)
fff = f_feature.readlines()
label_dict = {}
for line in f:
    num, label = line.strip().split()
    num = int(num)
    label = int(label)
    label_dict[label] = label_dict.get(label, [])
    label_dict[label].append(num)
f_name = open(name_file)
all_name = f_name.readlines()
f_cen = open('centroids.txt')
all_cen = f_cen.readlines()
use_name = ''
f_cen_name = open('cen_name.txt','w')
for index,line in enumerate(all_cen):
    min_distance = 10000.0
    if index in label_dict.keys():
        print >> f_cen_name, index, 
        for item in label_dict[index]:
            #if line.strip().split()[0] != 'nan':
            point = mat([float(number) for number in fff[item].strip().split()])
            center = mat([float(number) for number in line.strip().split()])
            new_distance = float(sqrt((point-center)*((point-center).T)))
            if new_distance < min_distance:
                min_distance = new_distance
                use_name = all_name[item].strip()
                #print item, new_distance
        print >> f_cen_name, use_name 
f_label.close()
f_feature.close()
f_name.close()
f_cen.close()  
f_cen_name.close()

f_cen = open('cen_name.txt')
f = f_cen.readlines()
for line in f:
#    print line.strip().split()
    num, name = line.strip().split()
    os.system('cp %s %s%d/center.png' % (name, work_dir, int(num)))
f_cen.close()


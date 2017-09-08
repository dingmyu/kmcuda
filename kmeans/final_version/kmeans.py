#python kmeans.py 1000 16781 /home/face/Downloads/test/video/v_feature.txt /home/face/Downloads/test/video/video.txt
import os
import sys
classes = int(sys.argv[1])
feature_num = int(sys.argv[2])
feature_file = sys.argv[3]
name_file = sys.argv[4]

def compute_distance(point, center):
	return float(sqrt((point-center)*((point-center).T)))/float(sqrt(center*center.T))

def get_center(all_list):
	#length = fff[all_list[0]].shape[1]
	#data = mat(zeros((1,length)))
	#for item in all_list:
	#	data = vstack((data, fff[item]))
	data = vstack([fff[item] for item in all_list])
	return data.sum(axis=0)/len(all_list)


from numpy import *
import copy
import json

os.system("/home/face/index/workspace/example %s %d %d" % (feature_file, classes, feature_num))
f_label = open("label.txt")
f = f_label.readlines()
f_feature = open(feature_file)
fff = [mat([float(number) for number in line.strip().split()]) for line in f_feature.readlines()]
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
label_dict['others'] = []
tmp_dict = copy.deepcopy(label_dict)
for index,line in enumerate(all_cen):
    if index in tmp_dict.keys():
        for item in tmp_dict[index]:
            #if line.strip().split()[0] != 'nan':
            point = fff[item]
            #point = mat([float(number) for number in fff[item].strip().split()])
            center = mat([float(number) for number in line.strip().split()])
            new_distance = compute_distance(point, center)
            if new_distance > 0.6: #cong yiyou lei shanchu
                label_dict[index].remove(item)
                label_dict['others'].append(item)

center_dict = {}
tmp_dict = copy.deepcopy(label_dict)
del tmp_dict['others']
for index in tmp_dict.keys():
	if len(tmp_dict[index]) < 3:
		for item in tmp_dict[index]:
			label_dict[index].remove(item)
			label_dict['others'].append(item)
	else:
		center_dict[index] = mat([float(number) for number in all_cen[index].strip().split()])

label_sum = classes
tmp_dict = copy.deepcopy(label_dict)
others_num = 0
for item in tmp_dict['others']:
	others_num += 1
	if others_num % 1000 == 0:
		print 'pic', others_num
	point = fff[item]
	flag = 0
	#point = mat([float(number) for number in fff[item].strip().split()])
	for center in center_dict:
		cen_point = center_dict[center]
		new_distance = compute_distance(point, cen_point)
		if new_distance < 0.6: # hebing dao yiyou lei zhong
			label_dict[center].append(item)
			center_dict[center] = get_center(label_dict[center])
			label_dict['others'].remove(item)
			flag = 1
			#print new_distance
			break
	if flag == 0:
		label_dict[label_sum] = []
		label_dict[label_sum].append(item)
		center_dict[label_sum] = fff[item]
		label_sum += 1
		label_dict['others'].remove(item)

			
#print label_sum
#print len(label_dict['others'])





os.system("rm -rf /home/face/result")
os.mkdir("/home/face/result")
#os.system("/home/face/index/workspace/example %s %d %d" % (feature_file, classes, feature_num))
work_dir ="/home/face/result/"


label_num = 0
f_cenfea = open('cen_feature.txt', 'w')
f_cenname = open('cen_name.txt', 'w')
for label in label_dict:
	label_num += 1
	#print >> f_cenname, label,
	if label_num % 100 == 0:
		print 'label', label_num
	if len(label_dict[label]) > 2:
		if not os.path.exists(work_dir + str(label)):
			os.mkdir(work_dir + str(label))
		center = center_dict[label]
		min_name = ''
		min_distance = 1.0
		for item in label_dict[label]:
			os.system("cp %s %s" % (all_name[item].strip(), work_dir + str(label)))
			point = fff[item]
			new_distance = compute_distance(point, center)
			if new_distance < min_distance:
				min_distance = new_distance
				min_name = all_name[item].strip()
		os.system("cp %s %s/center.png" % (min_name, work_dir + str(label)))
                print  >> f_cenname, min_name
		print  >> f_cenfea, json.dumps(center.tolist()[0])
	#else:
        #    print  >> f_cenname, ''
f_cenfea.close()
f_label.close()
f_feature.close()
f_name.close()
f_cen.close()
f_cenname.close()

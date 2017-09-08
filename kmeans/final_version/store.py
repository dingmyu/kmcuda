from numpy import *
import pymongo
import time
import multiprocessing
import bson.binary  
from cStringIO import StringIO
import json
 
client = pymongo.MongoClient(host="localhost", port=27017)
db = client['test']
coll = db['face']


def compute_distance(point, center):
	return float(sqrt((point-center)*((point-center).T)))/float(sqrt(center*center.T))

#{'feature': [1,2,3,3,4,2], 'pic': '' ...}
def insert_one_doc(information):
    flag = 1
    for item in coll.find():
        if compute_distance(mat(item['feature']), mat(information['feature'])) < 0.6:
            flag = 0
            print "already have this person, the id is %d, the similarity is" % item['id'], compute_distance(mat(item['feature']), mat(information['feature']))
            #print item
            break
    if flag == 1:
        id_ = coll.count()
        information['id'] = id_ + 1
        coll.insert(information)
        print 'store this person'
        #print information
        

f = open('cen_feature.txt')
f1 = open('cen_name.txt')
f_fea = f.readlines()
f_name = f1.readlines()
line_num = len(f_fea)
num = 0
pool = multiprocessing.Pool(processes = 10)
for i in range(line_num):
    num += 1
    if num % 50 == 0:
        print num
    line = json.loads(f_fea[i].strip())
   # line = [float(item) for item in line.strip().split()]
    if line[0] == line[0]:
        with open (f_name[i].strip(),'rb') as myimage:
            content = StringIO(myimage.read())
            information = {'id': '', 'feature': line, 'pic': bson.binary.Binary(content.getvalue()), 'store_time': time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}
            pool.apply_async(insert_one_doc, (information, ))
pool.close()
pool.join()
print "Sub-process(es) done."
        #insert_one_doc(information)
f.close()
f1.close()


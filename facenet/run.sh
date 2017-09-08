# /usr/bin/env sh

python /home/face/Downloads/test/extract_frame.py

#cd /home/face/tensorflow

#source ~/tensorflow/bin/activate

python /home/face/facenet-master/src/align/align_dataset_mtcnn.py \
/home/face/Downloads/test/Input \
/home/face/Downloads/test/output \
--image_size 160 \
--gpu_memory_fraction=0.9

sh /home/face/Downloads/test/create_filelist.sh

python /home/face/facenet-master/convert_to_tfrecord.py \
--image_size=160 --save_name='save_name.tfrecord' \
--filename='/home/face/Downloads/test/test.txt' --image_dir=''
line=`sed -n '$=' /home/face/Downloads/test/test.txt`
python /home/face/facenet-master/jd/extract_feature_tf.py \
--image_size=160 --batch_size=64 --image_num=$line \
--filename='save_name.tfrecord' \
--model='/home/face/facenet-master/data/20170512-110547' \
--feature_path='/home/face/Downloads/test/feature_path.txt'

python /home/face/move.py 10 $line \
/home/face/Downloads/test/feature_path.txt \
/home/face/Downloads/test/test.txt
# /usr/bin/env sh
#gcc -std=c99 -O2 kmeans.c -I/home/face/index/kmcuda/src -L/home/face/index/kmcuda/src -l KMCUDA -Wl,-rpath=/home/face/index/kmcuda/src -o example
python kmeans.py 1000 16781 /home/face/Downloads/test/video/v_feature.txt /home/face/Downloads/test/video/video.txt
python store.py
#rm *.txt

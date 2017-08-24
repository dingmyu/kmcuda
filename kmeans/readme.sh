gcc -std=c99 -O2 kmeans.c -I/home/face/index/kmcuda/src -L/home/face/index/kmcuda/src -l KMCUDA -Wl,-rpath=/home/face/index/kmcuda/src -o example
./example data.txt 400

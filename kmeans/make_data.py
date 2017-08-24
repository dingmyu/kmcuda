import random
f = open('data.txt','w')
for i in range(400000):
    for j in range(2048):
        f.write(str(random.uniform(-1, 1)) + ' ')
    f.write('\n')
    if i % 1000 == 0:
        print i

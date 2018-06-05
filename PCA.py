import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
import os

#task 01
path = '.' + os.sep + 'optdigits-orig.tra' + os.sep + 'optdigits-orig.tra'
input_file = open(path)
sample = []
flag = 1
for line in islice(input_file,21,None,1):
    line = line.strip('\n')
    line = line.strip(' ')
    if line == '3':
        data_list = np.array(list(map(int, sample)))
        data_list = data_list[ :, np.newaxis]
        sample = []
        if flag:
            data = data_list;
            flag = 0
        else:
            data = np.hstack((data, data_list))
    elif len(line) == 1:
        sample = []
    else:
        sample = sample + list(line)

#task 02
meandata = np.mean(data, axis =1)
meandata = meandata[ :, np.newaxis]
meanpic = meandata.reshape(32,32)
data_n = (data - meandata).transpose();

u, s, vh = np.linalg.svd(data_n, full_matrices=False)
w = np.matmul(u[:,0:2], np.diag(s[0:2]))
x =  vh[0:2,:]

plt.figure(1)
coordinate1 =  np.tile([-5, -2.5, 0.0, 2.5, 5], 5)
temp = np.arange(5, -6, -2.5)
temp = temp[ :, np.newaxis]
coordinate2 =  np.matmul(temp,np.ones((1,5)))
coordinate2 = coordinate2.reshape(25,)
plt.scatter(coordinate1, coordinate2, color='', marker='o', edgecolors='r')

plt.plot(w[:,0], w[:,1], 'yo',markersize=2) 
plt.ylabel('Second Principal Component')
plt.xlabel('First Principal Component')
plt.grid(color='k', linestyle='--', linewidth=1)
#plt.savefig('1.png', dpi=300)

plt.figure(2)
for index in range(1,26):
    plt.plot([1,2,3])
    components = np.matmul(np.array([coordinate1[index-1], coordinate2[index-1]]), x)
    components = components[ :, np.newaxis]
    pic_pca = np.add(meandata, components)
    #pic_pca = meandata
    pic_pca = 1 - pic_pca
    pic_pca =  (pic_pca >= 0.5) * 1 ## binary image
    pic_pca = pic_pca.reshape(32,32)
    plt.subplot(5,5,index)
    plt.axis('off') 
    plt.imshow(pic_pca, cmap="gray")
#plt.savefig('2.png', dpi=300)
plt.show()


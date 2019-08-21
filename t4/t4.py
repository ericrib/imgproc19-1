'''
Assignment 4: Colour image processing and segmentation
'''

import numpy as np
import imageio
import random
# import matplotlib.pyplot as plt


def opt_transform(img,opt):

    # 1 RGB
    if (opt==1):
        return img

    # 2 RGB,x,y
    elif (opt==2):
        new = np.empty((img.shape[0],img.shape[1],5))
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                new[i,j,0]=img[i,j,0]
                new[i,j,1]=img[i,j,1]
                new[i,j,2]=img[i,j,2]
                new[i,j,3]= float(j)
                new[i,j,4]= float(i)

        return new

    # 3 Luminance
    elif (opt==3):
        new = np.empty(img.shape)
        new = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]

        return new

    # 4 Luminance,x,y
    elif (opt==4):
        new = np.empty((img.shape[0],img.shape[1],3),dtype=float)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                new[i,j,0] = 0.299 * img[i,j,0] + 0.587 * img[i,j,1] + 0.114 * img[i,j,2]
                new[i,j,1]= float(j)
                new[i,j,2]= float(i)

        return new

    else:
        return img


def dist_eucld(a,b):
    x = a.astype(float) - b.astype(float)
    square = np.square(x)
    return np.sqrt(np.sum(square))


def cluster(img, dict, k , seg):


    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            comp = np.inf
            for q in range(k):
                di,dj = dict[q]
                dist = dist_eucld(img[i,j],img[di,dj])
                if(dist < comp):
                    seg[i,j] = np.uint8(q)
                    comp = dist

    return seg

def update_dict(dict,k,seg):

    sums = np.zeros((k,3))
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            sums[seg[i,j],0] += 1
            sums[seg[i,j],1] += i
            sums[seg[i,j],2] += j

    for q in range(k):
        if(sums[q,0]>0):
            mean_i = int(sums[q,1]/sums[q,0])
            mean_j = int(sums[q,2]/sums[q,0])
            dict[q] = (mean_i,mean_j)
        else:
            tup = (random.randint(0,seg.shape[0]-1),
                    random.randint(0,seg.shape[1]-1))
            dict[q] = tup


def kmeans(k,img,n,opt,S):

    random.seed(S)
    img = opt_transform(img,opt)

    centroid_dict = {}
    ids = np.sort(random.sample(range(0,img.shape[0]*img.shape[1]),k))
    for i in range(k):
        tup = (ids[i]//img.shape[0], ids[i]%img.shape[0])
        centroid_dict[i] = tup

    seg = np.empty((img.shape[0],img.shape[1]),dtype=np.uint8)

    for i in range(n):
        # assign each pixel to cluster using distance
        seg = cluster(img,centroid_dict,k, seg)

        # update cluster by average of members
        update_dict(centroid_dict, k, seg)

    return seg


# normalize numpy ndarray to [0,2^8-1]
def normalize(matrix):
    max = np.amax(matrix)
    min = np.amin(matrix)
    matrix = (matrix - min) / (max - min)
    matrix *= ((2**8)-1)

    return matrix

# calculate error
def RMSE(ref,new):
    ref = ref.astype(float)
    new = new.astype(float)
    error = np.sqrt(np.sum((ref - new)**2)) / ref.size
    return error

def main():

    # Parameter input
    img_name = str(input()).rstrip()
    ref_name = str(input()).rstrip()
    # Reading images
    img = imageio.imread(img_name)
    ref = np.load(ref_name)

    # plt.imshow(img, cmap='gray')
    # plt.show()

    # Option for pixel attributes
    opt = int(input())
    # Number of clusters k
    k = int(input())
    # Number of iterations
    n = int(input())
    # Seed S for random centroid choice
    S = int(input())

    new_img = kmeans(k, img, n, opt, S)
    normalize(new_img)
    # print error
    print("%.4f" % RMSE(ref, new_img))
    # plt.imshow(new_img, cmap='gray')
    # plt.show()

if __name__ == '__main__':
    main()
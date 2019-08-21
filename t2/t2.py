'''
Image Processing
Assignment 2
'''

import numpy as np
import imageio
import matplotlib.pyplot as plt

# convert image to b&w based on estimated threshold
def limiarization(img, thresh_ini):

    thresh = thresh_ini
    dif = 1
    #until dif < 0.5
    while dif >= 0.5:
        g1 = img[img>thresh]
        g2 = img[img<=thresh]
        thresh_new = (np.mean(g1) + np.mean(g2)) / 2
        dif = abs(thresh_new - thresh)
        thresh = thresh_new

    return np.where(img < thresh, 0, 1)

# edit flattened image with 1D filter
def filtr1D(img, filter_siz, weights):

    eps = filter_siz // 2
    pad_img = np.pad(img.flatten(), eps, 'wrap')
    new_img = np.empty(img.flatten().shape)

    for i in range(len(new_img)):
        begi = i
        endi = i + 2 * eps + 1
        new_img[i] = np.sum(pad_img[begi:endi]*weights)

    new_img = new_img.reshape(img.shape)
    return new_img

# edit image with 2D filter
def filtr2DLimiar(img, filter_siz, thresh_ini, weights):

    eps = filter_siz // 2
    pad_img = np.pad(img, eps, 'wrap')
    new_img = np.empty(img.shape)

    it = np.nditer(new_img, flags=['multi_index'])
    with it:
        while not it.finished:
            i = it.multi_index[0]
            j = it.multi_index[1]
            begi = i
            endi = i + 2*eps + 1
            begj = j
            endj = j + 2*eps + 1
            new_img[i,j] = np.sum(pad_img[begi:endi,begj:endj] * weights)

            it.iternext()

    new_img = limiarization(new_img, thresh_ini)

    return new_img

# edit image based on median of surrounding pixels
def filtr2DMedian(img, filter_siz):

    new_img = np.empty(img.shape)
    eps = filter_siz // 2
    pad_img = np.pad(img, eps, 'constant')

    it = np.nditer(new_img, flags=['multi_index'])
    with it:
        while not it.finished:
            i = it.multi_index[0]
            j = it.multi_index[1]
            begi = i
            endi = i + 2*eps + 1
            begj = j
            endj = j + 2*eps + 1

            new_img[i,j] = \
                np.median(pad_img[begi:endi, begj:endj], axis=None)

            it.iternext()

    return new_img

# nomalize numpy ndarray to [0,2^8-1]
def normalize(matrix):
    max = np.amax(matrix)
    min = np.amin(matrix)
    matrix = (matrix - min) / (max - min)
    matrix *= ((2**8)-1)

    return matrix

# calculate error
def RMSE(ref,new):
    new = normalize(new)
    error = np.sqrt(np.sum((ref - new)**2) / ref.size)
    return error

def main():
    # 1. Parameter input

    # filename for reference image I
    filename = str(input()).rstrip()
    # print(filename)
    # # change for custom execution
    img = imageio.imread("Casos_Teste/" + filename)

    # read image file
    # img = imageio.imread(filename)

    # # plot for curiosity
    # plt.imshow(img, cmap='gray')
    # plt.show()

    # Choice of method 1 - limiarization  2 - 1D filtering
    # 3 - 2D filtering with limiarization 4 - 2D median filter
    choice = int(input())

    # parameters for each choice
    if choice == 1:
        thresh_ini = int(input())
        new_img = limiarization(img, thresh_ini)

    elif choice == 2:
        filter_siz = int(input())
        weights = np.empty(filter_siz)
        filter = input()
        filter = filter.split()
        for i in range(filter_siz):
            weights[i] = float(filter[i])

        new_img = filtr1D(img, filter_siz, weights)

    elif choice == 3:
        filter_siz = int(input())
        weights = np.empty((filter_siz,filter_siz))
        for i in range(filter_siz):
            filter = input()
            filter = filter.split()
            for j in range(filter_siz):
                weights[i,j] = float(filter[j])

        thresh_ini = int(input())

        new_img = filtr2DLimiar(img, filter_siz,
                                thresh_ini, weights)

    else:
        filter_siz = int(input())
        new_img = filtr2DMedian(img, filter_siz)



    print("%.4f" % RMSE(img,new_img))
    plt.imshow(new_img, cmap='gray')
    plt.show()
if __name__ == '__main__':
    main()

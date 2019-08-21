'''
Image Processing
case generator for Assignment 3

Eric Ribero Augusto
no.USP 10295242
'''

import numpy as np
import imageio
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

def noise(img):

    noise = np.multiply(np.random.normal(0.0, 0.1, img.shape), 255)
    img_noise = np.clip(img.astype(int) + noise, 0, 255)
    return img_noise.astype(np.uint8)

# def blur(img):
#
#     # add padding
#     size = 2
#     max_side = max(img.shape)
#
#     while size < max_side:
#         size *=2
#
#     pad_i1 = (size - img.shape[0])//2 + (size - img.shape[0])%2
#     pad_i2 = (size - img.shape[0])//2
#     pad_j1 = (size - img.shape[1]) // 2 + (size - img.shape[1]) % 2
#     pad_j2 = (size - img.shape[1])//2
#     filt_img = np.pad(img, ((pad_i1,pad_i2),(pad_j1,pad_j2)), 'constant')
#
#     # fourier transform image
#     fft_img = np.fft.fft2(filt_img)
#     fft_img = np.fft.fftshift(fft_img)
#
#     # square lowpass filter
#     # side = size//8
#     # filter = np.zeros(fft_img.shape,dtype=np.cdouble)
#     # filter[size//2 - side : size//2 + side,
#     #         size//2 - side : size//2 + side] = 1.0
#
#     # gaussian lowpass filter
#     sigmax, sigmay = 30, 30
#     cy, cx = fft_img.shape[0] / 2, fft_img.shape[1] / 2
#     x = np.linspace(0, fft_img.shape[0], fft_img.shape[0])
#     y = np.linspace(0, fft_img.shape[1], fft_img.shape[1])
#     X, Y = np.meshgrid(x, y)
#     filter = np.exp(-(((X - cx) / sigmax) ** 2 + ((Y - cy) / sigmay) ** 2))
#
#     # apply filter(multiplication in freq. domain)
#     fft_img = fft_img * filter

    # # inverse fourier
    # filt_img = np.abs(np.fft.ifft2(fft_img)).astype(np.uint8)
    # filt_img = filt_img[pad_i1:filt_img.shape[0] - pad_i2,
    #                   pad_j1:filt_img.shape[1] - pad_j2]

    # return filt_img

# Image padding for fourier transform
def pad_preftt2(img, shape):

    size = 2
    max_side = max(shape)
    while size < max_side:
        size *=2

    pad_i = size - img.shape[0]
    pad_j = size - img.shape[1]

    return np.pad(img,((0,pad_i),(0,pad_j)), 'constant')

# Image unpadding for inverse fft output
def unpad_postftt2(padimg, orig_shape):
    return padimg[0:orig_shape[0], 0:orig_shape[1]]


def blur(img, k, sigma):


    arx = np.arange((-k // 2) + 1.0, (k // 2) + 1.0)
    x, y = np.meshgrid(arx, arx)
    filt = np.exp(-(1 / 2) * (np.square(x) + np.square(y)) / np.square(sigma))
    filt = filt / np.sum(filt)

    img_pad = pad_preftt2(img, img.shape)
    filt_pad = pad_preftt2(filt, img.shape)

    img_fft = np.multiply(np.fft.fft2(img_pad), np.fft.fft2(filt_pad))
    img_new = unpad_postftt2(np.fft.ifft2(img_fft).real, img.shape)


    return img_new


# nomalize numpy ndarray to [0,2^8-1]
def normalize(matrix):
    max = np.amax(matrix)
    min = np.amin(matrix)
    matrix = (matrix - min) / (max - min)
    matrix *= ((2**8)-1)

    return matrix



def main():

    mpath = "dip_t03_imagefiles/"

    files = [f for f in listdir(mpath) if isfile(join(mpath,f))]
    for file in files:
        print(file)
        # get image
        img = imageio.imread(mpath+file, as_gray=True)
        # apply blur to original and save it
        blur_path = mpath+"blur/blur_"+ file
        imageio.imwrite(blur_path, blur(img, 5, 1.0))

        # apply noise to original and save it
        noise_path = mpath+"noise/noise_"+ file
        imageio.imwrite(noise_path, noise(img))



if __name__ == '__main__':
    main()
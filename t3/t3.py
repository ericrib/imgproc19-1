'''
Assignment 3: Image Restoration
'''

import numpy as np
import imageio
# import matplotlib.pyplot as plt
import scipy.fftpack as spfft

# 1 Adaptive Denoising
def denoise(deg, f_size, mode, gamma):

    denoised = np.copy(deg)
    size_i = deg.shape[0] - f_size
    size_j = deg.shape[1] - f_size
    dx = f_size//2

    dispn = np.std(deg[: deg.shape[0] // 6, : deg.shape[1] // 6])
    if dispn == 0:
        dispn = 1.0

    for conti in range(size_i ):
        i = conti + dx
        for contj in range(size_j):
            j = contj + dx

            # calculate measures for function
            if mode == 'average':
                # mean and std dev
                central = np.mean(deg[i-dx : i+dx, j-dx : j+dx ])
                displ = np.std(deg[i - dx: i + dx, j - dx: j + dx])
            else:
                # median and IQR
                q = np.percentile(deg[i - dx: i + dx, j - dx: j + dx], [75, 50, 25])
                central = q[1]
                displ = q[0] - q[2]


            if displ == 0:
                displ = dispn

            # denoise image
            denoised[i,j] = deg[i,j] - gamma*(dispn/displ)*(deg[i,j]-central)

    # normalization
    denoised = (denoised - denoised.min())/(denoised.max() - denoised.min())
    denoised = denoised*(deg.max()-deg.min()) + deg.min()

    return denoised

# Degradation Function
def gaussian_filter(k, sigma):
     arx = np.arange ((-k // 2) + 1.0, (k // 2) + 1.0)
     x,y = np.meshgrid(arx, arx)
     filt = np.exp(-(1/2)*( np.square(x) + np.square(y)) / np.square(sigma))
     return filt/np.sum(filt)

# Image padding for fourier transform
def pad_ftt2(img, shape):

    pad_i1 = (shape[0] - img.shape[0])//2 + (shape[0] - img.shape[0])%2
    pad_i2 = (shape[0] - img.shape[0])//2
    pad_j1 = (shape[1] - img.shape[1])//2 + (shape[1] - img.shape[1])%2
    pad_j2 = (shape[1] - img.shape[1])//2

    return np.pad(img,((pad_i1,pad_i2),(pad_j1,pad_j2)), 'constant')

# 2 Constrained Least Squares Filtering
def deblur(deg, f_size, sigma, gamma):

    p = np.array([[ 0.0, -1.0,  0.0],
                  [-1.0,  4.0, -1.0],
                  [ 0.0, -1.0,  0.0]])

    h = gaussian_filter(f_size, sigma)

    # Fourier-transform matrices

    DG =spfft.fftn(deg)
    # DG = spfft.fftshift(DG)
    # plt.imshow(np.log(np.abs(DG.real)+1), cmap='gray')
    # plt.show()

    P = spfft.fftn(pad_ftt2(p, deg.shape))
    # P = spfft.fftshift(P)
    # plt.imshow(np.log(np.abs(P.real)+1), cmap='gray')
    # plt.show()

    H = spfft.fftn(pad_ftt2(h, deg.shape))
    # H = spfft.fftshift(H)
    # plt.imshow(np.log(np.abs(H.real)+1), cmap='gray')
    # plt.show()

    # Calculate CLS
    F = np.multiply(np.conjugate(H)/(np.square(np.abs(H)) + gamma*(np.square(np.abs(P)))), DG)
    # F = spfft.fftshift(F)
    # plt.imshow(np.log(np.abs(F.real)+1), cmap='gray')
    # plt.show()

    # Inverse Fourier
    deblurred = spfft.ifftshift(spfft.ifftn(F).real)


    return deblurred

# nomalize numpy ndarray to [0,2^8-1]
def normalize(matrix):
    max = np.amax(matrix)
    min = np.amin(matrix)
    matrix = (matrix - min) / (max - min)
    matrix *= ((2**8)-1)

    return matrix

# calculate error
def RMSE(ref,new):
    error = np.sqrt(np.sum((ref - new)**2) / ref.size)
    return error

def main():

    # filename for reference image ref and degraded image deg
    ref_name = str(input()).rstrip()
    deg_name = str(input()).rstrip()
    # reading images
    ref = imageio.imread(ref_name, as_gray=True).astype(np.uint8)
    deg = imageio.imread(deg_name, as_gray=True).astype(np.float64)


    # plot for curiosity
    # plt.imshow(ref, cmap='gray')
    # plt.show()
    #
    # plt.imshow(deg, cmap='gray')
    # plt.show()

    # filter type 1 - denoising 2- deblurring
    filter = int(input())

    # parameter gamma
    gamma = np.float64(input())
    # size  of filter kernel type 1 - denoising
    # or size of  degradation funciton  2- deblurring
    f_size = int(input())

    # restauration calling
    if filter == 1:
        mode = str(input()).rstrip()
        new_img = denoise(deg, f_size, mode, gamma).astype(float)
    else :
        sigma = np.float64(input())
        new_img = deblur(deg, f_size, sigma, gamma).astype(float)


    # print error
    print("%.3f" % RMSE(ref,new_img))
    # plt.imshow(new_img, cmap='gray')
    # plt.show()

if __name__ == '__main__':
    main()
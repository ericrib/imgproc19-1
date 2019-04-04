'''
Image Processing
Assignment 1

Eric Ribero Augusto
no.USP 10295242
'''

import numpy as np
#import imageio
# import matplotlib.pyplot as plt
import random


# generate image acording to
# f(x,y) = (x*y + 2*y)
def function1(scene):
    it = np.nditer(scene, flags=['multi_index'],
                   op_flags=['readwrite'])
    with it:
        while not it.finished:
            it[0] = (it.multi_index[1]*it.multi_index[0] +
                     2*it.multi_index[1])
            it.iternext()

    return scene

# generate image acording to
# f(x,y) = |cos(x/Q) + 2*sin(y/Q)|
def function2(scene, Q):
    it = np.nditer(scene, flags=['multi_index'],
                   op_flags=['readwrite'])
    with it:
        while not it.finished:
            it[0] = abs(np.cos(it.multi_index[0]/Q) +
                       2*np.sin(it.multi_index[1]/Q))
            it.iternext()

    return scene

# generate image acording
# f(x,y) = |3*(x/Q) - (y/Q)**(1/3)|
def function3(scene, Q):
    it = np.nditer(scene, flags=['multi_index'],
                   op_flags=['readwrite'])
    with it:
        while not it.finished:
            it[0] = abs(3*(it.multi_index[0]/Q) -
                       (it.multi_index[1]/Q)**(1/3))
            it.iternext()


    return scene

# generate image acording
# f(x,y) = random(0,1,S)
def function4(scene,S):
    random.seed(S)
    it = np.nditer(scene, flags=['multi_index'],
                   op_flags=['readwrite'])
    with it:
        while not it.finished:
            it[0] = random.random()
            it.iternext()

    return scene

# generate image acording to
# f(x,y) = randomwalk(S)
def function5(scene,S):

    random.seed(S)
    C = scene.shape[0]
    steps = 1 + C**2
    x,y = 0,0
    scene[x,y] = 1
    for k in range(steps):
        dx = random.randint(-1,1)
        dy = random.randint(-1,1)
        x = ((x + dx) % C)
        y = ((y + dy) % C)
        scene[x, y] = 1

    return scene


# nomalize numpy ndarray to [0,2^16-1]
def normalize(matrix):
    max = np.amax(matrix)
    min = np.amin(matrix)
    matrix = (matrix - min) / (max - min)
    matrix *= ((2**16)-1)
    return matrix

def digitise(scene, N, B):

    # # normalize scene to 2^8
    # max = np.amax(scene)
    # min = np.amin(scene)
    # scene = (scene - min) / (max - min)
    # scene *= ((2**8)-1)
    # # bitshift B
    # scene = scene.astype(np.uint8)
    # scene = np.right_shift(scene, (8-B))
    # scene = np.left_shift(scene, (8-B))


    # reduce resolution CxC to NxN
    C = scene.shape[0]
    ds = C//N
    # digital = np.ones((N,N),dtype=np.uint8)
    digital = np.ones((N,N))

    it = np.nditer(digital, flags=['multi_index'],
                   op_flags=['readwrite'])
    with it:
        while not it.finished:
            i = it.multi_index[0] * ds
            j = it.multi_index[1] * ds
            it[0] = scene[i, j]
            it.iternext()

    # normalize scene to 2^8
        max = np.amax(digital)
        min = np.amin(digital)
        digital = (digital - min) / (max - min)
        digital *= ((2 ** 8) - 1)
        digital = digital.astype(np.uint8)
        # bitshift B
        digital = digital // (2 ** (8 - B))
        # digital = digital * (2 ** (8 - B))

    return digital

def RMSE(g,R):
    error = np.sqrt(np.sum(g-R))
    return error


def main():
    # 1. Parameter input

    # filename for reference image r
    filename = str(input()).rstrip()
    #print(filename)
    #change for custom execution
    # R = np.load("example/" + filename)
    R = np.load(filename)
    R = R.astype(np.uint8)

    # plot for curiosity
    # plt.imshow(R,cmap='gray')
    # plt.show()

    # lateral size of scene
    C = int(input())
    # function to generate scene image
    f = int(input())
    # parameter for functions 2 and 3
    Q = int(input())
    # size of digital image
    N = int(input())
    # number of bits per pixel [1,8]
    B = int(input())
    # seed for random function
    S = int(input())

    scene = np.zeros((C,C))
    # 2. Generate scene image f
    funDictionary = {1: lambda: function1(scene),
                       2: lambda: function2(scene,Q),
                       3: lambda: function3(scene,Q),
                       4: lambda: function4(scene,S),
                       5: lambda: function5(scene,S)
                       }
    scene = funDictionary.get(f, lambda: 'Invalid')()
    scene = normalize(scene)
    # plt.imshow(scene, cmap='gray')
    # plt.show()


    # 3. Geneate digital image g
    g = digitise(scene, N, B)
    # plt.imshow(g, cmap='gray')
    # plt.show()


    # 4. Compare g with r
    # 5. Print RSE/RMSE between g and
    print("%.4f" % RMSE(g,R))

    #
    # print(g.shape)
    # print(R.shape)


if __name__ == '__main__':
    main()

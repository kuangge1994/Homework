import cv2
import numpy as np 
import random
import math
from scipy import ndimage

# Function of Gaussian Noise, k is the coefficent of Gaussian Noise
def GaussianNoise(src, means, sigma, k):
    Img_noise = src
    height = src.shape[0]
    width =  src.shape[1]

    im_r = np.zeros((height,width),np.uint8)
    im_g = np.zeros((height,width),np.uint8)
    im_b = np.zeros((height,width),np.uint8)

    for i in range(height):
        for j in range(width):
            noise = k * random.gauss(means,sigma)
            for c in range(3):      #BGR channels
                Img_noise[i][j][c] = Img_noise[i][j][c] + noise
                if Img_noise[i][j][c] < 0:
                    Img_noise[i][j][c] = 0
                elif Img_noise[i][j][c] > 255:
                    Img_noise[i][j][c] = 255
                
                if c == 0:
                    im_b[i][j] = Img_noise[i][j][c]
                elif c == 1:
                    im_g[i][j] = Img_noise[i][j][c]
                else:
                    im_r[i][j] = Img_noise[i][j][c]
    
    return Img_noise, im_r, im_g, im_b

# Define Gaussian Filter
def Gauss_filter(im_r, im_g, im_b):
    # define a Gaussian kernel(5*5)
    kernel_gau = np.array([
        [1,4,6,4,1],
        [4,16,24,16,4],
        [6,24,36,24,6],
        [4,16,24,16,4],
        [1,4,6,4,1]
    ])
    h = im_r.shape[0]
    w = im_r.shape[1]
    
    # padding = 2
    im_tmpr = np.zeros((h+4,w+4),np.uint8)
    im_tmpg = np.zeros((h+4,w+4),np.uint8)
    im_tmpb = np.zeros((h+4,w+4),np.uint8)
    
    im_gauss = np.zeros((h,w,3),np.uint8)

    # obtain images after padding
    im_tmpr[2:h+2,2:w+2] = im_r[:,:]
    im_tmpg[2:h+2,2:w+2] = im_g[:,:]
    im_tmpb[2:h+2,2:w+2] = im_b[:,:]
    
    # generate images after gaussian filter
    for i in range(2,h+2):
        for j in range(2,w+2):
            im_gauss[i-2][j-2][0] = np.sum(im_tmpb[i-2:i+3,j-2:j+3] * (kernel_gau/256))   #for b channels
            im_gauss[i-2][j-2][1] = np.sum(im_tmpg[i-2:i+3,j-2:j+3] * (kernel_gau/256))   #for g channels
            im_gauss[i-2][j-2][2] = np.sum(im_tmpr[i-2:i+3,j-2:j+3] * (kernel_gau/256))   #for r channels
    
    cv2.imwrite('GaussianFilter.png', im_gauss)
    cv2.imshow('Image after Gaussian Filter', im_gauss)

# Define Median Filter
def Median_filter(im_r, im_g, im_b, kernel_size):
    h = im_r.shape[0]
    w = im_r.shape[1]

    # compute number of paddings
    # W2=(W1 - kernel_size + 2*pad)/stride + 1; stride = 1 here
    pad = int((kernel_size - 1) / 2)
    dev = kernel_size - pad     #the deviation between kernel and pad
    im_tmpr = np.zeros((h+2*pad,w+2*pad),np.uint8)
    im_tmpg = np.zeros((h+2*pad,w+2*pad),np.uint8)
    im_tmpb = np.zeros((h+2*pad,w+2*pad),np.uint8)

    im_median = np.zeros((h,w,3),np.uint8)

    # obtain images after padding
    im_tmpr[pad:(h+pad),pad:(w+pad)] = im_r[:,:]
    im_tmpg[pad:(h+pad),pad:(w+pad)] = im_g[:,:]
    im_tmpb[pad:(h+pad),pad:(w+pad)] = im_b[:,:]

    # generate images after median filter
    for i in range(pad,h+pad):
        for j in range(pad,w+pad):
            im_median[i-pad][j-pad][0] = np.median(im_tmpb[i-pad:i+dev,j-pad:j+dev])    #for b channels
            im_median[i-pad][j-pad][1] = np.median(im_tmpg[i-pad:i+dev,j-pad:j+dev])    #for g channels
            im_median[i-pad][j-pad][2] = np.median(im_tmpr[i-pad:i+dev,j-pad:j+dev])    #for r channels
    
    cv2.imwrite('MedianFilter.png', im_median)
    cv2.imshow('Image after Median Filter', im_median)

# Define Bilateral Filter
def Bilateral_filter(im_r, im_g, im_b, kernel_size):
    h = im_r.shape[0]
    w = im_r.shape[1]

    # compute number of paddings
    # W2=(W1 - kernel_size + 2*pad)/stride + 1; stride = 1 here
    pad = int((kernel_size - 1) / 2)
    dev = kernel_size - pad     #the deviation between kernel and pad

    # define domain kernelfloat_
    d = np.zeros((kernel_size,kernel_size),np.float)
    # define range kernel
    r = np.zeros((kernel_size,kernel_size),np.float)
    # define weight kernel
    weight = np.zeros((kernel_size,kernel_size),np.float)
    # restore temp values for domain, codomain and weight
    tmp = np.zeros((kernel_size,kernel_size),np.float)

    im_tmpr = np.zeros((h+2*pad,w+2*pad),np.uint8)
    im_tmpg = np.zeros((h+2*pad,w+2*pad),np.uint8)
    im_tmpb = np.zeros((h+2*pad,w+2*pad),np.uint8)

    im_blt = np.zeros((h,w,3),np.uint8)     # the result of bilateral filter

    # obtain images after padding
    im_tmpr[pad:(h+pad),pad:(w+pad)] = im_r[:,:]
    im_tmpg[pad:(h+pad),pad:(w+pad)] = im_g[:,:]
    im_tmpb[pad:(h+pad),pad:(w+pad)] = im_b[:,:]

    # compute domain kernel, kernel_size must be Odd numbers
    # domain kernel: d(i,j,k,l) = exp(-((i-k)^2 + (j-l)^2) / (2*var(d)))
    ker_mean = int((kernel_size - 1) / 2)
    for i in range(kernel_size):
        for j in range(kernel_size):
            tmp[i][j] = math.sqrt((i-ker_mean)**2 + (j-ker_mean)**2)        
    for m in range(kernel_size):
        for n in range(kernel_size):
            d[m][n] = math.exp(-((m-ker_mean)**2 + (n-ker_mean)**2) / (2*np.var(tmp)))
    d = np.around(d, decimals=1)

    # generate images after bilateral filter
    for i in range(pad,h+pad):
        for j in range(pad,w+pad):
            # range kernel: r(i,j,k,l) = exp(-(||f(i,j) - f(k,l)||^2) / (2*var(r)))
            # weight: weight = range * domain
            # compute range kernel, weight kernel, filter result for b channel
            tmp[:,:] = im_tmpb[i-pad:i+dev,j-pad:j+dev]
            if np.var(tmp) == 0:
                im_blt[i-pad][j-pad][0] = im_tmpb[i][j]
            else:
                for m in range(kernel_size):
                    for n in range(kernel_size):
                        r[m][n] = math.exp(- (tmp[m][n] - tmp[ker_mean][ker_mean])**2 / (2*np.var(tmp)))
                        #r[m][n] = math.exp(- (tmp[m][n] - tmp[ker_mean][ker_mean]) / (2*0.01))
                r = np.around(r, decimals=1)
                weight = d * r
                im_blt[i-pad][j-pad][0] = np.uint8(np.sum(im_tmpb[i-pad:i+dev,j-pad:j+dev]*weight) / np.sum(weight))

            # compute range kernel, weight kernel, filter result for g channel
            tmp[:,:] = im_tmpg[i-pad:i+dev,j-pad:j+dev]
            if np.var(tmp) == 0:
                im_blt[i-pad][j-pad][1] = im_tmpg[i][j]
            else:
                for m in range(kernel_size):
                    for n in range(kernel_size):
                        r[m][n] = math.exp(- (tmp[m][n] - tmp[ker_mean][ker_mean])**2 / (2*np.var(tmp)))
                        #r[m][n] = math.exp(- (tmp[m][n] - tmp[ker_mean][ker_mean]) / (2*0.01))
                r = np.around(r, decimals=1)
                weight = d * r
                im_blt[i-pad][j-pad][1] = np.uint8(np.sum(im_tmpg[i-pad:i+dev,j-pad:j+dev]*weight) / np.sum(weight))

            # compute range kernel, weight kernel, filter result for r channel
            tmp[:,:] = im_tmpr[i-pad:i+dev,j-pad:j+dev]
            if np.var(tmp) == 0:
                im_blt[i-pad][j-pad][2] = im_tmpr[i][j]
            else:
                for m in range(kernel_size):
                    for n in range(kernel_size):
                        r[m][n] = math.exp(- (tmp[m][n] - tmp[ker_mean][ker_mean])**2 / (2*np.var(tmp)))
                        #r[m][n] = math.exp(- (tmp[m][n] - tmp[ker_mean][ker_mean]) / (2*0.01))
                r = np.around(r, decimals=1)
                weight = d * r
                im_blt[i-pad][j-pad][2] = np.uint8(np.sum(im_tmpr[i-pad:i+dev,j-pad:j+dev]*weight) / np.sum(weight))
    
    cv2.imwrite('BilateralFilter.png', im_blt)
    cv2.imshow('Image after Bilateral Filter', im_blt)

def main():
    img_path = "./images/HW2Pic2.png"
    im_source = cv2.imread(img_path)
    h = im_source.shape[0]      #height of im_source
    w = im_source.shape[1]      #width of im_source

    # Obtain BGR channels of im_noise
    im_R = np.zeros((h,w),np.uint8)
    im_G = np.zeros((h,w),np.uint8)
    im_B = np.zeros((h,w),np.uint8)

    # Add Gaussian Noise to im_source
    im_noise, im_R, im_G, im_B = GaussianNoise(im_source, 0, 0.1, 1)
    cv2.imwrite('GNoise.png',im_noise)
    cv2.imshow('Image Gaussian Noise', im_noise)

    # Remove noise with Gaussian Filter
    Gauss_filter(im_R, im_G, im_B)
    # Remove noise with Median Filter
    Median_filter(im_R, im_G, im_B, 5)
    # Remove noise with Bilateral Filter
    Bilateral_filter(im_R, im_G, im_B, 5)

    cv2.waitKey()

if __name__ == '__main__':
    main()
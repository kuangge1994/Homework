import numpy as np
from PIL import Image
from matplotlib import pyplot as plt 
from scipy.misc import toimage
from time import sleep

# Histogram equalization function
# Parameters im_arr ——> im_gray, type(array); cdh——>comulative distribution
def HistImageAvg(im_arr, cdh):
    im_num = len(im_arr[0]) * len(im_arr)
    color_tran = []     #restore pixels of transformed

    # compute pixels of transformed by comulative distribution function
    for i in range(256):
        # the shape of cdh is 255, while i should count to 255(as 256 numbers)
        if i > len(cdh) - 1:
            color_tran.append(color_tran[i-1])
            break
        temp = cdh[i]*255 / im_num
        color_tran.append(temp)
    
    # Generate image data of Histogram equalization
    im_tran = []
    for itemL in im_arr:
        tmp_line = []
        for item in itemL:
            tmp_line.append(color_tran[item])
        im_tran.append(tmp_line)
    
    return im_tran

def main():
    # Get a picture named 'HW2Pic1.png'
    im_source = Image.open('images/HW2Pic1.png')
    im_source.show()

    # Initial channels of RGB
    im_RGB = np.array(im_source)
    im_R = []
    im_G = []
    im_B = []
    i_row = 0

    # Detach three primary color channels
    for itemL in im_RGB:
        im_R.append([])
        im_G.append([])
        im_B.append([])
        for itemC in itemL:
            im_R[i_row].append(itemC[0])
            im_G[i_row].append(itemC[1])
            im_B[i_row].append(itemC[2])
        i_row += 1

    # Convert RGB to gray for addressing the problem
    im_gray = im_source.convert('L')
    im_gray.show()
    arr_gray = np.array(im_gray)
    im_hist, bins = np.histogram(arr_gray.flatten(), range(256))    #obtain the histogram
    plt.figure(num=1,figsize=(50,10),dpi=40)
    plt.subplot(1,5,1)
    plt.hist(arr_gray.flatten(), range(256))
    plt.title('Histogram of Gray')
    # plt.show()
    cdh = im_hist.cumsum()  #obtain the comulative distribution histogra
    plt.subplot(1,5,2)
    plt.hist(cdh.flatten(), range(256))
    plt.title('Histogram of Comulative distribution')
    # plt.show()

    # Get three channels of processed
    im_r = HistImageAvg(im_R, cdh)
    im_g = HistImageAvg(im_G, cdh)
    im_b = HistImageAvg(im_B, cdh)
    plt.subplot(1,5,3)
    plt.hist(np.array(im_r).flatten(), range(256), color='red')
    plt.title('The result of R')
    plt.subplot(1,5,4)
    plt.hist(np.array(im_g).flatten(), range(256), color='green')
    plt.title('The result of G')
    plt.subplot(1,5,5)
    plt.hist(np.array(im_b).flatten(), range(256), color='blue')
    plt.title('The result of B')
    plt.savefig('images/Histogram.png')
    plt.show()
    # sleep(3)
    # plt.close()

    # Combine the data of RGB
    pic = []
    length1 = len(im_r)
    for i in range(length1):
        tmp_line = []
        length2 = len(im_r[i])
        for j in range(length2):
            tmp_point = [im_r[i][j],im_g[i][j],im_b[i][j]]
            tmp_line.append(tmp_point)
        pic.append(tmp_line)
    
    Picture = toimage(np.array(pic), 255)
    Picture.show()
    Picture.save('images/HW2Pic1_HE.png')

if __name__ == '__main__':
    main()
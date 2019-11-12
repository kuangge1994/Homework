from matplotlib import pyplot as plt
import numpy as np 
import cv2 as cv
import math

# Compute information entropy for the fusion images]
def Info_cros(img):
    tmp = []
    for i in range(256):
        tmp.append(0)
    val = 0
    k = 0
    res = 0

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    im = np.array(img)
    for i in range(len(im)):
        for j in range(len(im[i])):
            val = im[i][j]
            tmp[val] = float(tmp[val] + 1)
            k = float(k + 1)
    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)
    for i in range(len(tmp)):
        if tmp[i] == 0:
            res = res
        else:
            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
    print("The information entropy of the result is {0}".format(res))

# Fusion and Stiching function
def Fusion(img_src1, img_src2, kpt1, kpt2, key_good):
    rows, cols = img_src1.shape[:2]

    MIN_MATCH_COUNT = 10
    if len(key_good) > MIN_MATCH_COUNT:
        # obtain positions of keypoints of img_src1 and img_src2
        pt_src1 = np.float32([kpt1[m[0].queryIdx].pt for m in key_good]).reshape(-1, 1, 2)  #queryIdx for img_src1
        pt_src2 = np.float32([kpt2[m[0].trainIdx].pt for m in key_good]).reshape(-1, 1, 2)  #trainIdx for img_src2

        # Compute homograph
        M, mask = cv.findHomography(pt_src1, pt_src2, cv.RANSAC, 5.0)
        img_wrap = cv.warpPerspective(img_src2, np.array(M), (img_src2.shape[1], img_src2.shape[0]), flags=cv.WARP_INVERSE_MAP)
        wrap = cv.cvtColor(img_wrap,cv.COLOR_BGR2RGB)
        plt.imshow(wrap)
        plt.savefig('./images/Homograph_img.jpg')
        plt.show()

        for col in range(cols):
            if img_src1[:, col].any() and img_wrap[:, col].any():
                left = col
                break
        for col in range(cols-1, 0, -1):
            if img_src1[:, col].any() and img_wrap[:, col].any():
                right = col
                break
    
    # Create images after fusion
    img_st = np.zeros([rows, cols, 3], np.uint8)
    for row in range(rows):
        for col in range(cols):
            if not img_src1[row, col].any():
                img_st[row, col] = img_wrap[row, col]
            elif not img_wrap[row, col].any():
                img_st[row, col] = img_src1[row, col]
            else:
                imglen_src1 = float(abs(col - left))
                imglen_src2 = float(abs(col - right))
                a = imglen_src1 / (imglen_src1 + imglen_src2)
                img_st[row, col] = np.clip(img_src1[row, col] * (1-a) + img_wrap[row, col] * a, 0, 255)
    
    # Computer Information entropy
    Info_cros(img_st)

    # Show the result
    img_st = cv.cvtColor(img_st, cv.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(img_st)
    plt.savefig('./images/Fusion_img.jpg')
    plt.show()

# Define a function of ORB(Oriented Fast and Rotated BRIEF)
def ORB_Match(img1, img2):
    # Expanding with value 0
    expand_right = img1.shape[1] 
    img_src1 = cv.copyMakeBorder(img1, 100, 100, 0, expand_right, cv.BORDER_CONSTANT, value = (0,0,0))
    img_src2 = cv.copyMakeBorder(img2, 100, 100, 0, expand_right, cv.BORDER_CONSTANT, value = (0,0,0))

    # Convert source images to gray images
    gray_src1 = cv.cvtColor(img_src1, cv.COLOR_BGR2GRAY)
    gray_src2 = cv.cvtColor(img_src2, cv.COLOR_BGR2GRAY)

    # Define a generator of ORB
    orb = cv.ORB_create()

    # Obtain the keypoints and descriptors
    keypoint1, descriptor1 = orb.detectAndCompute(gray_src1, None)
    keypoint2, descriptor2 = orb.detectAndCompute(gray_src2, None)

    # Horizontal merge two gray images
    hmerge_gray = np.hstack((gray_src1, gray_src2))
    plt.imshow(hmerge_gray, 'gray')
    plt.title('Gray images')
    plt.savefig('./images/Gray_img.jpg')
    plt.show()
    # cv.imshow('Gray images', hmerge_gray)
    # cv.waitKey(0)

    # Draw keypoints on source images
    im_src1 = cv.drawKeypoints(img_src1, keypoint1,img_src1, color=(255,0,255))
    im_src2 = cv.drawKeypoints(img_src2, keypoint2,img_src2, color=(255,0,255))

    # Horizontal merge two keypoints images
    hmerge_key = np.hstack((im_src1, im_src2))
    hmerge_key = cv.cvtColor(hmerge_key, cv.COLOR_BGR2RGB)
    plt.imshow(hmerge_key)
    plt.title('KeyPoints')
    plt.savefig('./images/Key_img.jpg')
    plt.show()
    # cv.imshow('KeyPoints', hmerge_key)
    # cv.waitKey(0)

    # Match with BFMatcher
    bfmatcher = cv.BFMatcher()
    matches = bfmatcher.knnMatch(descriptor1, descriptor2, k=2)

    # Obtain the ratio of good points
    key_good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            key_good.append([m])

    img = cv.drawMatchesKnn(img_src1, keypoint1, img_src2, keypoint2, key_good, None, flags=2)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title('ORB')
    plt.savefig('./images/ORBMatch_img.jpg')
    plt.show()
    # cv.imshow('ORB', img)
    # cv.waitKey(0)

    # Fusion source images: img_src1, img_src2
    img_1 = cv.copyMakeBorder(img1, 100, 100, 0, expand_right, cv.BORDER_CONSTANT, value = (0,0,0))
    img_2 = cv.copyMakeBorder(img2, 100, 100, 0, expand_right, cv.BORDER_CONSTANT, value = (0,0,0))
    Fusion(img_1, img_2, keypoint1, keypoint2, key_good)

def main():
    path_src1 = '.\\images\\HW3Pic1.jpg'
    path_src2 = '.\\images\\HW3Pic2.jpg'

    # Read images from './images/'
    img_src1 = cv.imread(path_src1)  #Reference image
    img_src2 = cv.imread(path_src2)  #Stiching image

    # Get feature points and match them
    ORB_Match(img_src1, img_src2)
    

if __name__ == '__main__':
    main()
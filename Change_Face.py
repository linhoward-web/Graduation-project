# -*- coding: utf-8 -*-
"""
Created on Mon May 27 18:25:10 2019

@author: JAVEN
"""

import cv2
import dlib
import numpy
import matplotlib.pyplot as plt   
#import sys
 
PREDICTOR_PATH = "C:/shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1 
FEATHER_AMOUNT = 11
 
FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))
 
# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                               RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)
 
# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]
 
# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.  (pupillary:瞳距)
COLOUR_CORRECT_BLUR_FRAC = 0.6
 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
 
class TooManyFaces(Exception):
    pass
 
class NoFaces(Exception):
    pass
 
def get_landmarks(im):
    rects = detector(im, 1)
    
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces
 
    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
 
def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im
 
def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)
 
def get_face_mask(im, landmarks):
    im = numpy.zeros(im.shape[:2], dtype=numpy.float64)
 
    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)
 
    im = numpy.array([im, im, im]).transpose((1, 2, 0))
 
    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)
 
    return im
    
def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimized.
    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
 
    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)
 
    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
 
    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2
 
    U, S, Vt = numpy.linalg.svd(points1.T * points2)
 
    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T
 
    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])
 
def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    w = im.shape[1]
    h = im.shape[0]
    print("%s, w=%d, h=%d" %(fname, w, h))
 
    #把小边缩放到800
    min_wh = min(w,h)
    ratio = 800.0/min_wh 
 
    new_w = (int)(im.shape[1] * ratio)
    new_h = (int)(im.shape[0] * ratio)
    print("new_w=%d, new_h=%d" %(new_w, new_h))
 
    im = cv2.resize(im, (new_w, new_h))
 
    s = get_landmarks(im)
 
    return im, s
 
def warp_im(im, M, dshape):
    output_im = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im
 
def geteye_rect(imgpath):  
    bgrImg = cv2.imread(imgpath)  
    if bgrImg is None:  
        return False  
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)  
    facesrect = detector(rgbImg, 1)  
    if len(facesrect) <=0:  
        return False  
  
    for k, d in enumerate(facesrect):  
        shape = predictor(rgbImg, d)  
        for i in range(68):  
            pt=shape.part(i)  
            plt.plot(pt.x,pt.y,'ro')  
        plt.imshow(rgbImg)  
        plt.show()  
 
 
def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
                              numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)
 
    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)
 
    return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
                                                im2_blur.astype(numpy.float64))
 
#检测关键点
#im1, landmarks1 = read_im_and_landmarks(sys.argv[1])
#im2, landmarks2 = read_im_and_landmarks(sys.argv[2])
im1, landmarks1 = read_im_and_landmarks('../Ai_fortune_teller/1.jpg') #讀入圖1
im2, landmarks2 = read_im_and_landmarks('../Ai_fortune_teller/test_1_1.jpg') #讀入圖2
 
M = transformation_from_points(landmarks1[ALIGN_POINTS],
                               landmarks2[ALIGN_POINTS])
 
mask = get_face_mask(im2, landmarks2)
#cv2.imwrite('mask.jpg', mask)
warped_mask = warp_im(mask, M, im1.shape)
#cv2.imwrite('warped_mask.jpg', warped_mask)
combined_mask = numpy.max([get_face_mask(im1, landmarks1), warped_mask], axis=0)
 
#cv2.imwrite('combined_mask.jpg', combined_mask)
 
warped_im2 = warp_im(im2, M, im1.shape)
cv2.imwrite('warped_im2.jpg', warped_im2)
warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)
 
output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
cv2.imwrite('warped_corrected_im2.jpg', warped_corrected_im2)
cv2.imwrite('output.jpg', output_im) #合成圖 圖1的臉 圖2的眼鼻
 
#显示人脸关键点
#geteye_rect('../Ai_fortune_teller/test_0_2.jpg')
#geteye_rect('../Ai_fortune_teller/1.jpg')

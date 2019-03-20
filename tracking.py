# -*- coding: utf-8 -*
import cv2
import numpy as np
import math
import frst

# golbal variable
gray_threshold = 40
x1_lx=120
y1_lx=200
x2_lx=120
y2_lx=200
x1_rx=120
y1_rx=200
x2_lx=120
y2_lx=200
def find_eye_region(mat1):
    x1_rx=120
    y1_rx=200

    weight, height, depth = mat1.shape
    eye_region_width = int(weight * 30/100.0)
    eye_region_height = int(height * 10/100.0)
    eye_region_top = int(height * 35/100.0)
    eye_region_left = int(weight * 13/100.0)
    left_eye_x_prior=38
    left_eye_y_prior=105
    right_eye_x_prior=172
    right_eye_y_prior=105
    left_eye_x = eye_region_left
    left_eye_y = eye_region_top
    right_eye_x = int(weight - eye_region_width - eye_region_left)
    right_eye_y = eye_region_top

    left_eye_mat = mat1[left_eye_y:left_eye_y +
                        eye_region_height, left_eye_x:left_eye_x + eye_region_width]
    right_eye_mat = mat1[right_eye_y:right_eye_y +
                         eye_region_height, right_eye_x:right_eye_x + eye_region_width]
    img=r"C:\Users\PIYUSH\Desktop\pic.jpg"
    l_x=abs(left_eye_x_prior-left_eye_x)
    l_y=abs(left_eye_y_prior-left_eye_y)
    r_x=abs(right_eye_x_prior-right_eye_x)
    r_y=abs(right_eye_y_prior-right_eye_y)
    if(l_x):
        x2_lx=l_x*100
        y2_lx=l_y*100
        cv2.line(img,(x1_lx,y1_lx),(x2_lx,y2_lx),(0,0,255),2)
        x1_lx=x2_lx
        y1_lx=y2_lx
        cv2.imwrite(r'C:\Users\PIYUSH\Desktop\pic2.jpg',img)
    if(l_y):
        x2_ly=l_x*100
        y2_ly=l_y*100
        cv2.line(img,(x1_ly,y1_ly),(x2_ly,y2_ly),(0,0,255),2)
        x1_ly=x2_ly
        y1_ly=y2_ly
        cv2.imwrite(r'C:\Users\PIYUSH\Desktop\pic2.jpg',img)
    if(r_x):
        x2_rx=r_x*100
        y2_rx=r_y*100
        cv2.line(img,(x1_rx,y1_rx),(x2_rx,y2_rx),(0,255,255),2)
        x1_rx=x2_rx
        y1_rx=y2_rx
        cv2.imwrite(r'C:\Users\PIYUSH\Desktop\pic2.jpg',img)

    if(r_y):

        x2_ry=r_x*100
        y2_ry=r_y*100
        cv2.line(img,(x1_ry,y1_ry),(x2_ry,y2_ry),(0,255,255),2)
        x1_ry=x2_ry
        y1_ry=y2_ry
        cv2.imwrite(r'C:\Users\PIYUSH\Desktop\pic2.jpg',img)


    left_eye_y_prior=left_eye_y
    left_eye_x_prior=left_eye_x
    right_eye_x_prior=right_eye_x
    right_eye_y_prior=right_eye_y
    return left_eye_mat, right_eye_mat


def main():
    vs = cv2.VideoCapture(0)
    width = 640
    vs.set(3, width)
    height = 480
    vs.set(4, height)
    face_detector = cv2.CascadeClassifier(
        'haarcascade_frontalface_default.xml')

    while True:
        ret, pic = vs.read()
        # pic_gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        pic_gray = cv2.split(pic)[2]

        face_boxs = face_detector.detectMultiScale(pic_gray, scaleFactor=1.1,
                                                   minNeighbors=5,
                                                   minSize=(100, 100),
                                                   flags=cv2.CASCADE_SCALE_IMAGE)
        if len(face_boxs):
            for x, y, w, h in face_boxs:
                e_right, e_left = find_eye_region(
                    pic[y:y + h, x:x + w])
                e_right_gray = cv2.split(e_right)[2]
                e_left_gray = cv2.split(e_left)[2]

                eye_radius = int(0.13 * (e_right.shape)[1])
                eye_radius = eye_radius if eye_radius % 2 == 1 else eye_radius + 1
                for i in range(e_left_gray.shape[0]):
                    for j in range(e_left_gray.shape[1]):
                        if e_left_gray[i][j] < 40:
                            e_left_gray[i][j] = 255 - e_left_gray[i][j]
                        else:
                            e_left_gray[i][j] = 0
                for i in range(e_right_gray.shape[0]):
                    for j in range(e_right_gray.shape[1]):
                        if e_right_gray[i][j] < 40:
                            e_right_gray[i][j] = 255 - e_right_gray[i][j]
                        else:
                            e_right_gray[i][j] = 0

                loc, mags = frst.frst(e_left_gray, eye_radius, 0.3)
                cv2.circle(e_left, loc, 1, (0, 255, 0))
                loc1, mags = frst.frst(e_right_gray, eye_radius, 0.3)
                cv2.circle(e_right, loc1, 1, (0, 255, 0))
                cv2.imshow('left eye', e_left)
                cv2.imshow('left1 eye', e_left_gray)
                cv2.imshow('right eye', e_right)
                cv2.imshow('right1 eye', e_right_gray)

        cv2.imshow('frame', pic)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            vs.release()
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()

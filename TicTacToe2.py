#!/usr/bin/env python 3.6
# w/ openCV3, version 3.4.2

import numpy as np
import cv2
from collections import deque
from checkWinner import check

def dist(a,b):
    return ((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5

def fingerCursor(device):
    cap = cv2.VideoCapture(device)
    cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # print(cap_height,cap_width)
    out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (1280, 720))

    ## gesture matching initialization
    gesture2 = cv2.imread('gesture2.png')
    gesture2 = cv2.cvtColor(gesture2, cv2.COLOR_BGR2GRAY)
    _, gesture2 , _ = cv2.findContours(gesture2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # skin color segmentation mask

    skin_min = np.array([0, 48, 80],np.uint8)  # HSV mask
    skin_max = np.array([20, 255, 255],np.uint8) # HSV mask

    ## trajectory drawing initialization
    des_1 = (540, 260)
    des_2 = (640, 260)
    des_3 = (740, 260)
    des_4 = (540, 360)
    des_5 = (640, 360)
    des_6 = (740, 360)
    des_7 = (540, 460)
    des_8 = (640, 460)
    des_9 = (740, 460)
    ori_1 = (540, 260)
    ori_2 = (640, 260)
    ori_3 = (740, 260)
    ori_4 = (540, 360)
    ori_5 = (640, 360)
    ori_6 = (740, 360)
    ori_7 = (540, 460)
    ori_8 = (640, 460)
    ori_9 = (740, 460)
    img_ori_position = deque([ori_1, ori_2, ori_3, ori_4, ori_5, ori_6, ori_7, ori_8, ori_9])
    img_ori_position_status = [-1, -1, -1, -1, -1, -1, -1, -1, -1]
    img_des_position = deque([des_1, des_2, des_3, des_4, des_5, des_6, des_7, des_8, des_9])
    img_des_position_status = [-1, -1, -1, -1, -1, -1, -1, -1, -1]
    topmost_last = ori_9    # initial position of finger cursor

    img = cv2.imread('IMG/logo.png')
    symbol = [1, 2, 1, 2, 1, 2, 1, 2, 1]
    currentsum = 0

    ## finger cursor position low_pass filter, keep robustness of moving object
    low_filter_size = 5
    # low_filter = deque([ori_9,ori_9,ori_9,ori_9,ori_9],low_filter_size )  # filter size is 5

    ## gesture_index low_pass filter
    gesture_filter_size = 5
    gesture_matching_filter = deque([0.0,0.0,0.0,0.0,0.0], gesture_filter_size )
    gesture_index_thres = 2.0

    ## color definition
    orange = (0,97,255)
    blue = (255,0,0)
    green = (0,255,0)
    red = (0,0,255)
    flag_1 = -1 # control if is gesture 2

    ## background segmentation


    # some kernels
    kernel_size = 5
    kernel1 = np.ones((kernel_size,kernel_size),np.float32)/kernel_size/kernel_size
    # kernel2 = np.ones((10,10), np.uint8)/100
    n = 0
    while(cap.isOpened()):
        flag_2 = -1  # control img_des_position_status
        if n < 10:
            n = n + 1
        # print(n)
        ## Capture frame-by-frame
        ret, frame_raw = cap.read()
        while not ret:
            ret,frame_raw = cap.read()
        frame_raw = cv2.flip(frame_raw,1)
        frame = frame_raw[:round(cap_height),:round(cap_width)]    # ROI of the image
        # cv2.imshow('raw_frame',frame)


        ## Color seperation and noise cancellation at HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, skin_min, skin_max)
        res = cv2.bitwise_and(hsv, hsv, mask= mask)
        res = cv2.erode(res, kernel1, iterations=1)
        res = cv2.dilate(res, kernel1, iterations=1)


        rgb = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
        # cv2.imshow('rgb_2',rgb)

        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray',gray)
        # gray = cv2.filter2D(gray,-1,kernel2)    # hacky
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        # cv2.imshow('gray',gray)


        ## Background segmentation using motion detection (Optional)

        ## main function: find finger cursor position & draw trajectory
        im2, contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)    # find all contours in the image
        print(len(contours))
        if len(contours) !=0:
            contours.sort(key = cv2.contourArea,reverse = True)
            c = contours[0]  # find biggest contour in the image


            if True:
                if cv2.contourArea(c) > 1000:
                    topmost = tuple(c[c[:, :, 1].argmin()][0])  # consider the topmost point of the biggest contour as cursor
                    gesture_index = cv2.matchShapes(c, gesture2[0], 2, 0.0)

                    # obtain gesture matching index using gesture matching low_pass filter
                    gesture_matching_filter.append(gesture_index)
                    sum_gesture = 0
                    for i in gesture_matching_filter:
                        sum_gesture += i
                    gesture_index = sum_gesture / gesture_filter_size

                    dis_bound = 50
                    flag = -1
                    # decide which img is chosen
                    for i in range(len(img_ori_position)):
                        dis = dist(topmost, img_ori_position[i])
                        if (img_ori_position_status[i] == -1) and (dis < dis_bound):
                            print("nearest: ", i)
                            flag = i
                            patch_current = i
                            flag_1 = 0
                    if flag != -1:
                        topmost_last = img_ori_position[patch_current]
                        low_filter = deque([topmost_last, topmost_last, topmost_last, topmost_last, topmost_last], low_filter_size)
                        img_ori_position_status[patch_current] = patch_current


                    dist_pts = dist(topmost,topmost_last)  # calculate the distance of last cursor position and current cursor position

                    if dist_pts < 150:  # filter big position change of cursor
                        try:

                            low_filter.append(topmost)
                            sum_x = 0
                            sum_y = 0
                            for i in low_filter:
                                sum_x += i[0]
                                sum_y += i[1]
                            topmost = (sum_x//low_filter_size, sum_y//low_filter_size)

                            if gesture_index < gesture_index_thres:
                                print(gesture_index)
                                for i in range(len(img_des_position)):
                                    dis = dist(topmost,img_des_position[i])
                                    if (dis < dis_bound) and (img_des_position_status[i] == -1):
                                        img_des_position_status[i] = symbol[currentsum]
                                        currentsum = currentsum + 1
                                        flag_1 = 1

                            else:
                                pass
                            topmost_last = topmost  # update cursor position
                            # print(topmost_last)
                        except:
                            print('error')
                            pass


        # draw img_patches already in grid
        for i in range(len(img_des_position)):
            if img_des_position_status[i] != -1:
                img_temp = cv2.imread('IMG2/' + '00' + str(img_des_position_status[i]) + '.jpg')
                frame[int(img_des_position[i][1] - img_temp.shape[0]/2):int(img_des_position[i][1] + img_temp.shape[0]/2), \
                int(img_des_position[i][0] - img_temp.shape[0]/2):int(img_des_position[i][0] + img_temp.shape[0]/2)] = img_temp
        # print(img_des_position_status)
        total = 0
        for item in img_des_position_status:
            if item != 1:
                total = total + 1
            # if total < 3:
            #     flag_2 = 0
            if item == -1 or total<3:
                flag_2 = 0

        # if flag_2 == -1:
        if True:
            print(img_des_position_status)
            result, winner = check(img_des_position_status)

            if result == 1:
                # print winner
                if winner == 1:
                    cv2.putText(img=frame, text='O win!', org=(int(cap_width / 2 - 100), int(cap_height / 2)),
                            fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3,
                            color=orange, thickness=5)
                else:
                    cv2.putText(img=frame, text='X win!', org=(int(cap_width / 2 - 100), int(cap_height / 2)),
                                fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3,
                                color=orange, thickness=5)
            elif result == 0:

                cv2.putText(img=frame, text='DRAW!', org=(int(cap_width / 2 - 150), int(cap_height / 2)),
                            fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3,
                            color=blue, thickness=5)


        if flag_1 == 1:
            cv2.circle(frame, topmost_last, 10, green, 3)

        if flag_2 == 0:
            cv2.line(frame, (490, 210), (790, 210), red, 1)
            cv2.line(frame, (490, 310), (790, 310), red, 1)
            cv2.line(frame, (490, 410), (790, 410), red, 1)
            cv2.line(frame, (490, 510), (790, 510), red, 1)
            cv2.line(frame, (490, 210), (490, 510), red, 1)
            cv2.line(frame, (590, 210), (590, 510), red, 1)
            cv2.line(frame, (690, 210), (690, 510), red, 1)
            cv2.line(frame, (790, 210), (790, 510), red, 1)



        ## Display the resulting frame


        # frame_raw = cv2.resize(frame_raw, (1024,768))
        # cv2.imshow('frame', frame_raw)
        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    ## When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    device = 0    # if device = 0, use the built-in computer camera
    fingerCursor(device)

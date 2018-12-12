#!/usr/bin/env python3

# w/ openCV3, version 3.3.0

import numpy as np
import cv2
from collections import deque
from mergeImage import mergeImage

def dist(a,b):
    return ((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5

def fingerCursor(device):
    cap = cv2.VideoCapture(device)

    # Height: 720, Width: 1280
    cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (1280, 720))

    ## gesture matching initialization
    gesture_stop = cv2.imread('gesture6.png')
    gesture2 = cv2.imread('gesture2.png')

    gesture_stop = cv2.cvtColor(gesture_stop, cv2.COLOR_BGR2GRAY)
    _, gesture_stop , _ = cv2.findContours(gesture_stop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    gesture2 = cv2.cvtColor(gesture2, cv2.COLOR_BGR2GRAY)
    _, gesture2 , _ = cv2.findContours(gesture2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # skin color segmentation mask
    # skin_min = np.array([0, 40, 50],np.uint8)  # HSV mask
    # skin_max = np.array([50, 250, 255],np.uint8) # HSV mask

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
    ori_1 = (200, 50)
    ori_2 = (310, 50)
    ori_3 = (420, 50)
    ori_4 = (530, 50)
    ori_5 = (640, 50)
    ori_6 = (750, 50)
    ori_7 = (860, 50)
    ori_8 = (970, 50)
    ori_9 = (1080, 50)
    img_ori_position = deque([ori_1, ori_2, ori_3, ori_4, ori_5, ori_6, ori_7, ori_8, ori_9])
    img_ori_position_status = [-1, -1, -1, -1, -1, -1, -1, -1, -1]
    img_des_position = deque([des_1, des_2, des_3, des_4, des_5, des_6, des_7, des_8, des_9])
    img_des_position_status = [-1, -1, -1, -1, -1, -1, -1, -1, -1]
    topmost_last = ori_9    # initial position of finger cursor
    topmost_last_second = ori_9

    img = cv2.imread('IMG/logo.png')
    img_1 = cv2.imread('IMG/001.jpg')
    img_2 = cv2.imread('IMG/002.jpg')
    img_3 = cv2.imread('IMG/003.jpg')
    img_4 = cv2.imread('IMG/004.jpg')
    img_5 = cv2.imread('IMG/005.jpg')
    img_6 = cv2.imread('IMG/006.jpg')
    img_7 = cv2.imread('IMG/007.jpg')
    img_8 = cv2.imread('IMG/008.jpg')
    img_9 = cv2.imread('IMG/009.jpg')


    ## finger cursor position low_pass filter
    low_filter_size = 5


    ## gesture_index low_pass filter
    gesture_filter_size = 5
    gesture_matching_filter = deque([0.0,0.0,0.0,0.0,0.0], gesture_filter_size )
    gesture_matching_filter_second = deque([0.0, 0.0, 0.0, 0.0, 0.0], gesture_filter_size)
    gesture_matching_stop_filter = deque([0.0, 0.0, 0.0, 0.0, 0.0], gesture_filter_size)
    gesture_index_thres = 2.0
    gesture_index_thres_stop = 1.5

    ## color definition
    orange = (0,97,255)
    blue = (255,0,0)
    green = (0,255,0)
    red = (0,0,255)
    patch_size = 100
    flag_1 = -1 # control if is gesture 2

    flag_1_second = -1  # control if is gesture 2



    # some kernels
    kernel_size = 5
    kernel1 = np.ones((kernel_size,kernel_size),np.float32)/kernel_size/kernel_size

    n = 0
    while(cap.isOpened()):
        flag_2 = -1  # control img_des_position_status
        if n < 10:
            n = n + 1

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


        ## Canny edge detection at Gray space.
        rgb = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
        # cv2.imshow('rgb_2',rgb)

        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray',gray)

        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        # cv2.imshow('gray',gray)


        ## main function: find finger cursor position & draw trajectory
        im2, contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)    # find all contours in the image
        print(len(contours))
        if len(contours) !=0:
            contours.sort(key = cv2.contourArea,reverse = True)
            c = contours[0]  # find biggest contour in the image
            if len(contours) > 1:
                c_second = contours[1] # find second biggest contour in image

            # c = max(contours_1, key = cv2.contourArea)
            if len(contours) > 1:
                if (cv2.contourArea(c) > 1000) and (cv2.contourArea(c_second) > 1000):
                    topmost = tuple(c[c[:, :, 1].argmin()][0])
                    topmost_second = tuple(c_second[c_second[:, :, 1].argmin()][0])  # consider the topmost point of the biggest contour as cursor

                    # obtain gesture matching index using gesture matching low_pass filter
                    # if it is not the first frame when 2 hands show at the same time
                    # let c always be the first hand shown in video or the first biggest contour's hand if two hands shown at the same time in the beginning frame
                    if (n > 1) and (dist(topmost_second, topmost_last) < dist(topmost, topmost_last)):
                        temp = c
                        c = c_second
                        c_second = temp
                        topmost = tuple(c[c[:, :, 1].argmin()][0])
                        topmost_second = tuple(c_second[c_second[:, :, 1].argmin()][0])


                    gesture_index = cv2.matchShapes(c, gesture2[0], 2, 0.0)
                    gesture_index_second = cv2.matchShapes(c_second, gesture2[0], 2, 0.0)
                    gesture_matching_filter.append(gesture_index)
                    gesture_matching_filter_second.append(gesture_index_second)

                    sum_gesture = 0
                    for i in gesture_matching_filter:
                        sum_gesture += i
                    gesture_index = sum_gesture / gesture_filter_size

                    sum_gesture = 0
                    for i in gesture_matching_filter_second:
                        sum_gesture += i
                    gesture_index_second = sum_gesture / gesture_filter_size

                    dis_bound = 50
                    flag = -1
                    flag_second = -1
                    # decide which img is chosen
                    for i in range(len(img_ori_position)):
                        dis = dist(topmost, img_ori_position[i])
                        if (img_ori_position_status[i] == -1) and (dis < dis_bound):
                            flag = i
                            patch_current = i
                            flag_1 = 0
                    if flag != -1:
                        topmost_last = img_ori_position[patch_current]
                        low_filter = deque([topmost_last, topmost_last, topmost_last, topmost_last, topmost_last],
                                           low_filter_size)
                        img_ori_position_status[patch_current] = patch_current

                    for i in range(len(img_ori_position)):
                        dis = dist(topmost_second, img_ori_position[i])
                        if (img_ori_position_status[i] == -1) and (dis < dis_bound):
                            flag_second = i
                            patch_current_second = i
                            flag_1_second = 0
                    if flag_second != -1:
                        topmost_last_second = img_ori_position[patch_current_second]
                        low_filter_second = deque(
                            [topmost_last_second, topmost_last_second, topmost_last_second, topmost_last_second,
                             topmost_last_second], low_filter_size)
                        img_ori_position_status[patch_current_second] = patch_current_second


                    dist_pts = dist(topmost,topmost_last)  # calculate the distance of last cursor position and current cursor position
                    # print("dist_1")
                    # print(topmost)
                    # print(topmost_last)
                    # print(dist_pts)

                    dist_pts_second = dist(topmost_second, topmost_last_second)
                    # print("dist_2")
                    # print(topmost_second)
                    # print(topmost_last_second)
                    # print(dist_pts_second)

                    if dist_pts < 150:  # filter big position change of cursor
                        try:
                            # cv2.drawContours(rgb, [c], 0 , (0, 255, 0),5)
                            low_filter.append(topmost)
                            sum_x = 0
                            sum_y = 0
                            for i in low_filter:
                                sum_x += i[0]
                                sum_y += i[1]
                            topmost = (sum_x // low_filter_size, sum_y // low_filter_size)

                            if gesture_index < gesture_index_thres:
                                for i in range(len(img_des_position)):
                                    dis = dist(topmost, img_des_position[i])
                                    if (dis < dis_bound) and (img_des_position_status[i] == -1):
                                        # cv2.circle(frame, img_des_position[i], 10, blue, 3)
                                        img_des_position_status[i] = patch_current
                                        flag_1 = 1

                            else:
                                pass
                            topmost_last = topmost  # update cursor position
                            print(topmost_last)
                        except:
                            print('error')
                            pass

                    if dist_pts_second < 150:  # filter big position change of cursor
                        try:
                            # cv2.drawContours(rgb, [c], 0 , (0, 255, 0),5)
                            low_filter_second.append(topmost_second)
                            sum_x = 0
                            sum_y = 0
                            for i in low_filter_second:
                                sum_x += i[0]
                                sum_y += i[1]
                            topmost_second = (sum_x//low_filter_size, sum_y//low_filter_size)

                            if gesture_index_second < gesture_index_thres:
                                for i in range(len(img_des_position)):
                                    dis = dist(topmost_second,img_des_position[i])
                                    if (dis < dis_bound) and (img_des_position_status[i] == -1):
                                        # cv2.circle(frame, img_des_position[i], 10, blue, 3)
                                        img_des_position_status[i] = patch_current_second
                                        flag_1_second = 1

                            else:
                                pass
                            topmost_last_second = topmost_second  # update cursor position
                            print(topmost_last_second)
                        except:
                            print('error')
                            pass


            elif True:
                if cv2.contourArea(c) > 1000:
                    topmost = tuple(c[c[:, :, 1].argmin()][0])  # consider the topmost point of the biggest contour as cursor
                    gesture_index = cv2.matchShapes(c, gesture2[0], 2, 0.0)
                    gesture_index_stop = cv2.matchShapes(c, gesture_stop[0], 2, 0.0)
                    print(gesture_index)
                    print(gesture_index_stop)
                    # obtain gesture matching index using gesture matching low_pass filter
                    gesture_matching_stop_filter.append(gesture_index_stop)
                    gesture_matching_filter.append(gesture_index)
                    sum_gesture = 0
                    for i in gesture_matching_filter:
                        sum_gesture += i
                    gesture_index = sum_gesture / gesture_filter_size

                    sum_gesture_stop = 0
                    for i in gesture_matching_stop_filter:
                        sum_gesture_stop += i
                    gesture_index_stop = sum_gesture_stop / gesture_filter_size
                    print(gesture_index_stop)

                    print("one_hand")
                    if gesture_index_stop < gesture_index_thres_stop:
                        print("restart")
                        img_ori_position_status = [-1, -1, -1, -1, -1, -1, -1, -1, -1]
                        img_des_position_status = [-1, -1, -1, -1, -1, -1, -1, -1, -1]


                    dis_bound = 50
                    flag = -1
                    # decide which img is chosen
                    for i in range(len(img_ori_position)):
                        dis = dist(topmost, img_ori_position[i])
                        if (img_ori_position_status[i] == -1) and (dis < dis_bound):
                            flag = i
                            patch_current = i
                            flag_1 = 0
                    if flag != -1:
                        topmost_last = img_ori_position[patch_current]
                        low_filter = deque([topmost_last, topmost_last, topmost_last, topmost_last, topmost_last], low_filter_size)
                        img_ori_position_status[patch_current] = patch_current


                    dist_pts = dist(topmost,topmost_last)  # calculate the distance of last cursor position and current cursor position
                    # print("dist_1")
                    # print(topmost)
                    # print(topmost_last)
                    # print(dist_pts)

                    if dist_pts < 150:  # filter big position change of cursor
                        try:
                            # cv2.drawContours(rgb, [c], 0 , (0, 255, 0),5)
                            low_filter.append(topmost)
                            sum_x = 0
                            sum_y = 0
                            for i in low_filter:
                                sum_x += i[0]
                                sum_y += i[1]
                            topmost = (sum_x//low_filter_size, sum_y//low_filter_size)

                            if gesture_index < gesture_index_thres:
                                for i in range(len(img_des_position)):
                                    dis = dist(topmost,img_des_position[i])
                                    if (dis < dis_bound) and (img_des_position_status[i] == -1):
                                        # cv2.circle(frame, img_des_position[i], 10, blue, 3)
                                        img_des_position_status[i] = patch_current
                                        flag_1 = 1

                            else:
                                pass
                            topmost_last = topmost  # update cursor position
                            # print(topmost_last)
                        except:
                            print('error')
                            pass


        # img_size : 100*100
        # draw current img_patches on top, which hasn't been choosen
        for i in range(len(img_ori_position)):
            if img_ori_position_status[i] == -1:
                img_temp = cv2.imread('IMG/' + '00' +str(i+1)+ '.jpg')
                frame[int(img_ori_position[i][1] - img_temp.shape[0]/2):int(img_ori_position[i][1] + img_temp.shape[0]/2), \
                int(img_ori_position[i][0] - img_temp.shape[1]/2):int(img_ori_position[i][0] + img_temp.shape[1]/2)] = img_temp

        # draw img_patches already in grid
        for i in range(len(img_des_position)):
            if img_des_position_status[i] != -1:
                img_temp = cv2.imread('IMG/' + '00' + str(img_des_position_status[i]+1) + '.jpg')
                frame[int(img_des_position[i][1] - img_temp.shape[0]/2):int(img_des_position[i][1] + img_temp.shape[0]/2), \
                int(img_des_position[i][0] - img_temp.shape[0]/2):int(img_des_position[i][0] + img_temp.shape[0]/2)] = img_temp
        # print(img_des_position_status)

        for item in img_des_position_status:
            if item == -1:
                flag_2 = 0
        if flag_2 == -1:
            img_des_position_status_1 = [x + 1 for x in img_des_position_status]
            result = mergeImage(img_des_position_status_1)
            # print("result")
            # print(result)
            if result == 0:
                frame[210:510, 490:790] = img
                # read original image, print "win"
                cv2.putText(img=frame, text='You win!', org=(int(cap_width / 2 - 200), int(cap_height / 2)),
                            fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3,
                            color=orange, thickness=5)
            else:
                # read black and white wrong image, print "lose"
                img_lose = cv2.imread('IMG/output.png')
                frame[210:510, 490:790] = img_lose
                cv2.putText(img=frame, text='You lose!', org=(int(cap_width / 2 - 250), int(cap_height / 2)),
                            fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3,
                            color=blue, thickness=5)


        if flag_1 == 1:
            cv2.circle(frame, topmost_last, 10, green, 3)
        elif flag_1 == 0:
            img_patch_current = cv2.imread('IMG/' + '00' + str(patch_current + 1) + '.jpg')
            if int(topmost_last[1] - img_patch_current.shape[0] / 2) >= 0 and int(topmost_last[0] - img_patch_current.shape[1] / 2) >= 0:
                frame[int(topmost_last[1] - img_patch_current.shape[0]/2):int(topmost_last[1] + img_patch_current.shape[0]/2), \
                int(topmost_last[0] - img_patch_current.shape[1]/2):int(topmost_last[0] + img_patch_current.shape[1]/2)] = img_patch_current

        if len(contours) > 1:
            if flag_1_second == 1:
                cv2.circle(frame, topmost_last_second, 10, green, 3)
            elif flag_1_second == 0:
                img_patch_current_second = cv2.imread('IMG/' + '00' + str(patch_current_second + 1) + '.jpg')
                if int(topmost_last_second[1] - img_patch_current_second.shape[0] / 2) >= 0 and int(topmost_last_second[0] - img_patch_current_second.shape[1] / 2) >= 0:
                    frame[int(topmost_last_second[1] - img_patch_current_second.shape[0]/2):int(topmost_last_second[1] + img_patch_current_second.shape[0]/2), \
                    int(topmost_last_second[0] - img_patch_current_second.shape[1]/2):int(topmost_last_second[0] + img_patch_current_second.shape[1]/2)] = img_patch_current_second
        if flag_2 == 0: #if the grid are not full
            cv2.line(frame, (490, 210), (790, 210), red, 1)
            cv2.line(frame, (490, 310), (790, 310), red, 1)
            cv2.line(frame, (490, 410), (790, 410), red, 1)
            cv2.line(frame, (490, 510), (790, 510), red, 1)
            cv2.line(frame, (490, 210), (490, 510), red, 1)
            cv2.line(frame, (590, 210), (590, 510), red, 1)
            cv2.line(frame, (690, 210), (690, 510), red, 1)
            cv2.line(frame, (790, 210), (790, 510), red, 1)

        frame[720 - img.shape[0]:720, 1280 - img.shape[1]:1280] = img
        # cv2.circle(rgb, topmost_last, 10, blue , 3)

        ## Display the resulting frame

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

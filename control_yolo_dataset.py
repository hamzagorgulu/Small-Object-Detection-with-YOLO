# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 22:59:10 2020

Control the images in a shout folder and their ground-truth YOLO box

@author: Onur Caki
"""
#!python3

import os
import cv2
import argparse
import re
import numpy as np
image_ext = ['.bmp', '.png', '.jpg']


def color_generator():
    label_colors = []
    for i in range(80):
        c = (int(np.random.randint(60, 255, 3)[0]),
             int(np.random.randint(60, 255, 3)[1]),
             int(np.random.randint(60, 255, 3)[2]))
        label_colors.append(c)
    return label_colors


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_path",
                    help="path of the folder in which shots are located")
    args = ap.parse_args()
    return args


def natural_sort(l):
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in image_ext:
                image_names.append(apath)
    return natural_sort(image_names)


def read_gt(label, experiment_image):
    with open(label) as f:
        labels = []
        Lines = f.readlines()
        for line in Lines:
            label = int(line.strip().split(' ')[0])
            # x_width
            xc = float(line.strip().split(' ')[1])
            # x_height
            yc = float(line.strip().split(' ')[2])
            w = float(line.strip().split(' ')[3])
            h = float(line.strip().split(' ')[4])
            # x_min = max(xc-w/2, 0)
            # y_min = max(yc-h/2, 0)
            # x_max = min(xc+w/2, 1)
            # y_max = min(yc+h/2, 1)

            xc_Real = xc * experiment_image.shape[1]
            yc_Real = yc * experiment_image.shape[0]
            w_Real = w * experiment_image.shape[1]
            h_Real = h * experiment_image.shape[0]

            # x_min = int(round(x_min * experiment_image.shape[1]))
            # x_max = int(round(x_max * experiment_image.shape[1]))

            # y_min = int(round(y_min * experiment_image.shape[0]))
            # y_max = int(round(y_max * experiment_image.shape[0]))

            # x_min = int(xc_Real-w_Real/2)
            # x_max = int(xc_Real+w_Real/2)
            # y_min = int(yc_Real-h_Real/2)
            # y_max = int(yc_Real+h_Real/2)

            x_min = int(round(xc_Real-w_Real/2))
            x_max = int(round(xc_Real+w_Real/2))
            y_min = int(round(yc_Real-h_Real/2))
            y_max = int(round(yc_Real+h_Real/2))

            labels.append([label, x_min, y_min, x_max, y_max])
    f.close()
    return labels


def read_class_names(path):
    path = os.path.join(path, 'classes.txt')
    classes = []
    if os.path.exists(path):
        with open(path) as f:
            lines = f.readlines()
            for line in lines:
                classes.append(line.strip())
    return classes


args = parse_args()
image_path = args.data_path
#image_path = "/Users/hamzagorgulu/Desktop/course_contents/COMP541_Deep_Learning/Project/images_crop"
image_list = get_image_list(image_path)
label_colors = color_generator()
classes = read_class_names(image_path)

i = 0
delete_list = []
new_anatotations = []
while 1:
    image = image_list[i]
    img = cv2.imread(image)

    # label_file_pth = image.replace("images", "labels")
    # label_path = label_file_pth.replace("png", "txt")
    label_path = image[:image.rfind('.')]+'.txt'

    if not os.path.exists(label_path):
        img = cv2.putText(img, 'There is no label for this image', (15, 15),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        label = read_gt(label_path, img)
        for l in label:
            # represents the top left corner of rectangle
            start_point = (l[1], l[2])
            # represents the bottom right corner of rectangle
            end_point = (l[3], l[4])
            # green color in BGR
            color = label_colors[l[0]]
            # Line thickness of -1 px
            thickness = 1
            radius = 1
            xc = int((start_point[0] + end_point[0])/2)
            yc = int((start_point[1] + end_point[1])/2)

            center_coordinates = (xc, yc)
            img = cv2.circle(img, center_coordinates, radius, color, thickness)

            img = cv2.rectangle(img, start_point, end_point, color, thickness)
            if len(classes) > 0:
                img = cv2.putText(img, classes[l[0]], (l[1]-5, l[2]-5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)
    if img.shape[0] >= 850:
        if img.shape[0] > img.shape[1]:
            img = cv2.resize(img, (540, 720))                # Resize image
        else:
            img = cv2.resize(img, (720, 540))
    cv2.imshow('window', img)
    ch = cv2.waitKey(0)

    if chr(ch) == "d":
        i = i + 1
        if i > len(image_list) - 1:
            print("finish")
            i = 0
    if chr(ch) == "a":
        i = i - 1
        if i < 0:
            i = len(image_list) - 1
    if chr(ch) == "1":
        delete_list.append(image)
    if chr(ch) == "2":
        new_anatotations.append(image)
    if chr(ch) == "p":
        print(image)
    if chr(ch) == "r":
        i = np.random.randint(low=0, high=len(image_list), size=1)[0]
    if chr(ch) == "q":
        cv2.destroyAllWindows()
        break

if len(delete_list) != 0:
    delete_txt = open('delete.txt', 'w')
    for ele in delete_list:
        delete_txt.write(ele + '\n')
    delete_txt.close()


if len(new_anatotations) != 0:
    label_txt = open('label.txt', 'w')
    for ele in new_anatotations:
        label_txt.write(ele + '\n')
    label_txt.close()

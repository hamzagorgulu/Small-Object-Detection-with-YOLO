#!python3

import os
import cv2
import numpy as np
import shutil
import re
from tqdm import tqdm
import time

image_ext = ['.png', '.jpg', '.bmp']
ONE_CHANNEL = False
CROP_SIZE = 480
NUMBER_OF_CROPS = 10
PATH = "/Users/hamzagorgulu/Desktop/course_contents/COMP541_Deep_Learning/Project/Visdrone/VisDrone/VisDrone2019-DET-val/images"

def natural_sort(images):
    """
    Sort the given iterable in the way that computer expect.
    :param images: iterable to be sorted.
    :return: sorted iterable.
    """
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)] # one or more consecutive digits
    return sorted(images, key=alphanum_key)


def get_files(path):
    """Get the list of files in a directory recursively and sort them in natural order.
    Args:
        path (string): The path to the directory.
    Returns:
        list: full directory path of the image files in list.
    """
    images = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]  # splitext is for getting the extension(root and ext) to check the extension whether it is acceptible or not
            if ext in image_ext:
                images.append(apath)
    return natural_sort(images)


def read_gt(label):
    """ Read the ground truth file and return the list of labels.

    Args:
        label (string): The path to the ground truth file.

    Returns:
        list: list of labels.
    """
    with open(label) as f:
        labels = []
        Lines = f.readlines()
        for line in Lines:
            label = int(line.strip().split(' ')[0])
            x = float(line.strip().split(' ')[1])  # x_width
            y = float(line.strip().split(' ')[2])  # x_height
            w_r = float(line.strip().split(' ')[3])
            h_r = float(line.strip().split(' ')[4])
            labels.append([label, x, y, w_r, h_r])
    f.close()
    return labels

# def random_crop(t, image, crop_size):
#     x= t[1]*image.shape[1]
#     y= t[2]*image.shape[0]
#     w= t[3]*image.shape[1]
#     h= t[4]*image.shape[0]

#     xmin = int(round(x-w/2))
#     ymin = int(round(y-h/2))

#     #randomly select top cropping points
#     new_ymin, new_xmin = np.random.randint(low = 20, high = crop_size-max(w/2,h/2)-10, size=2)

#     while(ymin-new_ymin<0):
#         new_ymin=new_ymin-1

#     while(ymin-new_ymin+crop_size>=image.shape[0]):
#         new_ymin=new_ymin+1

#     while(xmin-new_xmin<0):
#         new_xmin=new_xmin-1

#     while(xmin-new_xmin+crop_size>=image.shape[1]):
#         new_xmin=new_xmin+1

#     crop_img = image[ymin-new_ymin : ymin-new_ymin+crop_size, xmin-new_xmin : xmin-new_xmin+crop_size]

#     new_y = (new_ymin+h/2)/crop_img.shape[0]
#     new_x = (new_xmin+w/2)/crop_img.shape[1]
#     new_w = t[3]*image.shape[1]/crop_img.shape[1]
#     new_h = t[4]*image.shape[0]/crop_img.shape[0]
#     t_new= [0 , round(new_x,6) , round(new_y,6) , round(new_w,6) , round(new_h,6)]

#     return t_new, crop_img


# def random_crop(t, image, crop_size):
#     x = t[1]*image.shape[1]
#     y = t[2]*image.shape[0]

#     # randomly select new centers
#     new_y, new_x = np.random.randint(low=50, high=crop_size-50, size=2)

#     # keep new center of ball and 512 crop in image size
#     # Note: (y-new_y,x-new_x) new (0,0) origin

#     while(y-new_y < 0):
#         new_y = new_y-1

#     while(y-new_y+crop_size > image.shape[0]):
#         new_y = new_y+1

#     while(x-new_x < 0):
#         new_x = new_x-1

#     while(x-new_x+crop_size > image.shape[1]):
#         new_x = new_x+1

#     crop_img = image[y-new_y:y-new_y+crop_size, x-new_x:x-new_x+crop_size]

#     new_y = new_y/crop_img.shape[0]
#     new_x = new_x/crop_img.shape[1]
#     new_w = int(round(t[3]*image.shape[1]))/crop_img.shape[1]
#     new_h = int(round(t[4]*image.shape[0]))/crop_img.shape[0]
#     t_new = [0, new_x, new_y, new_w, new_h]

#     return t_new, crop_img

def random_crop(t, image, crop_size, num_crops = NUMBER_OF_CROPS): #bunlarda tek label var, bize uymuyor.
    """
    t: list of labels in the whole image
    """
    # determine how many labels are in the image
    num_labels = len(t)
    crop_img_list = []
    t_new_list = []

    if image.shape[0] < crop_size or image.shape[1] < crop_size:
        return [t], [image]


    for _ in range(num_crops):
        # randomly select new centers (left top corner of the crop)
        crop_x = np.random.randint(low=0, high = max(0,image.shape[1]-crop_size), size=1)[0]
        crop_y = np.random.randint(low=0, high= max(0,image.shape[0]-crop_size), size=1)[0]

        # find the labels in this crop
        labels_in_crop = []
        for label in t:
            x = label[1]*image.shape[1]
            y = label[2]*image.shape[0]
            w = label[3]*image.shape[1]
            h = label[4]*image.shape[0]

            xmin = int(round(x-w/2))
            ymin = int(round(y-h/2))
            xmax = int(round(x+w/2))
            ymax = int(round(y+h/2))

            if xmin >= crop_x and xmax <= crop_x+crop_size and ymin >= crop_y and ymax <= crop_y+crop_size:
                labels_in_crop.append(label)

        # if there is no label in the crop, select a new crop
        if len(labels_in_crop) == 0:
            continue

        # crop the image
        crop_img = image[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size, :]

        # update the labels
        t_new = []
        for label in labels_in_crop:
            obj = label[0]
            x = label[1]*image.shape[1]
            y = label[2]*image.shape[0]
            w = label[3]*image.shape[1]
            h = label[4]*image.shape[0]

            new_x = (x-crop_x)/crop_img.shape[1]
            new_y = (y-crop_y)/crop_img.shape[0]
            new_w = w / crop_img.shape[1]
            new_h = h / crop_img.shape[0]

            t_new.append([obj, new_x, new_y, new_w, new_h])

        crop_img_list.append(crop_img)
        t_new_list.append(t_new)

    return t_new_list, crop_img_list



def random_crop_column(t, image, crop_size):
    """
    t: list of labels in the whole image
    """
    x = t[1]*image.shape[1]
    w = t[3]*image.shape[1]

    x_min = int(round(x-w/2))
    x_max = int(round(x+w/2))

    # randomly select new centers
    crop_x = np.random.randint(low=max(
        0, x_max+1-crop_size), high=min(x_min-1, image.shape[1]-crop_size), size=1)[0]


    # keep new center of ball and 512 crop in image size
    # Note: (y-new_y,x-new_x) new (0,0) origin

    crop_img = image[:, crop_x:crop_x+crop_size]

    new_x = (x-crop_x)/crop_img.shape[1]
    # new_w = w/crop_img.shape[1]
    # new_h = h/crop_img.shape[0]

    # r = (w+h)/2
    new_w = w/crop_img.shape[1]

    t_new = [t[0], new_x, t[2], new_w, t[4]]

    return t_new, crop_img


def random_crop_column_n(t, image, crop_size):

    # randomly select new centers
    crop_x = np.random.randint(
        low=0, high=image.shape[1]-crop_size-1, size=1)[0]

    # keep new center of ball and 512 crop in image size
    # Note: (y-new_y,x-new_x) new (0,0) origin

    crop_img = image[:, crop_x:crop_x+crop_size]

    t_new = t

    return t_new, crop_img

"""
def random_crop(t, image, crop_size):
    x = t[1]*image.shape[1]
    y = t[2]*image.shape[0]
    w = t[3]*image.shape[1]
    h = t[4]*image.shape[0]

    x_min = int(round(x-w/2))
    y_min = int(round(y-h/2))
    x_max = int(round(x+w/2))
    y_max = int(round(y+h/2))

    # randomly select new centers, play with this
    crop_x = np.random.randint(low=max(
        0, x_max+5-crop_size), high=min(x_min-1, image.shape[1]-crop_size), size=1)[0]
    crop_y = np.random.randint(low=max(
        0, y_max+5-crop_size), high=min(y_min-1, image.shape[0]-crop_size), size=1)[0]

    # keep new center of ball and 512 crop in image size
    # Note: (y-new_y,x-new_x) new (0,0) origin

    crop_img = image[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]  # keep it same

    # do this label update in for loop because there will be many labels
    new_y = (y-crop_y)/crop_img.shape[0]
    new_x = (x-crop_x)/crop_img.shape[1]
    # new_w = w/crop_img.shape[1]
    # new_h = h/crop_img.shape[0]

    # r = (w+h)/2, normalize them
    new_w = w/crop_img.shape[1]
    new_h = h/crop_img.shape[0]

    t_new = [0, new_x, new_y, new_w, new_h]

    return t_new, crop_img
"""
def imread_8bit(img_path):
    img_anydept = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
    img8_8bit = (img_anydept/256).astype('uint8')
    return img8_8bit


def imread_normalized(img_path):
    img_anydept = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
    img8_8bit_normalized = cv2.normalize(
        img_anydept, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return img8_8bit_normalized


def main():
    start = time.time()
    data_path = PATH
    save_dir = data_path.split("/")[-1] + "_crop"
    i = 0

    # Crop Process
    # create output folder for processed images

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)



    image_files = get_files(data_path)
    for img_path in tqdm(image_files):

        label_path = img_path[:img_path.rfind('.')] + '.txt'  # rfind: index of the last occurance of the specified value, img_path'ın txt'lisi
        if not os.path.exists(label_path):
            print(img_path)
            continue
        current_shot = img_path.split('/')[-2]


        current_image = cv2.imread(img_path)

        labels = read_gt(label_path)  # list of labels(list of lists)
        

        # define crop img
        t_new_list, crop_img_list = random_crop(labels, current_image, crop_size=CROP_SIZE)


        # save new ground-truth otherfolder
        i = 0
        for t_new in t_new_list: # t_new represents multiple labels in a single cropped image
            tag = "_crop_" + str(i)
            new_txt = open(os.path.join(save_dir, label_path.split('/')[-1].split(".")[0] + tag + ".txt"), 'w')
            i += 1
            for ele in t_new:
                for value in ele:
                    new_txt.write(str(value)+' ')
                new_txt.write('\n')
            new_txt.close()
        i = 0
        for crop_img in crop_img_list:
            tag = "_crop_" + str(i)
            i += 1
            cv2.imwrite(os.path.join(save_dir, img_path.split('/')[-1].split(".")[0] + tag + ".jpg"), crop_img)

    end = time.time()
    print("Total time: ", end-start)


if __name__ == '__main__':
    main()
# img_def = cv2.imread(img_path,0)
# img_anydept = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
# img_8bit_norm_minmax = imread_normalized(img_path)
# img8_8bit_max = imread_8bit(img_path)

# import matplotlib.pyplot as plt

# plt.imshow(img_anydept,'gray')
# plt.imshow(img_def,'gray')
# plt.imshow(img_8bit_norm_minmax,'gray')
# plt.imshow(img8_8bit_max,'gray')

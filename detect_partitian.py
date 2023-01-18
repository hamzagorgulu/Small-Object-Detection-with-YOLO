# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

from utils.torch_utils import select_device, smart_inference_mode
from utils.plots import Annotator, colors, save_one_box
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from models.common import DetectMultiBackend
import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

np.random.seed(0)

class_name_map = {
    0: "pedestrian",
    1: 'people',
    2: 'bicycle',
    3: "car",
    4: 'van',
    5: "truck",
    6: "tricycle",
    7: "awning - tricycle",
    8: "bus",
    9: "motor"}


def color_generator():
    label_colors = []
    for i in range(80):
        c = (int(np.random.randint(60, 255, 3)[0]),
             int(np.random.randint(60, 255, 3)[1]),
             int(np.random.randint(60, 255, 3)[2]))
        label_colors.append(c)
    return label_colors


def infer(im, model, dt):
    detections = []
    height, width = im.shape[:2]
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)
    with dt[0]:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

    # Inference
    with dt[1]:
        pred = model(im, augment=False, visualize=False)

    # NMS
    with dt[2]:
        # conf_thres, iou_thres, classes, agnostic_nms, max_det
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

    for det in pred:
        if len(det):
            boxes = det[:, :4]
            boxes[:, 0].clamp_(0, width)  # x1
            boxes[:, 1].clamp_(0, height)  # y1
            boxes[:, 2].clamp_(0, width)  # x2
            boxes[:, 3].clamp_(0, height)  # y2
            det[:, :4] = boxes
            for *xyxy, conf, cls in reversed(det):
                x_min = xyxy[0].item()
                y_min = xyxy[1].item()
                x_max = xyxy[2].item()
                y_max = xyxy[3].item()
                w = x_max - x_min
                h = y_max - y_min
                detections.append([cls.item(), conf.item(), [x_min, y_min, w, h]])

    return detections


def get_detections_from_partitions(model, img, partition_size, dt, ROI_flag=False, ROI_SIZE=0.8, CONFIDENCE_TH=0.25, NMS_TH=0.45, SELECT_HC_FLAG=False, NUMBER_OF_HC=5):
    detections = []
    h, w = img.shape[:2]

    if ROI_flag:
        roi = int(min(img.shape[:2]) * ROI_SIZE)
        y_max_roi = int(h - 0.1 * w)
        y_min_roi = int(y_max_roi - roi)
        x_min_roi = int(w / 2) - int(roi / 2)
        x_max_roi = int(w / 2) + int(roi / 2)
        # Control panel
        # cv2.rectangle(img, roi_rect[0], roi_rect[1], (255,0,0) ,2)
        # im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # plt.imshow(im_rgb)
        # plt.show()

        # roi_rect = [(x_min_roi, y_min_roi), (x_max_roi, y_max_roi)]
        cropped_img = img[y_min_roi:y_max_roi, x_min_roi:x_max_roi]
    else:
        y_max_roi = h
        y_min_roi = 0
        x_min_roi = 0
        x_max_roi = w
        cropped_img = img[y_min_roi:y_max_roi, x_min_roi:x_max_roi]

    img_h, img_w = cropped_img.shape[:2]
    num_of_partition_w = int(img_w / partition_size) + 1
    num_of_partition_h = int(img_h / partition_size) + 1
    partition_shift_w = (partition_size * num_of_partition_w - img_w) / (num_of_partition_w - 1)
    partition_shift_h = (partition_size * num_of_partition_h - img_h) / (num_of_partition_h - 1)
    x_pad = x_min_roi
    y_pad = y_min_roi
    img_boxes = []
    img_confs = []
    img_labels = []
    partition_rectangles = []
    for c in range(num_of_partition_w):
        for r in range(num_of_partition_h):
            bxmin = int(c * (partition_size - partition_shift_w))
            bxmax = bxmin + partition_size
            bymin = int(r * (partition_size - partition_shift_h))
            bymax = bymin + partition_size
            current_input_img = cropped_img[bymin:bymax, bxmin:bxmax]
            partition_rectangles.append([(bxmin + x_min_roi, bymin + y_min_roi),
                                        (bxmax + x_min_roi, bymax + y_min_roi)])
            detections = infer(current_input_img, model, dt)
            x_pad_in = x_pad + bxmin
            y_pad_in = y_pad + bymin
            for label, confidence, box in detections:
                box = list(box)
                box = list(map(int, np.rint(np.array(box))))
                box[0] += int(x_pad_in)
                box[1] += int(y_pad_in)
                img_boxes.append(box)
                img_confs.append(confidence)
                img_labels.append(int(label))
    indexes = cv2.dnn.NMSBoxes(img_boxes, img_confs, CONFIDENCE_TH, NMS_TH)
    # print(indexes)
    # indexes = [x[0] for x in indexes]
    img_boxes = np.array(img_boxes)[indexes]
    img_confs = np.array(img_confs)[indexes]
    img_labels = np.array(img_labels)[indexes]

    if SELECT_HC_FLAG:
        if len(img_boxes) > NUMBER_OF_HC:
            img_boxes = img_boxes[:NUMBER_OF_HC - 1]
            img_confs = img_confs[:NUMBER_OF_HC - 1]
            img_labels = img_labels[:NUMBER_OF_HC - 1]
    # merge three arrays into one

    return img_boxes, img_confs, img_labels, partition_rectangles


class TestSmallObject:
    def __init__(self, iou_thres, save_dir):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.result_dict = {}
        self.iou_thres = iou_thres
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.fn_area_list = []
        self.tp_area_list = []
        self.fp_area_list = []
        self.iou_list = []
        self.non_labeled_frames = []
        self.gt_labels = None

    def read_gt(self, label, img_height, img_width):
        self.gt_labels = []
        with open(label) as f:
            Lines = f.readlines()
            for line in Lines:
                label = int(line.strip().split(' ')[0])
                x = float(line.strip().split(' ')[1]) * img_width
                y = float(line.strip().split(' ')[2]) * img_height
                w_r = float(line.strip().split(' ')[3]) * img_width
                h_r = float(line.strip().split(' ')[4]) * img_height
                x_min = x - w_r / 2
                y_min = y - h_r / 2
                x_max = x + w_r / 2
                y_max = y + h_r / 2
                detect_flag = 0
                self.gt_labels.append([label, x_min, y_min, x_max, y_max, detect_flag])

        f.close()

    def bb_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou

    def compare_detections(self, pred):
        cls = pred[0]
        x_min = pred[1]
        y_min = pred[2]
        x_max = pred[3]
        y_max = pred[4]
        detection_flag = False
        if cls not in self.result_dict:
            self.result_dict[cls] = {'tp': 0, 'fp': 0, 'fn': 0}
        for i in range(0, len(self.gt_labels)):
            current_label = self.gt_labels[i]
            gt_cls = current_label[0]
            x_min_gt = current_label[1]
            y_min_gt = current_label[2]
            x_max_gt = current_label[3]
            y_max_gt = current_label[4]

            iou = self.bb_intersection_over_union([x_min, y_min, x_max, y_max], [
                                                  x_min_gt, y_min_gt, x_max_gt, y_max_gt])
            if iou > self.iou_thres:
                if cls == gt_cls:
                    self.tp = self.tp + 1
                    self.tp_area_list.append((x_max_gt - x_min_gt) + (y_max_gt - y_min_gt) / 2)
                    detection_flag = True
                    self.gt_labels[i][-1] = 1
                    self.result_dict[cls]['tp'] += 1
                    self.iou_list.append(iou)
        if not detection_flag:
            self.fp = self.fp + 1
            self.fp_area_list.append((x_max - x_min) + (y_max - y_min) / 2)
            self.result_dict[cls]['fp'] += 1

    def false_negative_counter(self):
        for current_label in self.gt_labels:
            if current_label[5] == 0:
                cls = current_label[0]
                x_min_gt = current_label[1]
                y_min_gt = current_label[2]
                x_max_gt = current_label[3]
                y_max_gt = current_label[4]
                self.fn = self.fn + 1
                if cls not in self.result_dict:
                    self.result_dict[cls] = {'tp': 0, 'fp': 0, 'fn': 0}
                self.result_dict[cls]['fn'] += 1
                self.fn_area_list.append((x_max_gt - x_min_gt) + (y_max_gt - y_min_gt) / 2)

    def plot_width(self):

        _, bin_ratio_fn = np.histogram(self.fn_area_list)
        mean_area_fn = sum(
            self.fn_area_list) / len(self.fn_area_list)

        _, bin_ratio_fp = np.histogram(self.fp_area_list)
        mean_area_fp = sum(self.fp_area_list) / len(self.fp_area_list)

        _, bin_ratio_tp = np.histogram(self.tp_area_list)
        mean_area_tp = sum(
            self.tp_area_list) / len(self.tp_area_list)

        fig, axs = plt.subplots(2, 1)
        fig.set_figheight(16)
        fig.set_figwidth(20)

        min_point = min(min(
            self.fn_area_list, self.tp_area_list, self.fp_area_list)) - 5
        max_point = max(max(
            self.fn_area_list, self.tp_area_list, self.fp_area_list)) + 5

        sns.distplot(self.fn_area_list, axlabel=False,
                     rug=True, bins=bin_ratio_fn, ax=axs[0])
        axs[0].axvline(mean_area_fn, color='r', linestyle='--',
                       label='Mean = ' + str(round(mean_area_fn, 2)))
        axs[0].title.set_text('(W+H)/2 Distribution of False Negatives; mean: %0.2f' % (
            mean_area_fn))
        axs[0].set_xlim([min_point, max_point])
        axs[0].legend()
        axs[0].grid(True)

        sns.distplot(self.tp_area_list, axlabel=False,
                     rug=True, bins=bin_ratio_tp, ax=axs[1])
        axs[1].axvline(mean_area_tp, color='r', linestyle='--',
                       label='Mean = ' + str(round(mean_area_tp, 2)))
        axs[1].title.set_text('(W+H)/2 Distribution of True Positives; mean: %0.2f' % (
            mean_area_tp))
        axs[1].set_xlim([min_point, max_point])
        axs[1].legend()
        axs[1].grid(True)

        fig.savefig('{}/Radius_Distributions.png'.format(self.save_dir))

    def save(self):

        self.plot_width()
        total_samples = self.tp + self.fn

        mean_iou = round(sum(self.iou_list) / len(self.iou_list), 4)

        recall = round(self.tp / total_samples, 4)
        precision = round(self.tp / (self.tp + self.fp), 4)
        f_one = round(self.tp / (self.tp + (0.5 * (self.fp + self.fn))), 4)

        f = open("{}/results.txt".format(self.save_dir), 'w')
        f.write('The number of Total Image: ' +
                ' ' + str(total_samples) + '\n')
        f.write('\n')
        f.write('False Negatives: ' + ' ' + str(self.fn) + '\n')
        f.write('False Positives: ' + ' ' + str(self.fp) + '\n')
        f.write('True Positives:: ' + ' ' + str(self.tp) + '\n')
        f.write('\n')

        f.write('Precision: ' + ' ' + str(precision) + '\n')
        f.write('Recall: ' + ' ' + str(recall) + '\n')
        f.write('F1 Score: ' + ' ' + str(f_one) + '\n')
        f.write('Mean Iou: ' + ' ' + str(mean_iou) + '\n')
        f.write('\n')

        for cls in self.result_dict:
            tp, fp, fn = self.result_dict[cls]['tp'], self.result_dict[cls]['fp'], self.result_dict[cls]['fn']
            total_sample = tp + fn
            recall = round(tp / total_sample, 4)
            precision = round(tp / (tp + fp), 4)
            f_one = round(tp / (tp + (0.5 * (fp + fn))), 4)
            f.write('Class:' + ' ' + class_name_map[cls] + '\n')
            f.write('\n')
            f.write('Precision: ' + ' ' + str(precision) + '\n')
            f.write('Recall: ' + ' ' + str(recall) + '\n')
            f.write('F1 Score: ' + ' ' + str(f_one) + '\n')
            f.write('\n')
        f.close()


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    results = TestSmallObject(0.2, 'parti480')
    for path, im, im0s, vid_cap, s in tqdm(dataset):
        gt_path = path[:path.rfind('.')] + '.txt'
        img_height, img_width = im0s.shape[:2]
        results.read_gt(gt_path, img_height, img_width)
        label_colors = color_generator()
        img_boxes, img_confs, img_labels, partition_rectangles = get_detections_from_partitions(
            model, im0s, 480, dt)
        for i, rect in enumerate(partition_rectangles):
            cv2.rectangle(im0s, rect[0], rect[1], (0, 0, 0), 1)

        for i, box in enumerate(img_boxes):
            x_min = box[0]
            y_min = box[1]
            w = box[2]
            h = box[3]
            x_max = x_min + int(w)
            y_max = y_min + int(h)
            results.compare_detections([img_labels[i], x_min, y_min, x_max, y_max])
            cv2.rectangle(im0s, (x_min, y_min), (x_max, y_max), label_colors[img_labels[i]], 2)
            cv2.putText(im0s, class_name_map[img_labels[i]], (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 1)

        results.false_negative_counter()
        cv2.imwrite(os.path.join('parti480', path.split('/')[-1]), im0s)
    results.save()
    # Print time (inference-only)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

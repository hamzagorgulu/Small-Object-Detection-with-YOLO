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

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

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


class TestSmallObject:
    def __init__(self, iou_thres, save_dir):
        self.save_dir = save_dir
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
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

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

    results = TestSmallObject(0.2, '480part')
    for path, im, im0s, vid_cap, s in dataset:
        gt_path = path[:path.rfind('.')] + '.txt'
        img_height, img_width = im0s.shape[:2]
        results.read_gt(gt_path, img_height, img_width)

        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    x_min = xyxy[0].item()
                    y_min = xyxy[1].item()
                    x_max = xyxy[2].item()
                    y_max = xyxy[3].item()
                    results.compare_detections([cls.item(), x_min, y_min, x_max, y_max])

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            results.false_negative_counter()

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    results.save()
    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


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

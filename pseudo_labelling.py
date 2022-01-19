import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from glob import glob
import itertools
import torch

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
setup_logger()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    base_dir = "/home/samuelkim/.kaggle/data/sartorius"
    semi_data_dir = Path(f"{base_dir}/train_semi_supervised")
    
    # configurations setup
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.INPUT.MASK_FORMAT='bitmask'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.WEIGHTS = "/home/samuelkim/mask_rcnn_checkpoint/model_0003029.pth"
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000

    # initialize predictor
    predictor = DefaultPredictor(cfg)

    annotations = {
        "images": [],
        "annotations": [],
        "categories": [
            {"name": "shsy5y", "id": 1},
            {"name": "astro", "id": 2},
            {"name": "cort", "id": 3}
        ]
    }
    
    # prediction and pseudo labeling
    ids, masks = [], []
    test_names = sorted(glob(os.path.join(semi_data_dir, "*")))
    tbar = tqdm(range(len(test_names)))
    for i in tbar:
        image_dict, annotation_list = get_outputs2coco(test_names[i], predictor)
        annotations["images"].append(image_dict)
        annotations["annotations"].extend(annotation_list)

    pseudo_label_dir = "/home/samuelkim/workspace/pseudo_label/model_0003029.json"

    with open(pseudo_label_dir, "w") as f:
        json.dump(annotations, f)


THRESHOLDS = [.18, .38, .58]
MIN_PIXELS = [75, 150, 75]

def rle_decode(mask_rle, shape=(520, 704)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction

def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def get_masks(fn, predictor):
    im = cv2.imread(str(fn))
    pred = predictor(im)
    pred_class = torch.mode(pred['instances'].pred_classes)[0]
    take = pred['instances'].scores >= THRESHOLDS[pred_class]
    pred_masks = pred['instances'].pred_masks[take]
    pred_masks = pred_masks.cpu().numpy()
    res = []
    used = np.zeros(im.shape[:2], dtype=int)
    for mask in pred_masks:
        mask = mask * (1 - used)
        if mask.sum() >= MIN_PIXELS[pred_class]:  # skip predictions with small area
            used += mask
            res.append(rle_encode(mask))
    return res

def get_outputs(fn, predictor):
    im = cv2.imread(str(fn))
    pred = predictor(im)
    pred_class = torch.mode(pred['instances'].pred_classes)[0]
    take = pred['instances'].scores >= THRESHOLDS[pred_class]
    pred_masks = pred['instances'].pred_masks[take]
    pred_masks = pred_masks.cpu().numpy()
    res = []
    used = np.zeros(im.shape[:2], dtype=int)
    for mask in pred_masks:
        mask = mask * (1 - used)
        if mask.sum() >= MIN_PIXELS[pred_class]:  # skip predictions with small area
            used += mask
    return used

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(itertools.groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

N_CELL = 0

def get_outputs2coco(fn, predictor):
    global N_CELL
    im = cv2.imread(str(fn))
    pred = predictor(im)
    pred_class = torch.mode(pred['instances'].pred_classes)[0]
    take = pred['instances'].scores >= THRESHOLDS[pred_class]
    pred_masks = pred['instances'].pred_masks[take]
    pred_masks = pred_masks.cpu().numpy()
    res = []
    used = np.zeros(im.shape[:2], dtype=int)

    img_name = str(Path(fn).name)

    image_dict = {
        "file_name": os.path.join("train_semi_supervised", img_name),
        "height": im.shape[0],
        "width": im.shape[1],
        "id": img_name.replace(".png", ""),
    }

    cell_type = img_name.split("[")[0]
    if cell_type.startswith("astro"):
        cell_type = "astro"
    annotation_list = []

    for mk in pred_masks:
        # mk = mk * (1-used)
        if mk.sum() >= MIN_PIXELS[pred_class]:  # skip predictions with small area
            # used += mask
            ys, xs = np.where(mk)
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            enc = binary_mask_to_rle(mk)
            seg = {
                'segmentation': enc,
                'bbox': [int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1)],
                'area': int(np.sum(mk)),
                'image_id': img_name.replace(".png", ""),
                'category_id': {"shsy5y": 1, "astro": 2, "cort": 3}[cell_type],
                'iscrowd': 0,
                'id': N_CELL
            }
            annotation_list.append(seg)
            N_CELL += 1

    return image_dict, annotation_list

if __name__ == '__main__':
    main()
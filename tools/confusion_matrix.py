# Output TP, FN, FP for object detetcion (single class) using detectron2 Model
# By Tianlong Jia

# Some detectors can output multiple detections overlapping a single ground truth;
# For those cases, only one detection is counted as a TP,
#  and the others are counted as FPs

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import matplotlib.pyplot as plt
import os, json, cv2, random, gc
from detectron2.data.datasets import register_coco_instances
from detectron2.checkpoint import DetectionCheckpointer, Checkpointer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.modeling import build_model
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import pandas as pd
import torch
from detectron2.structures import Boxes, pairwise_iou

# transfer coco format (Xmin, Ymin, width, height) to (Xmin, Ymin, Xmax, Ymax)
def coco_bbox_to_coordinates(bbox):
    out = bbox.copy().astype(float)
    # print("out: ", out)
    out[0] = bbox[0]
    out[1] = bbox[1]
    out[2] = bbox[0] + bbox[2]
    out[3] = bbox[1] + bbox[3]
    return out

def conf_matrix_calc(labels, detections, iou_thresh):
    TPs = 0  # the number of litter detected correctly
    FP = 0   # the number of bbox detected non-litter as litter
    FN = 0   # the number of litter not detected
    d_bbox_match = 0 # the number of detected bbox match the ground-truth
    # if there is one label, maybe more than one detected bbox match it. 
    # print("len(detections): ",len(detections))
    for label in labels:
      # print("label_bbox: ", label[1:])
      l_bbox = coco_bbox_to_coordinates(label[1:]) # transfer gt-labels form 
      # print("l_bbox: ", l_bbox)
      # capture detetcions with IoU over IoU_thresh
      TP = 0
      for detection in detections:
        d_conf = np.array(detection)[4]
        # print("detection_bbox: ", detection[:4])
        # print("d_conf: ", d_conf)
        d_bbox = detection[:4]  # not need to transfer to (Xmin, Ymin, Xmax, Ymax)
        d_class = np.array(detection)[-1].astype(int)

        # Note: the bbox input of "pairwise_iou" should be (Xmin, Ymin, Xmax, Ymax)
        iou = pairwise_iou(Boxes(torch.from_numpy(np.array([l_bbox]))), Boxes(torch.from_numpy(np.array([d_bbox]))))
        # print("iou: ", iou)
        if iou >= iou_thresh:
            TP = 1
            d_bbox_match = d_bbox_match + 1
      TPs = TPs + TP
    FP = len(detections) - TPs
    FN = len(labels) - TPs
    # print("TPs in one image: ", TPs)
    # print("FP in one image: ", FP)
    # print("FN in one image: ", FN)
    return TPs, FP, FN

def confusion_matrix(model_config_path, model_checkpoint_path, test_annotations_path, test_images_path):
  cfg = get_cfg()
  cfg.merge_from_file(model_config_path)  ############## identify the model_config_path
  cfg.MODEL.WEIGHTS = model_checkpoint_path   ############## identify the model_checkpoint_path
  cfg.DATASETS.TEST = ("validation-corn",)
  cfg.DATALOADER.NUM_WORKERS = 2
  cfg.SOLVER.IMS_PER_BATCH = 2
  cfg.SOLVER.BASE_LR = 0.00025
  cfg.SOLVER.MAX_ITER = 10000
  cfg.SOLVER.STEPS = []
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 
  register_coco_instances(
    "validation-corn",
    {},
    test_annotations_path,
    test_images_path
    )
  dataset_dicts_validation = DatasetCatalog.get("validation-corn")
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # output detected bbox with confidence above this score
  predictor = DefaultPredictor(cfg)
  
  FN_total=0
  TP_total=0
  FP_total=0
  # print("dataset_dicts_validation: ", len(dataset_dicts_validation))
  for d in dataset_dicts_validation:
    FN_0 = 0
    TP = 0
    FP = 0
    FN = 0
    img = cv2.imread(d["file_name"])
    outputs = predictor(img)
    labels = list()
    detections = list()
    # get the ground-truth bbox information from dataset with labels
    for i in range(len(d["annotations"])):
      labels_cls=d["annotations"][i]["category_id"]
      labels_bbox=d["annotations"][i]["bbox"]
      labels.append([labels_cls] + list(labels_bbox))
    # print("labels: ", np.array(labels))
    
    # get the predicted bbox information
    for coord, conf, cls in zip(
        outputs["instances"].get("pred_boxes").tensor.cpu().numpy(),
        outputs["instances"].get("scores").cpu().numpy(),
        outputs["instances"].get("pred_classes").cpu().numpy()
    ):
        detections.append(list(coord) + [conf] + [cls])     
    # "detections" is the predicted results with class confidence and bbox
        
    # Note: (1) The form of predicted bbox (output by detectron2) is (Xmin, Ymin, Xmax, Ymax)
        # (2) The form of ground-truth bbox (load from json file) is (Xmin, Ymin, width, height)
      

    if detections==[]:
      FN_0 = FN_0 + len(labels) # save the FN if detections is empty
    if detections!=[]:
      TP, FP, FN = conf_matrix_calc(np.array(labels), np.array(detections), iou_thresh=0.5) 
    # print("****************")
    FN_total = FN_total + FN + FN_0
    TP_total = TP_total + TP
    FP_total = FP_total + FP
  
#   print("TP:", TP_total)
#   print("FP:", FP_total)
#   print("FN:", FN_total)
  return TP_total, FP_total, FN_total



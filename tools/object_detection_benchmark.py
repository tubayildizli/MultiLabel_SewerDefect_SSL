# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Full credits: https://github.com/facebookresearch/moco/blob/main/detection/train_net.py  # NOQA
"""


import os
import sys
import torch
import datetime
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import (
    default_argument_parser,
    default_setup,
    DefaultTrainer,
    launch as d2_launch,
)
from detectron2.evaluation import (
    COCOEvaluator, 
    PascalVOCDetectionEvaluator,
    DatasetEvaluators
)
from detectron2.layers import get_norm
from detectron2.modeling.roi_heads import Res5ROIHeads, ROI_HEADS_REGISTRY
from detectron2.data.datasets import register_coco_instances


from detectron2.engine.train_loop import HookBase
import logging


sys.path.append('/scratch/tjian/PythonProject/deep_plastic_SSL/')


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeadsExtraNorm(Res5ROIHeads):
    """
    As described in the MOCO paper, there is an extra BN layer
    following the res5 stage.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        norm = cfg.MODEL.RESNETS.NORM
        norm = get_norm(norm, self.res5[-1].out_channels)
        self.res5.add_module("norm", norm)


class BestCheckpointer(HookBase):

    def before_train(self):
        self.best_metric = 0.0
        self.logger = logging.getLogger("detectron2.trainer")
        self.logger.info("######## Running best check pointer")

    def after_step(self):
    
    # Choose the metric to validate the model
        metric_name="bbox/AP50"
        # metric_name="bbox/AP"
        # metric_name="bbox/AP75"
        # metric_name="bbox/APs"
        # metric_name="bbox/APm"
        # metric_name="bbox/APl"
        
        if metric_name in self.trainer.storage._history:
            eval_metric, batches = self.trainer.storage.history(metric_name)._data[-1]
            if self.best_metric < eval_metric:
                self.best_metric = eval_metric
                self.logger.info(f"######## New best metric: {self.best_metric}")
                self.trainer.checkpointer.save(f"model_best_{eval_metric:.4f}")
                

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if "coco" in dataset_name:
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        else:
            assert "voc" in dataset_name, "Dataset must be coco or voc"
            return PascalVOCDetectionEvaluator(dataset_name)
            
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        coco_evaluator = COCOEvaluator(dataset_name, output_dir=output_folder)
        evaluator_list = [coco_evaluator]
        return DatasetEvaluators(evaluator_list)
        
    def build_hooks(self):
        ret = super().build_hooks()
        ret.append(BestCheckpointer())
        return ret


def main(args):
    # clean the occupied cuda memory 
    torch.cuda.empty_cache()

    # register my custom dataset, if your dataset is in COCO format:
    
    register_coco_instances("Jakarta_train", {}, "/scratch/tjian/Data/Jakarta_coco/annotations/instances_train_DelftBlue.json", "/scratch/tjian/Data/Jakarta_coco/train/")
    register_coco_instances("Jakarta_val", {}, "/scratch/tjian/Data/Jakarta_coco/annotations/instances_val_DelftBlue.json", "/scratch/tjian/Data/Jakarta_coco/val/")
    
    # GJO_train_2050
    register_coco_instances("GJO_train_2050", {}, "/scratch/tjian/Data/GJO_SSL/SL_train_2050/annotations/SL_train_2050.json", "/scratch/tjian/Data/GJO_SSL/SL_train_2050/train/")
    register_coco_instances("GJO_val_2050", {}, "/scratch/tjian/Data/GJO_SSL/SL_train_2050/annotations/SL_val_199.json", "/scratch/tjian/Data/GJO_SSL/SL_train_2050/val/")
    
    # GJO_train_1626
    register_coco_instances("GJO_train_1626", {}, "/scratch/tjian/Data/GJO_SSL/SL_train_1626/annotations/SL_train_1626.json", "/scratch/tjian/Data/GJO_SSL/SL_train_1626/train/")
    register_coco_instances("GJO_val_1626", {}, "/scratch/tjian/Data/GJO_SSL/SL_train_1626/annotations/SL_val_161.json", "/scratch/tjian/Data/GJO_SSL/SL_train_1626/val/")
    
    # GJO_train_1243
    register_coco_instances("GJO_train_1243", {}, "/scratch/tjian/Data/GJO_SSL/SL_train_1243/annotations/SL_train_1243.json", "/scratch/tjian/Data/GJO_SSL/SL_train_1243/train/")
    register_coco_instances("GJO_val_1243", {}, "/scratch/tjian/Data/GJO_SSL/SL_train_1243/annotations/SL_val_118.json", "/scratch/tjian/Data/GJO_SSL/SL_train_1243/val/")
    
    # GJO_train_778
    register_coco_instances("GJO_train_778", {}, "/scratch/tjian/Data/GJO_SSL/SL_train_778/annotations/SL_train_778.json", "/scratch/tjian/Data/GJO_SSL/SL_train_778/train/")
    register_coco_instances("GJO_val_778", {}, "/scratch/tjian/Data/GJO_SSL/SL_train_778/annotations/SL_val_81.json", "/scratch/tjian/Data/GJO_SSL/SL_train_778/val/")
    
    # GJO_train_411
    register_coco_instances("GJO_train_411", {}, "/scratch/tjian/Data/GJO_SSL/SL_train_411/annotations/SL_train_411.json", "/scratch/tjian/Data/GJO_SSL/SL_train_411/train/")
    register_coco_instances("GJO_val_411", {}, "/scratch/tjian/Data/GJO_SSL/SL_train_411/annotations/SL_val_40.json", "/scratch/tjian/Data/GJO_SSL/SL_train_411/val/")


    
    
    # setup the config file
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    # trainer = Trainer(cfg)
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":

    starttime = datetime.datetime.now()

    args = default_argument_parser().parse_args()
    print("Arguments:", args)
    d2_launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    
    endtime = datetime.datetime.now()
    # calculate taining time
    days = (endtime - starttime).days
    seconds = (endtime - starttime).seconds
    training_time = round((days*24*60*60+seconds)/60)
    print("training_time: ", training_time, " min")
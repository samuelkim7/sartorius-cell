import os
import numpy as np
from pathlib import Path
import pycocotools.mask as mask_util

import detectron2.utils.comm as comm
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, \
                              default_setup, hooks, launch, BestCheckpointer, PeriodicWriter
from detectron2.evaluation import verify_results
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.data.datasets import register_coco_instances


def setup(args):
    """
    Create configs and perform basic setups.
    """
    data_dir = '/home/samuelkim/.kaggle/data/sartorius'
    data_path = Path(data_dir)

    cfg = get_cfg()
    register_coco_instances('sartorius_train',{}, f'{data_dir}/json_kaggle/annotations_train_090_1.json', data_path)
    register_coco_instances('sartorius_val',{}, f'{data_dir}/json_kaggle/annotations_val_010_1.json', data_path)

    cfg.MODEL_NAME = 'mask_rcnn_R_50_FPN_3x_pseudo_v2_with_train090_from_3029_test'
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    pretrained_dir = '/home/samuelkim/workspace/detectron2/tools/outputs'
    cfg.MODEL.WEIGHTS = f'{pretrained_dir}/mask_rcnn_R_50_FPN_3x_pseudo_v2_with_train090_from_3029/model_0025159.pth'

    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.DATASETS.TRAIN = ("sartorius_train",)
    cfg.DATASETS.TEST = ("sartorius_val",)
    cfg.DATALOADER.NUM_WORKERS = 12
    cfg.SOLVER.IMS_PER_BATCH = 2
    # cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.STEPS = []
    itrs_per_epoch = ( len(DatasetCatalog.get('sartorius_train')) ) // cfg.SOLVER.IMS_PER_BATCH
    cfg.SOLVER.WARMUP_ITERS = itrs_per_epoch 
    cfg.SOLVER.MAX_ITER = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = .5
    cfg.TEST.EVAL_PERIOD = 1
    cfg.SOLVER.CHECKPOINT_PERIOD = itrs_per_epoch
    cfg.OUTPUT_DIR = '/home/samuelkim/workspace/detectron2/tools/outputs/' + cfg.MODEL_NAME + '/'

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    return np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)


def score(pred, targ):
    pred_masks = pred['instances'].pred_masks.cpu().numpy()
    enc_preds = [mask_util.encode(np.asarray(p, order='F')) for p in pred_masks]
    enc_targs = list(map(lambda x:x['segmentation'], targ))
    ious = mask_util.iou(enc_preds, enc_targs, [0]*len(enc_targs))
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, ious)
        p = tp / (tp + fp + fn)
        prec.append(p)
    return np.mean(prec)


class MAPIOUEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name):
        dataset_dicts = DatasetCatalog.get(dataset_name)
        self.annotations_cache = {item['image_id']:item['annotations'] for item in dataset_dicts}
            
    def reset(self):
        self.scores = []

    def process(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            if len(out['instances']) == 0:
                self.scores.append(0)    
            else:
                targ = self.annotations_cache[inp['image_id']]
                self.scores.append(score(out, targ))

    def evaluate(self):
        return {"MaP IoU": np.mean(self.scores)}


class Trainer(DefaultTrainer):  
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return MAPIOUEvaluator(dataset_name)
    
    def build_hooks(self):
        cfg = self.cfg.clone()
        hooks = super().build_hooks()
        hooks.insert(-1, BestCheckpointer(cfg.TEST.EVAL_PERIOD, 
                                         DetectionCheckpointer(self.model, cfg.OUTPUT_DIR),
                                         "MaP IoU",
                                         "max",
                                         ))
        for hook in hooks:
            if isinstance(hook, PeriodicWriter):
                hooks.remove(hook)
        hooks.append(PeriodicWriter(self.build_writers(), period=cfg.TEST.EVAL_PERIOD))
        return hooks


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

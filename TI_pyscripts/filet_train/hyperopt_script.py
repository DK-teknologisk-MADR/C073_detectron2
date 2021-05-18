import random
import os , shutil , json
from copy import deepcopy
from numpy.random import choice, randint,uniform
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetMapper, build_detection_train_loader
import detectron2.data.transforms as T
from detectron2.evaluation import COCOEvaluator
from pruners import SHA
from trainers import TrainerPeriodicEval
from hyperoptimization import D2_hyperopt_Base
from numpy import random
from data_utils import get_data_dicts, register_data
import datetime

splits = ['train','val']
data_dir = "/pers_files/Combined_final/Filet"
COCO_dicts = {split: get_data_dicts(data_dir,split) for split in splits } #converting TI-annotation of pictures to COCO annotations.
data_names = register_data('filet',['train','val'],COCO_dicts,{'thing_classes' : ['filet']}) #register data by str name in D2 api
model_name = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x"




class D2_hyperopt(D2_hyperopt_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('task is',self.task)

    def suggest_helper(self,i):
        if i==0:
            return [[32, 64, 128, 256, 512]]
        elif i==1:
            return [[64, 128, 256, 512]]
        elif i==2:
            return [[128, 256, 512]]
        else:
            raise ValueError

    def suggest_values(self):
        hps = [
            (['model', 'backbone', 'freeze_at'], random.randint(0, 3)),
            (['model', 'anchor_generator', 'sizes'],self.suggest_helper(random.randint(0,3))),
            (['model', 'anchor_generator', 'aspect_ratios'], random.choice([[0.5, 1.0, 2.0], [0.25, 0.5, 1.0, 2.0]])),
            (['solver', 'BASE_LR'], random.uniform(0.0001, 0.0004)),
            (['model', 'roi_heads', 'batch_size_per_image'], int(random.choice([128, 256, 512]))),
        ]
        return hps

    def prune_handling(self,pruned_ids):
        for trial_id in pruned_ids:
            shutil.rmtree(self.get_trial_output_dir(trial_id))



def initialize_base_cfg(model_name,output_dir,cfg=None):
    '''
    setup base configuration of model SEE MORE AT https://detectron2.readthedocs.io/en/latest/modules/config.html
    '''
    if cfg is None:
        cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(f'{model_name}.yaml'))
    cfg.DATASETS.TRAIN = (data_names['train'],)
    cfg.DATASETS.TEST = []
    cfg.TEST.EVAL_PERIOD = 0 #set >0 to activate evaluation
    cfg.DATALOADER.NUM_WORKERS = 6 #add more workerss until it gives warnings.
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f'{model_name}.yaml')
    cfg.SOLVER.IMS_PER_BATCH = 4 #maybe more?
    cfg.OUTPUT_DIR = f'{output_dir}/{model_name}_output'
    cfg.SOLVER.WARMUP_ITERS = 500
    os.makedirs(f'{output_dir}/{model_name}_output',exist_ok=True)
    cfg.SOLVER.MAX_ITER = 100000
    cfg.SOLVER.STEPS = [] #cfg.SOLVER.STEPS = [2000,4000] would decay LR by cfg.SOLVER.GAMMA at steps 2000,4000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  #(default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    return cfg
output_dir = f'{data_dir}/output'

# model_name2 = 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x'
cfg = initialize_base_cfg(model_name,output_dir)
print("------------------------------------------")
print("------------------------------------------")
print("------------------------------------------")
print("------------------------------------------")

print(cfg.SOLVER.WARMUP_ITERS)
print("------------------------------------------")
print("------------------------------------------")
print("------------------------------------------")
print("------------------------------------------")
print("------------------------------------------")

task = 'segm'
evaluator = COCOEvaluator(data_names['val'],("segm",), False,cfg.OUTPUT_DIR)
round1 = D2_hyperopt(model_name,cfg,data_val_name=data_names['val'],task=task,evaluator=evaluator,output_dir=output_dir,step_chunk_size=600,max_iter=100000,pr_params={'factor' : 4,'topK' : 4})
res1 = round1.start()
output_dir2 = f"{data_dir}/output2"
cfg = initialize_base_cfg(model_name,output_dir2)
round2 = D2_hyperopt(model_name,cfg,data_val_name=data_names['val'],task=task,evaluator=evaluator,output_dir=output_dir2,step_chunk_size=130,max_iter=100000,pr_params={'factor' : 3, 'topK' : 3})
res2 = round2.start()
print(res1)
print(res2)
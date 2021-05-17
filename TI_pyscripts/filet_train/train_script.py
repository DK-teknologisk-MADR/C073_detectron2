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
data_dir = ""
COCO_dicts = {split: get_data_dicts(data_dir,split) for split in splits } #converting TI-annotation of pictures to COCO annotations.
data_names = register_data('filet',['train','val'],COCO_dicts,{'thing_classes' : ['filet']}) #register data by str name in D2 api
output_dir = f'{data_dir}/output'
model_name = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x"


def suggest_values():
    hps = [
        (['model','backbone','freeze_at'],random.randint(0,3)),
        (['model','anchor_generator','sizes'], random.choice([[[32,64,128,256,512]],[[64,128,256,512]],[[128,256,512]]])),
        (['model','anchor_generator','aspect_ratios'],random.choice([[0.5,1.0,2.0],[0.25,0.5,1.0,2.0]])),
        (['solver','BASE_LR'],random.uniform(0.0001,0.0006)),
        (['model', 'roi_heads','batch_size_per_image'], random.choice([128,256,512])),
    ]



def initialize_base_cfg(model_name,cfg=None):
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
    os.makedirs(f'{output_dir}/{model_name}_output',exist_ok=True)
    cfg.SOLVER.MAX_ITER = 100000
    cfg.SOLVER.STEPS = [] #cfg.SOLVER.STEPS = [2000,4000] would decay LR by cfg.SOLVER.GAMMA at steps 2000,4000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  #(default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    return cfg

# model_name2 = 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x'
cfg = initialize_base_cfg(model_name)
task = 'segm'
evaluator = COCOEvaluator(data_names['val'],("segm",), False,cfg.OUTPUT_DIR)
round1 = D2_hyperopt_Base(model_name,cfg,data_val_name=data_names['val'],task=task,evaluator=evaluator,output_dir=output_dir,step_chunk_size=250,max_iter=50000,pr_params={'factor' : 4, 'topK' : 4})
round2 = D2_hyperopt_Base(model_name,cfg,data_val_name=data_names['val'],task=task,evaluator=evaluator,output_dir=output_dir,step_chunk_size=250,max_iter=50000,pr_params={'factor' : 4, 'topK' : 4})

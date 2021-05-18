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

splits = ['train','val']
data_dir = "/pers_files/test_set"
COCO_dicts = {split: get_data_dicts(data_dir,split) for split in splits } #converting TI-annotation of pictures to COCO annotations.
data_names = register_data('filet',['train','val'],COCO_dicts,{'thing_classes' : ['filet']}) #register data by str name in D2 api
output_dir = f'{data_dir}/output'
print(data_names)


def initialize_base_cfg(model_name,cfg=None):
    '''
    setup base configuration of model SEE MORE AT https://detectron2.readthedocs.io/en/latest/modules/config.html
    '''
    if cfg is None:
        cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(f'{model_name}.yaml'))
    cfg.DATASETS.TRAIN = (data_names['train'],)
#    cfg.DATASETS.TEST = (data_names['val'],) # Use this with trainer_cls : TrainerPeriodicEval if you want to do validation every #.TEST.EVAL_PERIOD iterations
    cfg.DATASETS.TEST = []
    cfg.TEST.EVAL_PERIOD = 0 #set >0 to activate evaluation
    cfg.DATALOADER.NUM_WORKERS = 6 #add more workerss until it gives warnings.
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f'{model_name}.yaml')
    cfg.SOLVER.IMS_PER_BATCH = 3 #maybe more?
    cfg.OUTPUT_DIR = f'{output_dir}/{model_name}_output'
    os.makedirs(f'{output_dir}/{model_name}_output',exist_ok=True)
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 500
    cfg.SOLVER.STEPS = [] #cfg.SOLVER.STEPS = [2000,4000] would decay LR by cfg.SOLVER.GAMMA at steps 2000,4000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64  #(default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    return cfg

#example input
model_name = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x"
#model_name2 = 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x'
cfg = initialize_base_cfg(model_name)


class D2_hyperopt(D2_hyperopt_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('task is',self.task)


    def suggest_values(self):
        hps = [
            (['solver', 'BASE_LR'], random.uniform(0.0001, 0.0006)),
        ]
        return hps



    def prune_handling(self,pruned_ids):
        for trial_id in pruned_ids:
            shutil.rmtree(self.get_trial_output_dir(trial_id))


task = 'bbox'
evaluator = COCOEvaluator(data_names['val'],("bbox", "segm"), False,cfg.OUTPUT_DIR)

#hyperoptimization object that uses model_dict to use correct model, and get all hyper-parameters.
#optimized after "task" as computed by "evaluator". The pruner is (default) SHA, with passed params pr_params.
#number of trials, are chosen so that the maximum total number of steps does not exceed max_iter.

hyp = D2_hyperopt(model_name,cfg_base=cfg,data_val_name = data_names['val'],trainer_cls=DefaultTrainer,task=task,evaluator=evaluator,step_chunk_size=10,output_dir=output_dir,pruner_cls=SHA,max_iter = 350)
best_models = hyp.start()
#returns pandas object
print(best_models)


#-----------------------------
#example of a training procedure
class TrainerWithMapper(DefaultTrainer):
    '''
    Example of a trainer that applies argumentations at runtime. Argumentations available can be found here:
    https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html
    '''
    def __init__(self,augmentations,**params_to_DefaultTrainer):
        super().__init__(**params_to_DefaultTrainer)
        self.augmentations=augmentations

    #overwrites default build_train_loader
    @classmethod
    def build_train_loader(cls, cfg):
          mapper = DatasetMapper(cfg, is_train=True, augmentations=augmentations)
          return build_detection_train_loader(cfg,mapper=mapper)

augmentations = [
          T.RandomCrop('relative_range',[0.7,0.7]),
          T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
          T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
          T.RandomRotation(angle = [-20,20], expand=True, center=None, sample_style='range'),
          T.RandomBrightness(0.85,1.15)
              ]

#trainer = TrainerWithMapper(augmentations = augmentations,cfg=cfg)
#trainer.resume_or_load(resume=True)
#trainer.train()

#--------------------------------------------


#example of training with periodic evaluation
cfg.DATASETS.TEST = (data_names['val'],)
cfg.TEST.EVAL_PERIOD = 10
class TrainerWithEval(TrainerWithMapper):

    @classmethod
    def build_evaluator(cls,cfg,dataset_name,output_folder=None):
        if output_folder is None:
            output_folder = cfg.OUTPUT_DIR
        return COCOEvaluator(dataset_name, ('bbox', 'segm'), False, output_dir=output_folder)

#if argumentations are not neccesary, one does not need subclass
#trainer = TrainerWithMapper(cfg)
#trainer.resume_or_load(resume=True)
#trainer.train()

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
from D2TIDefaults import get_data_dicts, register_data,D2_hyperopt_Base,TrainerPeriodicEval



splits = ['train','val']
data_dir = "/pers_files/test_set"
COCO_dicts = {split: get_data_dicts(data_dir,split) for split in splits } #converting TI-annotation of pictures to COCO annotations.
data_names = register_data('filet',['train','val'],COCO_dicts,{'thing_classes' : ['filet']}) #register data by str name in D2 api
output_dir = f'{data_dir}/output'
print(data_names)


def initialize_base_cfg(model_name="",cfg=None):
    '''
    setup base configuration of model SEE MORE AT https://detectron2.readthedocs.io/en/latest/modules/config.html
    '''
    if cfg is None:
        cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(f'{model_name}.yaml'))
    cfg.DATASETS.TRAIN = (data_names['train'],)
#    cfg.DATASETS.TEST = (data_names['val'],) # Use this with trainer_cls : TrainerPeriodicEval if you want to do validation after every iterations
    cfg.DATASETS.TEST = []
    cfg.TEST.EVAL_PERIOD = 0
    cfg.TEST.EVAL_PERIOD = 8
    cfg.DATALOADER.NUM_WORKERS = 6 #add more workerss until it gives warnings.
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f'{model_name}.yaml')
    cfg.SOLVER.IMS_PER_BATCH = 3 #maybe more?
    cfg.OUTPUT_DIR = f'{output_dir}/{model_name}_output'
    os.makedirs(f'{output_dir}/{model_name}_output',exist_ok=True)
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 100
    cfg.SOLVER.STEPS = [] #cfg.SOLVER.STEPS = [2000,4000] would decay LR by cfg.SOLVER.GAMMA at steps 2000,4000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64  #(default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    return cfg



#example input
model_name = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x"
#model_name2 = 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x'
solver_dict = {
    'BASE_LR' : ('float',{'name' : 'lr', 'low' : 0.0001 , 'high' : 0.001}),
    'GAMMA'  : ('float', {'name' : 'gamma', 'low' : 0.005 , 'high' : 0.8}),
    }
backbone_dict = {
    'FREEZE_AT' : ('int',{'name' : 'freeze', 'low' : 0 , 'high' : 2})
    }
model_dict = {'BACKBONE' : backbone_dict}

hp_dict = {'SOLVER' : solver_dict,
           'MODEL' : model_dict,
          }

model_dict = {'name' : model_name,
            'base_cfg': initialize_base_cfg(model_name),
            'hp_dict' : {'SOLVER' : solver_dict,
                         'MODEL' : model_dict}}



class D2_hyperopt(D2_hyperopt_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('task is',self.task)


    def initialize(self,cfg,params):
            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
            to_json = [cfg, params]
            with open(f'{cfg.OUTPUT_DIR}/hyper_params.json', 'w') as fp:
                json.dump(to_json, fp)
            print('AT INITIALIZE')
            for cfg in self.suggested_cfgs:
                print(cfg.OUTPUT_DIR)

    def suggest_values(self,typ,params):
        params_wo_name = deepcopy(params)
        params_wo_name.pop('name')
        if typ == "int":
            return randint(**params_wo_name)
        elif typ == "float":
            return uniform(**params_wo_name)
       # elif typ == "categorical":
        #    return choice(**params_wo_name)

    def prune_handling(self,pruned_ids):
        for trial_id in pruned_ids:
            shutil.rmtree(self.get_trial_output_dir(trial_id))


cfg = initialize_base_cfg(model_name)
task = 'bbox'
evaluator = COCOEvaluator(data_names['val'],("bbox", "segm"), False,cfg.OUTPUT_DIR)

#hyperoptimization object that uses model_dict to use correct model, and get all hyper-parameters.
#optimized after "task" as computed by "evaluator". The pruner is (default) SHA, with passed params pr_params.
#number of trials, are chosen so that the maximum total number of steps does not exceed max_iter.

hyp = D2_hyperopt(model_dict,data_names=data_names,trainer_cls=DefaultTrainer,task=task,evaluator=evaluator,step_chunk_size=10,output_dir=output_dir,pruner_cls=SHA,max_iter = 350)
best_models = hyp.start()
print('BEST RESULTS ARE',best_models)


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

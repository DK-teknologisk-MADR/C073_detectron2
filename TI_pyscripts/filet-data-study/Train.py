import datetime

import numpy.random
import hooks
import os , shutil , json
from copy import deepcopy
from numpy.random import choice, randint,uniform
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetMapper, build_detection_train_loader
import detectron2.data.transforms as T
from detectron2.evaluation import COCOEvaluator
from math import log, floor
from trainers import TrainerPeriodicEval
from hyperoptimization import D2_hyperopt_Base
from data_utils import get_data_dicts, register_data
from numpy.random import choice
from detectron2.data import DatasetCatalog,MetadataCatalog
import re
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

splits = ['train','val']
data_dir = "/pers_files/Combined_final/Filet"
#data_dir = "/pers_files/test_set"
base_output_dir = f'{data_dir}/output'
DatasetCatalog.clear()
MetadataCatalog.clear()
model_name = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x"


def get_pairs(split):
    files = os.listdir(os.path.join(data_dir,split))
    jpg_fronts = [x.split(".")[0] for x in files if x.endswith(".jpg")]
    data_pairs = {x : [x+".jpg" , x + ".json"] for x in jpg_fronts if  x + ".json" in files}
    return data_pairs
train_pairs = get_pairs('train')
val_pairs = get_pairs('val')
regex_string = r'202[0-9]-'

year_finder = re.compile(regex_string)
def partition_pairs_by_year(data_pairs):

    data_pairs_2020, data_pairs_2021 = {}, {}
    for front,files in data_pairs.items():
        ma = year_finder.search(front)
        if ma:
            if ma.group()=='2020-':
                data_pairs_2020[front] = files
            elif ma.group()=='2021-':
                data_pairs_2021[front] = files
            else:
                print("warning: unknown matched year")
                print(front,ma.group())

        else:
            print("warning: unknown year")
            print(front)
    return data_pairs_2020,data_pairs_2021
data_pairs_2020 , data_pairs_2021 = partition_pairs_by_year(train_pairs)
_ , data_pairs_val = partition_pairs_by_year(val_pairs)
DatasetCatalog.register('filet_val', lambda : get_data_dicts(data_dir, 'val', data_pairs_val))
MetadataCatalog.get('filet_val').set(thing_classes=['filet'])

def generate_train_sets(include2020 = True):
    if include2020:
        sample_space = train_pairs
    else:
        sample_space = data_pairs_2021
    data_pair_train_samples= []
    for expo in range(6,floor(log(len(sample_space),2))+1):
        data_keys_sample = choice(list(sample_space.keys()),2**expo,replace=False)
        data_pair_train_samples.append((2**expo,include2020,  {key : sample_space[key]  for key in data_keys_sample }))
    if (len(sample_space) - floor(log(len(sample_space),2)) )/len(sample_space)>0.1: #if there is significant difference in taking total set, then take also total set
        data_pair_train_samples.append((len(sample_space),include2020,sample_space))
    return data_pair_train_samples




def initialize_cfg_and_register(trial_id,sample,output_dir = None,):
        '''
        setup base configuration of model SEE MORE AT https://detectron2.readthedocs.io/en/latest/modules/config.html
        '''
        if sample[1]:
            year_str = "w20_"
        else:
            year_str = ""
        data_name = f'filet_{year_str}{sample[0]}_train'
        try:
            DatasetCatalog.register(data_name,lambda : get_data_dicts(data_dir,'train',sample[2]))
            MetadataCatalog.get(data_name).set(thing_classes = ['filet'])
        except AssertionError as e:
            print(e)
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(f'{model_name}.yaml'))
        cfg.DATASETS.TRAIN = (data_name,)
        cfg.DATASETS.TEST = ('filet_val',) # Use this with trainer_cls : TrainerPeriodicEval if you want to do validation after every iterations
        cfg.TEST.EVAL_PERIOD = 500
        cfg.DATALOADER.NUM_WORKERS = 6 #add more workerss until it gives warnings.
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f'{model_name}.yaml')
        cfg.SOLVER.IMS_PER_BATCH = 3 #maybe more?
        if output_dir is None:
            cfg.OUTPUT_DIR = base_output_dir
        else:
            cfg.OUTPUT_DIR = output_dir
        cfg.SOLVER.STEPS =(5000,25000)
        cfg.SOLVER.GAMMA = 0.5
        os.makedirs(cfg.OUTPUT_DIR,exist_ok=True)
        cfg.SOLVER.MAX_ITER = 10**5
        cfg.SOLVER.BASE_LR = 0.00025
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  #(default: 512)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        return cfg


def TOY_cfg_and_register(trial_id,sample,output_dir = None,):
        '''
        setup base configuration of model SEE MORE AT https://detectron2.readthedocs.io/en/latest/modules/config.html
        '''
        if sample[1]:
            year_str = "w20_"
        else:
            year_str = ""
        data_name = f'filet_{year_str}{sample[0]}_train'
        try:
            DatasetCatalog.register(data_name,lambda : get_data_dicts(data_dir,'train',sample[2]))
            MetadataCatalog.get(data_name).set(thing_classes = ['filet'])
        except AssertionError as e:
            print(e)
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(f'{model_name}.yaml'))
        cfg.DATASETS.TRAIN = (data_name,)
        cfg.DATASETS.TEST = ('filet_val',) # Use this with trainer_cls : TrainerPeriodicEval if you want to do validation after every iterations
        cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
        cfg.TEST.EVAL_PERIOD = 0
        cfg.SOLVER.BASE_LR = 0.00025
        cfg.TEST.EVAL_PERIOD = 50
        cfg.SOLVER.STEPS =(21,41,)
#        cfg.SOLVER.GAMMA = cfg.SOLVER.BASE_LR/2
        cfg.SOLVER.WARMUP_ITERS = 1
        cfg.DATALOADER.NUM_WORKERS = 6 #add more workerss until it gives warnings.
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f'{model_name}.yaml')
        cfg.SOLVER.IMS_PER_BATCH = 3 #maybe more?
        if output_dir is None:
            cfg.OUTPUT_DIR = base_output_dir
        else:
            cfg.OUTPUT_DIR = output_dir
        os.makedirs(cfg.OUTPUT_DIR,exist_ok=True)
        cfg.SOLVER.MAX_ITER = 10**5
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  #(default: 512)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        return cfg


class Trainer(TrainerPeriodicEval):
    '''
    Example of a trainer that applies argumentations at runtime. Argumentations available can be found here:
    https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html
    '''
    def __init__(self,augmentations,cfg):
        self.period_between_evals = cfg.TEST.EVAL_PERIOD
        super().__init__(cfg)
        self.augmentations=augmentations


    #overwrites default build_train_loader
    @classmethod
    def build_train_loader(cls, cfg):
          mapper = DatasetMapper(cfg, is_train=True, augmentations=augmentations)
          return build_detection_train_loader(cfg,mapper=mapper)


    def build_hooks(self):
        res = super().build_hooks()
        hook = hooks.StopByProgressHook(patience=10*self.period_between_evals,delta_improvement=0.5,score_storage_key='segm/AP',save_name_base="best_model")
        #hook = hooks.StopByProgressHook(patience=2*self.period_between_evals,delta_improvement=0,score_storage_key='segm/AP',save_name_base="best_model")
        res.append(hook)
        return res

augmentations = [
          T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
          T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
          T.RandomRotation(angle = [-10,10], expand=True, center=None, sample_style='range'),
          T.RandomBrightness(0.9,1.1)
              ]




def experiment():
    result_df = pd.DataFrame(columns=['w20', 'size', 'AP', 'iter'])
    result_df.astype({'w20': 'bool', 'size': 'int32', 'AP': 'float32'})
    date = datetime.date.today()
    time = datetime.datetime.now()
    try:
        for trial_id in range(10**6):
            sample_batch = generate_train_sets(True)
            sample_batch.extend(generate_train_sets(False))
            for sample in sample_batch:
                print("---------------------------------------\n",
                      'RUNNING EXPERIMENT WITH',sample[0],sample[1],'TRIAL_ID IS ',trial_id,
                      "\n---------------------------------------------")
                #cfg = TOY_cfg_and_register(trial_id,sample)
                cfg = initialize_cfg_and_register(trial_id,sample)
                trainer = Trainer(augmentations,cfg)
                trainer.resume_or_load(resume=False)
                try:
                    trainer.train()
                except hooks.StopFakeExc:
                    ap,iter = trainer.info_at_stop
                else:
                    ap,iter = trainer.storage.latest()['segm/AP']
                result = {'w20': sample[1],'size':str(sample[0]),'AP': ap, 'iter' : iter }

                result_df = result_df.append(result,ignore_index=True)
                with open(f'{base_output_dir}/results', 'a+') as f:
                    json.dump(result, f)
                    f.write(os.linesep)
                agg= result_df.groupby(['w20','size']).agg({'AP': ['mean', 'std']})
                t = torch.cuda.get_device_properties(0).total_memory //(10**6)
                r = torch.cuda.memory_reserved(0) //(10**6)
                a = torch.cuda.memory_allocated(0) //(10**6)
                f = (r - a)  # free inside reserved
                DatasetCatalog.remove(cfg.DATASETS.TRAIN[0])
                MetadataCatalog.remove(cfg.DATASETS.TRAIN[0])
                print("---------------------------------------\n",
                      agg,
                      "\n---------------------------------------------")
                titles = ['TOTAL', 'RESERVED','ALLOCATED','FREE INSIDE RESERVED']
                vals = [t,r,a,f]
                strs = []

                for title,val in zip(titles,vals):
                    strs.append(f'{title}:\t,{val}')
                print( "\n".join(strs) )
    except Exception as e:
        print(e)
    finally:
        time_info_str = "-".join([str(x) for x in [date.year,date.month,date.day,time.hour,time.minute]])
        result_df.to_csv(f'{base_output_dir}/results_pd-{time_info_str}.csv')
        agg.to_csv(f'{base_output_dir}/agg_pd-{time_info_str}.csv')
experiment()

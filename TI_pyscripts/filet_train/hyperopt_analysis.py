import gc
import os

import torch.cuda
from detectron2.config import get_cfg
import numpy as np
import pandas as pd
from trainers import TI_Trainer
from detectron2.data import build_detection_train_loader, DatasetMapper
from detectron2.evaluation import COCOEvaluator
import hooks
import detectron2.data.transforms as T
from data_utils import get_data_dicts,register_data


pd.set_option('display.max_columns', None)
data_dir = "/pers_files/Combined_final/Filet"
input_dir1 = f'{data_dir}/output'
input_dir2 = f'{data_dir}/output2'
model_name_base = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x"
output_dir = os.path.join(data_dir,'output_train')
os.makedirs(output_dir,exist_ok=True)
splits = ['train','val']
data_dir = "/pers_files/Combined_final/Filet"
COCO_dicts = {split: get_data_dicts(data_dir,split) for split in splits } #converting TI-annotation of pictures to COCO annotations.
data_names = register_data('filet',['train','val'],COCO_dicts,{'thing_classes' : ['filet']}) #register data by str name in D2 api
model_name = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x"

dfs = []
for i, input_dir in enumerate([input_dir1,input_dir2]):
    files = os.listdir(input_dir)
    for str in files:
        if str.endswith(".csv"):
            df = pd.read_csv(os.path.join(input_dir,str))
            df['round']= i
            dfs.append(df)
df = pd.concat(dfs,axis=0,ignore_index=False)
df = df.iloc[:,1:]
print()
df_alive =df.loc[np.logical_not(df['pruned']),:]
df_alive = df_alive.sort_values(by="score")
print(df_alive)
cfgs = []
df_alive['score_final'] = 0

df_alive.to_csv(os.path.join(output_dir,'pd_result_pre.csv'))

class Trainer(TI_Trainer):
    '''
    Example of a trainer that applies argumentations at runtime. Argumentations available can be found here:
    https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html
    '''
    def __init__(self,augmentations,**params_to_Trainer):
        self.augmentations=augmentations
        self.period_between_evals = cfg.TEST.EVAL_PERIOD
        self.top_score_achieved = 0
        super().__init__(**params_to_Trainer)
    #overwrites default build_train_loader
    @classmethod
    def build_train_loader(cls, cfg):
          mapper = DatasetMapper(cfg, is_train=True, augmentations=augmentations)
          return build_detection_train_loader(cfg,mapper=mapper)

    @classmethod
    def build_evaluator(cls,cfg,dataset_name,output_folder=None):
        if output_folder is None:
            output_folder = cfg.OUTPUT_DIR
        return COCOEvaluator(dataset_name, ('bbox', 'segm'), False, output_dir=output_folder)


    def build_hooks(self):
        res = super().build_hooks()

        res.append(hooks.StopByProgressHook(patience=25*self.period_between_evals,delta_improvement=0.5,score_storage_key='segm/AP',save_name_base="best_model"))
        return res

    def helper_after_train(self,**kwargs):
        self.top_score_achieved = self.storage.latest()[f'best_segm/AP']

    def handle_stop(self,**kwargs):
        self.helper_after_train()

    def handle_else(self,**kwargs):
        self.helper_after_train()

augmentations = [
          T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
          T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
          T.RandomBrightness(0.9,1.1),
          T.RandomExtent([0.9, 1], [0, 0]),
          T.RandomRotation(angle=[-10, 10], expand=False, center=None, sample_style='range'),
          T.Resize((1024,1024)),

]


def do_train(augmentations,cfg):
    trainer = Trainer(augmentations=augmentations, cfg=cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()
    return trainer.top_score_achieved[0]

for i in range(1,df_alive.shape[0]):
    model_specs = df_alive.iloc[i,:]
    output_dir = input_dir1 if model_specs['round']== 0 else input_dir2
    model_name = f"{model_name_base}_{model_specs.name}_output"
    cfg_dir =  os.path.join(data_dir,output_dir,'trials',model_name,'cfg.yaml')
    cfg = get_cfg()
    cfg.merge_from_file(cfg_dir)
    cfg.SOLVER.STEPS = (6000,12000,18000,24000,30000,)
    cfg.SOLVER.GAMMA = 0.75
    cfg.TEST.EVAL_PERIOD = 250
    cfg.DATASETS.TEST = ('filet_val',)
    cfgs.append(cfg)
    df_alive.iloc[i,-1] = do_train(augmentations,cfg)
    torch.cuda.empty_cache()
    gc.collect()

df_alive.to_csv(os.path.join(output_dir,'pd_result.csv'))

print(df_alive)

#--------------------------------------------

# install dependencies:
import torch, torchvision

import hooks

assert torch.cuda.is_available(), "torch cant find cuda. Is there GPU on the machine?"
# opencv is pre-installed on colab
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from collections.abc import Mapping
from detectron2.engine import DefaultTrainer
from detectron2.data import  transforms as T
import matplotlib.pyplot as plt
from copy import deepcopy
import json
from contextlib import contextmanager
from hooks import StopAtIterHook
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
setup_logger()
import numpy as np
import os, json, cv2, random, copy
from pruners import SHA
import os
from detectron2.structures.boxes import BoxMode


def get_file_pairs(data_dir,split):
    '''
    input: data directory, and split ( train val test).
    output: dict of pairs of files one image, and one json file.
    '''
    data_dict = {}
    file_to_pair = os.listdir(os.path.join(data_dir,split))
    while(file_to_pair):
        name = file_to_pair.pop()
        front = name.split(".")[0]
        if front in data_dict.keys():
            data_dict[front].append(name)
        else:
            data_dict[front] = [name]

#drop those where there is not both json and jpg:
    to_drop = []
    for key,data in data_dict.items():
        if not len(data) >=2:
            to_drop.append(key)
    for key in to_drop:
        data_dict.pop(key)
        print("dropping data", key, " due to missing data")
    return data_dict



#TODO:: change name to something like "ti_format_to_coco_format"

def get_data_dicts(data_dir,split,file_pairs = None):
    '''
    input:
    list(str) each str refers to a split of dataset. example : split = ['train','val','test']
    If file_pairs is none, file_pairs will be created all files in each data_dir split.
    output: dict withCOCO compliant structure, ready to be registered as a dataset in detectron2
    '''
    if file_pairs is None:
        file_pairs= get_file_pairs(data_dir,split)
    dataset_dicts = []
    data_dir_cur = os.path.join(data_dir,split)
    assert all(len(files)==2 for files in file_pairs.values() )
    img_id = 0
    for idx,tup in enumerate(file_pairs.items()):
        name, files = tup
        files.sort()
        jpg_name,json_name = files
        #print(jpg_name)
        #print(cv2.imread(data_dir_cur))
        height, width = cv2.imread(os.path.join(data_dir_cur,jpg_name)).shape[:2]
        record = {}
        record["file_name"] = os.path.join(data_dir_cur,jpg_name)
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        json_file = os.path.join(data_dir_cur,json_name)
        with open(json_file) as f:
            imgs_anns = json.load(f)
        try:
            #try to access
            test_poly = imgs_anns['shapes'][0]['points']
        except(IndexError):
            print('did not load ',jpg_name,'due to missing/wrong annotations')
        else:
            objs = []
            for shape in imgs_anns['shapes']:
                poly = shape['points']
                xs = [point[0] for point in poly]
                ys = [point[1] for point in poly]

                poly_flatten = []
                for xy in poly:
                    poly_flatten.extend(xy)

                obj = {
                            "bbox": [np.min(xs), np.min(ys), np.max(xs), np.max(ys)],
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "segmentation": [poly_flatten],
                            "category_id": 0,
                        }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts

def register_data(prefix_name,splits,COCO_dicts,metadata):
    '''
    "TODO:write smt
    '''
    names_result = {}
    for d in splits:
        name = f"{prefix_name}_{d}"
        names_result[d] = name
        try:
            DatasetCatalog.register(name, lambda d=d: COCO_dicts[d])
            MetadataCatalog.get(name).set(**metadata)
        except AssertionError:
            print(f"{name} is already registered.")
    return names_result



#TODO::write more about evaluator
#TODO:: Change name super_step_size to smt like iter_chunk
#TODO:: Change max_epoch to max_iter
class D2_hyperopt_Base():
    '''
    does hyper-optimization for detectron2 models

    input:
      model_dict: dict with 3 keys: (str) model_name, (cfg)base_cfg and (dict) hp_dict.
        hp dict: a dict with same structure as cfg. If key is should be treated as hyper_parameters then
        value should be a dict {type1 : name, sample_params} where:
            -'type1' is a str categorizing the type of sampling (see suggest_values)
            -'name' is a str giving name of hyper-parameters
            -sample_params are kwargs to sampler.
        if key should not be treated as hyper_param, leave it out of hp_dict entirely.
      task: Possible choices are at time of writing "bbox", "segm", "keypoints".
      evaluator: Use COCOEvaluator if in doubt
      https://detectron2.readthedocs.io/en/latest/modules/evaluation.html#detectron2.evaluation.COCOEvaluator
      super_step_size: chunk of iters corresponding to 1 ressource in pruner
      output_dir : dir to output
      max_epoch : maximum TOTAL number of iters across all tried models. WARNING: Persistent memory needed is proportional to this.
      pruner_cls : class(not object) of a pruner. see pruner class
    '''
    def __init__(self, model_dict,data_names, task,evaluator,output_dir, step_chunk_size=30,
                 max_iter = 90,
                 trainer_cls = DefaultTrainer,
                 pruner_cls = SHA,
                 pr_params = {}):
        print("I HAVE STARTED")
        self.step_chunk_size = step_chunk_size
        self.model_name=model_dict['name']
        self.model_dict = model_dict
        self.task = task
        self.trainer_cls = trainer_cls
        self.suggested_cfgs = []
        self.data_names = data_names
        self.suggested_params = []
        self.output_dir=output_dir
        self.evaluator = evaluator
        self.pruner = pruner_cls(max_iter // self.step_chunk_size, **pr_params)
        class TrainerWithHook(trainer_cls):
            def __init__(self,trial_id,iter,*args,**kwargs):
                self.iter_to_stop = iter
                self.trial_id = trial_id
                super().__init__(*args,**kwargs)


            def build_hooks(self):
                res = super().build_hooks()
                print('sent',self.iter_to_stop,'to hook')
                hook = StopAtIterHook(f"{self.trial_id}_stop_at_{self.iter_to_stop}", self.iter_to_stop)
                res.append(hook)
                return res
        self.trainer_cls = TrainerWithHook

    # parameters end



    def initialize(self):
        raise NotImplementedError

    # TODO:complete with all types
    def suggest_values(self, typ, params):
        '''
        MEANT TO BE SUBCLASSED AND SHADOWED.
        input: (typ,params)
        structure:
        if/elif chain of if typ == "example_type":
            sample and return sample
        output: sample of structure corresponding to typ
        '''
        raise NotImplementedError

    def get_model_name(self,trial_id):
        return f'{self.model_name}_{trial_id}'

    def get_trial_output_dir(self,trial_id):
        return f'{self.output_dir}/trials/{self.get_model_name(trial_id)}_output'

    def load_from_cfg(self,cfg,res,trial_id=-1):
        '''
        load a model specified by cfg and train
        '''
        #cfg.SOLVER.MAX_ITER += self.super_step_size*res
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        print("THE OUTPUT DIRECTORY FOR TRIAL ID",trial_id,"IS ",cfg.OUTPUT_DIR)
        trainer = self.trainer_cls(trial_id=trial_id,iter=res * self.step_chunk_size,cfg=cfg)
        trainer.resume_or_load(resume= True)
        return trainer

    def validate(self,cfg_sg,trainer):
      '''
      takes a partially trained model and evaluate it
      '''
      cfg_sg_pred = cfg_sg.clone()
      cfg_sg_pred.MODEL.WEIGHTS = os.path.join(cfg_sg.OUTPUT_DIR, "model_final.pth")
      val_loader = build_detection_test_loader(cfg_sg_pred, self.data_names['val']) #ud af loop?
      infe = inference_on_dataset(trainer.model, val_loader, self.evaluator)
      print(infe.keys())
      val_to_report = infe[self.task]['AP']
      return val_to_report

    def suggest_cfg(self,trial_id):
        model_name = self.model_dict['name']
        cfg_sg = self.model_dict['base_cfg'].clone() #deep-copy
        dict_queue = [ ( [], self.model_dict['hp_dict'] ) ]  # first coordinate is parent-dict-keys
        suggested_params = {}
        # BFS of nested dicts:
        while (dict_queue):
            parents, current_dict = dict_queue.pop()
            sub_cfg=cfg_sg
        #        print("printing parents",parents)
            if parents:
                for parent in parents:
                    print("printing parents",parents)
                    sub_cfg = sub_cfg[parent]
            for key, value in current_dict.items():
                if isinstance(value, Mapping):
                    print("PARENTS WAS", parents)
                    print("parent to add",key)
                    new_parents = copy.deepcopy(parents)
                    new_parents.append(key)  # shallow is enough
                    print("parents is",new_parents)
                    dict_queue.append((new_parents, value))
                    print("DICT QUEUE IS",dict_queue)
                else:
                    #make assertions
                    typ,params = value
                    value = self.suggest_values(typ,params)
                    sub_cfg[key] = value
                    suggested_params[params['name']]  = value
                    cfg_sg.OUTPUT_DIR = self.get_trial_output_dir(trial_id)
        print("OUTPUTDIR FOR THIS", trial_id, "are in directory", cfg_sg.OUTPUT_DIR)
        self.suggested_cfgs.append(cfg_sg)
        self.suggested_params.append(suggested_params)
        for cfg in self.suggested_cfgs:
            print(cfg.OUTPUT_DIR)
        return cfg_sg , suggested_params



    def sprint(self,trial_id,res,cfg_sg):
        trainer = self.load_from_cfg(cfg_sg, res,trial_id)
        try:
            trainer.train()
        except hooks.StopFakeExc:
            print("Stopped per request of hook")
            val_to_report = self.validate(cfg_sg,trainer)
        except (FloatingPointError, ValueError):
            print("Bad_model")
            val_to_report = 0
        else:
            val_to_report = self.validate(cfg_sg,trainer)
        return val_to_report


    def start(self):
        for i in range(self.pruner.participants):
            suggested_cfg,params = self.suggest_cfg(i)
            self.initialize(suggested_cfg,params)
        id_cur = 0
        done = False
        while not done:
            print("NOW RUNNING ID,----------------------------------------------------------------------",id_cur)
            cfg = self.suggested_cfgs[id_cur]
            val_to_report = self.sprint(id_cur,self.pruner.get_cur_res(),cfg)
            id_cur, pruned, done = self.pruner.report_and_get_next_trial(val_to_report)
            self.prune_handling(pruned)
        return self.get_result()

    def prune_handling(self,pruned_ids):
        pass

    def get_result(self):
        return self.pruner.get_best_models()


class TrainerPeriodicEval(DefaultTrainer):
    """
    Completely like
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)
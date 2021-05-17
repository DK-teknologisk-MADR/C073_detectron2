import copy
import os
from typing import Mapping

from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import inference_on_dataset

import hooks
from hooks import StopAtIterHook
from pruners import SHA

# install dependencies:
import torch

assert torch.cuda.is_available(), "torch cant find cuda. Is there GPU on the machine?"
# opencv is pre-installed on colab
from detectron2.utils.logger import setup_logger

setup_logger()


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
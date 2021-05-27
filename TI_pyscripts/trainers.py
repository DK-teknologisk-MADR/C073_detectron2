from hooks import StopFakeExc
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
class TI_Trainer(DefaultTrainer):
    '''
    We should always use this trainer instead of defaulttrainer. Its exactly like it, except that it catches the StopFakeExc used for early stopping by hooks in hooks.py
    '''
    def __init__(self,cfg):
        super().__init__(cfg)

    def train(self,**kwargs):
        try:
            super().train()
        except StopFakeExc:
            self.handle_stop(**kwargs)
        else:
            self.handle_else(**kwargs)

    def handle_stop(self,**kwargs):
        pass

    def handle_else(self,**kwargs):
        pass

class TrainerPeriodicEval(TI_Trainer):
    """
    Trainer with coco-evaluator implemented. if cfg.DATASETS.TEST is filled and
    cfg.TEST.evalperiod>0 then it will evaluate periodically during training.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)
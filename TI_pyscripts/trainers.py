from hooks import StopFakeExc
from detectron2.engine import DefaultTrainer

class TI_Trainer(DefaultTrainer):
    def __init__(self,cfg):
        super().__init__(cfg)

    def trainer(self,**kwargs):
        try:
            super().train()
        except StopFakeExc:
            self.handle_stop(**kwargs)


    def handle_stop(self,**kwargs):
        pass


class TrainerPeriodicEval(TI_Trainer):
    """
    Completely like
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)
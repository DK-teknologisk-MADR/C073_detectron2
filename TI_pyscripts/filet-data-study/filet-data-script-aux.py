from detectron2.evaluation import COCOEvaluator, inference_on_dataset
class EarlyStopByProgressMixin():
    def __init__(self):
        self.score_best = float('-inf')
        self.score_milestone = float('-inf')
        self.info_best = 0
        self.iter_best = 0
        self.iter_milestone = 0
        self.patience = 10
        self.delta_improvement = 1

    def report_score(self, score_cur,iter , info=None):
        if self.score_best < score_cur:
            self.score_best, self.iter_best, self.info_best = score_cur, iter, info
            # self.save_and_stuff()
            if self.score_milestone < score_cur - self.delta_improvement:
                self.iter_milestone, self.score_milestone = iter, score_cur
        if self.iter_milestone < iter - self.patience:
            done = True
        else:
            done = False
        return done

    def __str__(self):
        return f'best score:\t{self.score_best}\nbest iter:\t{self.iter_best}\nmilestone score:\t{self.score_milestone}\nmilestone iter:\t{self.score_milestone}\n'

        #class EarlyStopHook(EarlyStopHookBase):

ap,iter = trainer.storage.latest()['segm/AP']
class EvalMixin():

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)

ap,iter = trainer.storage.latest()['segm/AP']
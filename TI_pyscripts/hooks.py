from detectron2.engine.hooks import HookBase
import sys
from contextlib import contextmanager


@contextmanager
def disable_exception_traceback():
    """
    All traceback information is suppressed and only the exception type and value are printed
    """
    default_value = getattr(sys, "tracebacklimit", 1000)  # `1000` is a Python's default value
    sys.tracebacklimit = 0
    yield
    sys.tracebacklimit = default_value  # revert changes

class save_hook(HookBase):
    def after_step(self):
        if self.trainer.iter % 500== 0:
            self.trainer.checkpointer.save(f"model_save_{self.trainer.iter}",iteration = self.trainer.iter)



class StopFakeExc(Exception):
    '''
    This exception is used for hooks to stop training early due to some condition. Use TI_trainer from trainers.py in combination.
    '''
    pass


#implement early stopping
#very hackish, detectron2 can't stop training, but official suggestion is to implement via raised exception.
 # lives in trainer

class EarlyStopHookBase(HookBase):
    def __init__(self,save_name_base):
        self.save_name = save_name_base
        self.did_save = True


    def before_step(self):
        self.did_save = False

    def before_stop(self):
        pass

    def stopping_criteria(self):
      '''
      output: (bool) should be true if this is last stop.
      '''
      raise NotImplementedError

    def save(self):
        if not self.did_save:
            self.trainer.checkpointer.save(self.get_save_name(),iteration = self.trainer.iter)
            self.did_save = True

    def get_save_name(self):
        '''
        overwrite this if not satisfied
        '''
        return f"{self.save_name}_{self.trainer.iter}"


    def _handle_stop(self):
        with disable_exception_traceback():
            raise StopFakeExc('PLANNED STOP. Nothing to worry about here')

    def after_step(self):
        if self.stopping_criteria():
            self.before_stop()
            if not self.did_save:
                self.save()
            self._handle_stop()

class StopAtIterHook(EarlyStopHookBase):
    def __init__(self,save_base_name,iter_to_stop):
        super().__init__(save_name_base=save_base_name)
        self.iter_to_stop = iter_to_stop
        self.iter = 0

    def after_step(self):
        self.iter += 1
        super().after_step()

    def stopping_criteria(self):
        return self.iter_to_stop <= self.iter


class StopByProgressHook(EarlyStopHookBase):
    '''
    Classic early stop by tracking progress. Assumes that score can be obtained as (score,key) value from key "storage_key" from trainer storage.
    '''
    def __init__(self,patience,delta_improvement,score_storage_key,save_name_base):
        super().__init__(save_name_base=save_name_base)
        self.should_stop = False
        self.score_best = float('-inf')
        self.score_milestone = float('-inf')
        self.info_best = 0
        self.iter_best = 0
        self.iter_milestone = 0
        self.patience = patience
        self.delta_improvement = delta_improvement
        self.score_storage_key = score_storage_key
        self.remaining_patience = float('inf')

    def report_score_from_storage(self, score_cur,iter, info=None):
        if self.score_best < score_cur:
            self.score_best, self.iter_best, self.info_best = score_cur, iter, info
            self.save()
            if self.score_milestone < score_cur - self.delta_improvement:
                self.iter_milestone, self.score_milestone = iter, score_cur
            print(self.__str__())
        self.remaining_patience = (self.patience- (iter- self.iter_milestone) )
        if self.remaining_patience > 0:
            self.should_stop = True
        else:
            self.should_stop = False

    def before_train(self):
        super().before_train()
        self.trainer.storage.put_scalar('segm/AP',0,False)

    def get_save_name(self):
        return f"{self.save_name}"

    def before_stop(self):
        self.trainer.info_at_stop = (self.score_best,self.iter_best)

    def stopping_criteria(self):
        return self.should_stop

    def after_step(self):
        score,iter = self.trainer.storage.latest()[self.score_storage_key]
        self.report_score_from_storage(score,iter)
        super().after_step()


    def __str__(self):
        return f'best score:\t{self.score_best}\nbest iter:\t{self.iter_best}\nmilestone score:\t{self.score_milestone}\nmilestone iter:\t{self.iter_milestone}\nRemaining Patience:\t{self.remaining_patience}'

        #class EarlyStopHook(EarlyStopHookBase):


#ap,iter = trainer.storage.latest()['segm/AP']


#class EarlyStopHook(EarlyStopHookBase):



##early_stop_trainer(DefaultTrainer)


























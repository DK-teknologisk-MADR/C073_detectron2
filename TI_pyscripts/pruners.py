from math import log,floor, ceil
import numpy as np
from scipy.special import lambertw

class pruner_base():
    '''
    Requirement of any pruner:
    input: Max_res
    attributes: participants (max)
    '''
    def report_and_get_next_trial(self,reported_val):
        '''
        input: reported_val score to book-keep for current running trial.
        output: 3-Tuple of
        (int)next_trial to run
        list(int) list of trials that should be pruned
        (bool) whether we are finished or not.
        '''
        raise NotImplementedError

    def get_cur_res(self):
        raise NotImplementedError

    def get_best_models(self):
        raise NotImplementedError
class SHA(pruner_base):
    '''
    Given max_res, a number of participants will be initialized which of the form factor^(exp) for some exp integer.
    participant 0 will train for 1 ressource (e.g. iter chunks or time-period), and then a score is reported through .report_and_get_next_trial(),
    and a new participant is returned to trial. This will keep going until all has trained for 1 ressource, at which all but 1/factor will be eliminated (pruned), and the first "rung" is completed
    Then, the rung starts, but with factor*(previous ressource) are allocated to each trial.
    The procedure stops when topK models remains standing.

    '''
    def __init__(self, max_res, factor=3, topK=1):
        self.max_res = max_res
        self.factor = factor
        self.rungs_to_skip = ceil(1 + log(topK, self.factor))
        self.rungs = floor(self.compute_rungs())
        self.participants = floor(pow(factor, self.rungs - 1))
        self.rung_results = [[j, float('-inf')] for j in range(pow(self.factor, self.rungs - 1))]
        self.rung_cur = 0
        self.trial_in_rung_cur = 0
        self.trial_id_cur = 0
        self.max_trials_in_rung_cur = len(self.rung_results)
        self.done = False
        print(
            f"----------------------\n SHA INITIALIZED \n NUMBER OF TRIALS: {self.participants}\n RUNGS TO COMPLETE: {self.rungs - self.rungs_to_skip}\n ----------------------")

    def compute_rungs(self):
        return np.real(lambertw(log(self.factor) * self.max_res / pow(self.factor, self.rungs_to_skip - 1))) / np.log(
            self.factor) + self.rungs_to_skip

    def get_cur_res(self):
        return int(pow(self.factor, self.rung_cur))

    def report_and_get_next_trial(self, reported_val):
        if self.done:
            raise ValueError("sha already completed")
        self.rung_results[self.trial_in_rung_cur][1] = reported_val
        self.trial_in_rung_cur += 1
        pruned = []
        if self.trial_in_rung_cur == self.max_trials_in_rung_cur:
            self.max_trials_in_rung_next = self.max_trials_in_rung_cur // self.factor
            #            next_rung = self.rung_cur +1
            #            self.next_rung_size = self.max_trials_in_rung_cur // self.factor
            self.rung_results.sort(reverse=True, key=lambda tup: tup[1])
            self.rung_results, pruned = self.rung_results[:self.max_trials_in_rung_next], self.rung_results[
                                                                                          self.max_trials_in_rung_next:]
            #            for winner_nr in range(self.max_trials_in_rung_cur // self.factor):
            #               self.rung_results[self.rung_cur + 1][winner_nr] = self.rung_results[self.rung_cur][winner_nr]
            #            pruned = self.rung_results[self.rung_cur][self.max_trials_in_rung_cur // self.factor :]
            pruned = [tup[0] for tup in pruned]
            #        print(pruned)
            self.rung_cur += 1
            self.max_trials_in_rung_cur = self.max_trials_in_rung_next
            self.trial_in_rung_cur = 0
            if self.rung_cur >= self.rungs - self.rungs_to_skip:
                self.done = True
        self.trial_id_cur = self.rung_results[self.trial_in_rung_cur][0]
        return self.trial_id_cur, pruned, self.done

    def get_best_models(self):
        return self.rung_results

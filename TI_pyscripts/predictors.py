class TI_Predictor_Base():
    def __init__(self,cfg_fp,chk_fp=None,predictor_cls = DefaultPredictor):
        '''
        Inputs:
        cfg_fp : file path of cfg
        chk_fp: file path of checkpoint (a .pkl file). Defaults to None, in which case the first .pkl file found in wd will be used.
        ----
        You'll have a folder with a
        - cfg for the model, encoded in a .yaml file.
        - a checkpoint file  (.pkl)
        Simply set cfg_fp as the string equal to the .yaml file, and if there are several .pkls,
        set chk_fp equal to the file of the desired checkpoint.
        '''
        self.cfg = get_cfg()
        self.cfg.merge_from_file(cfg_fp)
        if chk_fp is None:

            pattern = '.+\.pkl$'
            regex_pkl = re.compile(pattern,re.MULTILINE)
            result = regex_pkl.search("\n".join(os.listdir()))
            if result:
                chk_fp = f"./{result.group()}"
            else:
              raise ValueError("no checkpoint given, and no checkpoint found in wd")
        self.cfg.MODEL.WEIGHTS = chk_fp
        self.predictor = predictor_cls(self.cfg)


    def __call__(self, inputs):
        pred_outputs = self.predictor(inputs)
        return self.post_process(pred_outputs)

    def post_process(self,pred_outputs):
        raise NotImplementedError
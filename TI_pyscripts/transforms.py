import detectron2.data.transforms as T

class CropAndResize(T.Augmentation):
    def __init__(self, scale_range,shift_range=[0,0]):
        self.scale_range = scale_range
        self.shift_range = shift_range

    def __call__(self,aug_input):
        oldx,oldy,oldc = aug_input.image.shape
        scaler = T.RandomExtent(self.scale_range,self.shift_range)(aug_input)
        resizer = T.Resize((oldx,oldy))(aug_input)
        return T.TransformList([scaler,resizer])
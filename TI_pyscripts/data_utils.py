import json
import os

import cv2
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode


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


def get_data_dicts(data_dir,split,file_pairs = None):
    '''
    input:
    list(str) each str refers to a split of dataset. example : split = ['train','val','test']
    If file_pairs is none, file_pairs will be created all files in each data_dir split.
    output: dict withCOCO compliant structure, ready to be registered as a dataset in detectron2
    '''
    if file_pairs is None:
        file_pairs= get_file_pairs(data_dir, split)
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
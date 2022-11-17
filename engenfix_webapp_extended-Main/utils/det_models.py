# import some common libraries
import numpy as np

from detectron2.engine import DefaultPredictor # consists of 2 engines one for train and other for prediction
from detectron2.config import get_cfg


class Det2seg():

    def __init__(self, yaml_file, pth_file, score_thresh, device, num_class):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(yaml_file)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        self.cfg.MODEL.WEIGHTS = pth_file
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_class
        self.cfg.MODEL.DEVICE = device #use without the gpu where cpu only
        self.predictor = DefaultPredictor(self.cfg)


    def get_preds(self, image, dup_img, color_map):
        # im = cv2.imread(image)
        outputs = self.predictor(image)
        mask_array = outputs['instances'].pred_masks.numpy()
        dmg_classes = outputs["instances"].pred_classes
        # for mask_array.shape[]
        num_instances = mask_array.shape[0]
        mask_array = np.moveaxis(mask_array, 0, -1)
        mask_array_instance = []
        last_drawn_img = []
        dmg_count = len(dmg_classes)

        for i in range(num_instances):
            mask_array_instance.append(mask_array[:, :, i:(i+1)])
        # print('mask_array_instance', mask_array_instance)
            dup_img = np.where(mask_array_instance[i] == True, color_map, dup_img)
            last_drawn_img.append(dup_img)
        
        if len(last_drawn_img) > 0:
            return last_drawn_img[-1], dmg_count
        else: 
            return dup_img, dmg_count




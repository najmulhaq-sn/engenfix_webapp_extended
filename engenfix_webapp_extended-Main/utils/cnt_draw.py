import numpy as np
import os, cv2

def create_dup_img(file_path):
    img = cv2.imread(file_path)
    dup_img = np.zeros_like(img)
    dup_img.fill(80)

    return img, dup_img

def draw_segments(pred_img, ori_img, out_path, file_name):
    pred_img = np.asarray(pred_img, np.float64)
    ori_img = np.asarray(ori_img, np.float64)
    img_final = cv2.addWeighted(ori_img, 0.5, pred_img, 0.5, 0)
    img_proc_saved_path = os.path.join(out_path,file_name)
    cv2.imwrite(img_proc_saved_path, img_final)

    return img_proc_saved_path


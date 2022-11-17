from utils.det_models import Det2seg
from utils.cnt_draw import create_dup_img, draw_segments
import cv2

model_versions = 'model/ver2/'
IMAGEDIR_out = "assets/output/"

def main(file_save_path):
    cfg_glass= Det2seg(model_versions+'glass_model.yaml', model_versions+'glass_model.pth' , 0.5,'cpu', 2)
    cfg_broken= Det2seg(model_versions+'broken_model.yaml', model_versions+'broken_model.pth' , 0.5,'cpu', 2)
    img_original, img_duplicate = create_dup_img(file_save_path)
    # do model preds
    img_pred_drawn, dmg_count = cfg_broken.get_preds(img_original, img_duplicate, (0,0,254))
    print('dmg_count_broken', dmg_count)
    img_pred_drawn, dmg_count = cfg_glass.get_preds(img_original, img_pred_drawn, (254,0,0))
    print('dmg_count_glass', dmg_count)
    # img_pred_drawn = cfg_dent.get_preds(img_original, img_pred_drawn, cfg_broken['color_code'])
    # img_pred_drawn = cfg_scratch.get_preds(img_original, img_pred_drawn, cfg_broken['color_code'])
    cv2.imwrite('assets/output/test_unit14.jpg', img_pred_drawn)

    # overlay final image
    proc_out_img_path = draw_segments(img_pred_drawn, img_original,IMAGEDIR_out, 'sample14.jpg')


    return proc_out_img_path


if __name__ == '__main__':
    out_thing= main('assets/input/lvl1_val_146.jpg')
    print(out_thing)
# from utils.preds_classifiers import image_prep
from utils.preds_classifiers import image_prep
from utils.anpr import image_preprocess
from utils.vehicle_validation import valid_vehicle


from multiprocessing import allow_connection_pickling
from re import template
from tkinter.tix import Form
from urllib.request import Request
from utils.det_models import Det2seg
from utils.cnt_draw import create_dup_img, draw_segments

from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi import FastAPI, Request, Form, Depends, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi import APIRouter
import os
from pathlib import Path
from fastapi.responses import FileResponse
from random import randint
import uuid
import uvicorn # ASGI 
from fastapi import FastAPI, File, UploadFile


# classifier related libraries
import os
# import pickle
# import PIL
# from PIL import Image
import numpy as np
# import pandas as pd
# import sklearn
# from glob import glob
# import tensorflow as tf
# from tensorflow import keras
# import tensorflow as tf
# import tensorflow
import tensorflow as tf
# from tf import keras
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img



IMAGEDIR_in = "assets/input/"
IMAGEDIR_out = "assets/output/"

# detector models
broken_model = {'config':'model/ver2/broken_model.yaml' , 'weights': 'model/ver2/broken_model.pth', 'threshold': 0.65, 'device':'cpu' , 'no_classes':2 , 'color_code': (15,153,10)}
scratch_model = {'config':'model/ver2/scratch_model.yaml' , 'weights': 'model/ver2/scratch_model.pth', 'threshold': 0.75, 'device':'cpu' , 'no_classes':1 , 'color_code': (255, 78, 164)}
glass_model = {'config':'model/ver2/glass_model.yaml' , 'weights': 'model/ver2/glass_model.pth', 'threshold': 0.65, 'device':'cpu' , 'no_classes':2 , 'color_code': (35, 201, 209)}
dent_model = {'config':'model/ver2/dent_model.yaml' , 'weights': 'model/ver2/dent_model.pth', 'threshold': 0.75, 'device':'cpu' , 'no_classes':1 , 'color_code':  (251, 166, 230)}


#declare the models 
cfg_broken = Det2seg(broken_model['config'], broken_model['weights'], broken_model['threshold'], broken_model['device'], broken_model['no_classes'])
cfg_scratch = Det2seg(scratch_model['config'], scratch_model['weights'], scratch_model['threshold'], scratch_model['device'], scratch_model['no_classes'])
cfg_glass = Det2seg(glass_model['config'], glass_model['weights'], glass_model['threshold'], glass_model['device'], glass_model['no_classes'])
cfg_dent = Det2seg(dent_model['config'], dent_model['weights'], dent_model['threshold'], dent_model['device'], dent_model['no_classes'])


app = FastAPI()

# router = APIRouter()

app.mount("/static", StaticFiles(directory=Path(__file__).parent.parent.absolute() / "static"), name="static",)
app.mount("/assets/output", StaticFiles(directory="assets/output"), name="assets")



templates = Jinja2Templates(directory='templates')

# index route
@app.get("/")
def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.post("/submit")
async def get_damage_detection(request:Request, file : UploadFile = File(...)):

    # print(assignment)
    # content = await assignment_file.read()
    # print(content)


    # file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()  # <-- Important!
    file_save_path = os.path.join(IMAGEDIR_in, file.filename)
    # example of how you can save the file
    with open(f"{file_save_path}", "wb") as f:
        f.write(contents)

    # do the image segmentation
    # print(file_save_path)

    # do the image magic here
    # get copy of original image
    img_original, img_duplicate = create_dup_img(file_save_path)

    #validation of the image from the vehicle validator

    vehicle_image_is_valid = valid_vehicle(file_save_path)


    #classification model predictions 
    classification_report = image_prep(file_save_path)


    #ANPR for the front or back image only
    if classification_report['img_ori'] == 'Front' or classification_report['img_ori'] == 'Back':
        vehicle_numberplate = image_preprocess(file_save_path)
    else: 
        vehicle_numberplate = 'Not Available'



    # do model preds
    img_pred_drawn,dmg_count_broken = cfg_broken.get_preds(img_original, img_duplicate, tuple(broken_model['color_code']))
    img_pred_drawn, dmg_count_glass = cfg_glass.get_preds(img_original, img_pred_drawn, tuple(glass_model['color_code']))
    img_pred_drawn,dmg_count_dent= cfg_dent.get_preds(img_original, img_pred_drawn, tuple(dent_model['color_code']))
    img_pred_drawn, dmg_count_scratch = cfg_scratch.get_preds(img_original, img_pred_drawn,  tuple(scratch_model['color_code']))
    # img_pred_drawn, dmg_count_scratch = cfg_scratch.get_preds(img_original, img_duplicate,  tuple(scratch_model['color_code']))

    
    damages_description_json = {
        'Broken Damages Count': dmg_count_broken,
        'Glass Damages Count': dmg_count_glass,
        'Dent Damages Count': dmg_count_dent,
        'Scratch Damages Count': dmg_count_scratch
    }

    # overlay final image
    proc_out_img_path = draw_segments(img_pred_drawn, img_original,IMAGEDIR_out, file.filename)


    tot_dmgs = int(dmg_count_broken) + int(dmg_count_dent) + int(dmg_count_glass) + int(dmg_count_scratch)

    # return templates.TemplateResponse("portfolio-details.html", {"request": request, 'path': file_save_path})
    return templates.TemplateResponse("portfolio-details.html", {"request": request, 'path': proc_out_img_path, 'dent': damages_description_json['Dent Damages Count'],
    'scratch': damages_description_json['Scratch Damages Count'], 'broken': damages_description_json['Broken Damages Count'], 'shatter': damages_description_json['Glass Damages Count'], 'tot_dmgs':tot_dmgs ,
    'vehicle_orientation': classification_report['img_ori'], 'vehicle_damage_severity': classification_report['img_damage'] , 'vehicle_type': classification_report['img_type'], 'vehicle_numberplate': vehicle_numberplate,
    'isvalid': vehicle_image_is_valid})


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)

# uvicorn main:app --reload


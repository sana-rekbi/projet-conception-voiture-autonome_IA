

import os
import io
import numpy as np
import h5py
from tempfile import TemporaryFile
import tensorflow as tf
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Response

from google.cloud import storage
from google.oauth2 import service_account

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img




def colorize(mask):
    classes = ['void', 'flat', 'construction', 'object', 'nature', 'sky', 'human', 'vehicle']
    
    colors = {
        'void': (0,0,0), 
        'flat': (128,128,128), 
        'construction': (100,66,0), 
        'object': (200,200,0), 
        'nature': (0,200,0), 
        'sky': (0,0,200), 
        'human': (200,0,0), 
        'vehicle': (255,255,255), 
        }
    
    color_mask = np.zeros(mask.shape, dtype='int8')
    for n, class_name in enumerate(classes):
        class_mask = np.where(mask==n, colors[class_name], 0)
        color_mask = color_mask + class_mask
        
    return color_mask




def get_pred(img_pil, model, img_size, n_channels):
    img = img_to_array(img_pil) / 255
    x = np.reshape(img, (1, *img_size , n_channels))

    y_pred = model(x, training=False)
    y_pred = np.reshape(y_pred, y_pred.shape[1:])

    pred = np.argmax(y_pred, axis=-1)
    pred = np.expand_dims(pred, axis=-1)
    pred_color = colorize(pred)

    return pred_color




# variables
test_mode = False

img_size = (256, 512)
n_channels = 3

bucket_name = 'bucket-oc-8-api-model'
model_bucket = 'model.h5'
json_key_path = 'cle.json'


if test_mode:
    print('\nTEST MODE\n')
else:
    #get model from bucket
    print('\nAccessing model in cloud...')
    credentials = service_account.Credentials.from_service_account_file(json_key_path)
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(model_bucket)

    print('Loading model...')
    with TemporaryFile() as temp_file:
        blob.download_to_file(temp_file)
        f = h5py.File(temp_file)
        model = tf.keras.models.load_model(f)
    
    print('Model ready.\n')



# api
model_api = FastAPI()

@model_api.post('/')
async def predict(image: UploadFile = File()):
    content = await image.read()

    #img_flat = np.fromstring(content, np.uint8)
    #img = cv2.imdecode(img_flat, cv2.IMREAD_COLOR).astype(np.float32)
    img_pil = Image.open(io.BytesIO(content))
    img = img_to_array(img_pil)

    #img_pil = array_to_img(img)

    if test_mode:
        # temp pred
        pred = (np.random.rand(*img.shape)*255).astype(np.uint8)
        pred_pil = array_to_img(pred)
    else:
        # model pred
        pred = get_pred(img_pil, model, img_size, n_channels)
        pred_pil = array_to_img(pred)

    with io.BytesIO() as buf:
        pred_pil.save(buf, format='PNG')
        pred_pil_bytes = buf.getvalue()

    return Response(pred_pil_bytes, media_type='image/png')
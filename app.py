# Front end app to interact with the api deployed on Azure.


import os
import io

import requests
import numpy as np
import streamlit as st
from PIL import Image



def select_img_old(folder_path='.'):
    filenames = os.listdir(folder_path)
    imgs = [filename for filename in filenames if filename.startswith('image')]
    selected_filename = st.selectbox('Select a file', imgs)
    return folder_path + '/' + selected_filename



def select_img(folder_path='.'):
    filenames = os.listdir(folder_path)
    imgs = [filename for filename in filenames if filename.startswith('image')]
    masks = [filename for filename in filenames if filename.startswith('mask')]

    indexes = list(range(len(imgs)))
    index = st.selectbox('Select an image:', indexes)

    i = imgs[index][len('image_') : -len('.png')]
    selected_img = folder_path + '/' + imgs[index]
    selected_mask = folder_path + '/' + 'mask_' + i + '.png'

    return selected_img, selected_mask



def main():
    #url = 'http://127.0.0.1:8000/'    # local url
    url = 'https://oc-8-api.azurewebsites.net/'

    samples_dir = './samples'


    st.title('Image Segmentation')
    img_path, mask_path = select_img(folder_path=samples_dir)

    img_pil = Image.open(img_path)
    st.image(img_pil, caption='image')
    mask_pil = Image.open(mask_path)
    st.image(mask_pil, caption='mask')


    img_pil_bytes = open(img_path, 'rb')


    predict_btn = st.button('Predict')
    if predict_btn:
        files = {
            'image': img_pil_bytes, 
            }
        
        r = requests.post(url, files=files)

        try:
            pred_pil_bytes = io.BytesIO(r.content)
            pred_pil = Image.open(pred_pil_bytes)
            st.image(pred_pil, caption='predicton')
        except Exception as e:
            st.write(r)
            st.write('Raw json returned by api:')
            st.write(r.text)
            raise e
        


if __name__ == '__main__':
    main()
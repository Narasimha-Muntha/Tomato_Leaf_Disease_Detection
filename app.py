# %%writefile tomatoDL.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import cv2
#import matplotlib.pyplot as plt
#import pickle
from PIL import Image
st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('Image classifier Deep learning')
st.text('Upload the Image')
class_names=['Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']
reloaded_model = tf.keras.models.load_model('saved_model.h5')
uploaded_file = st.file_uploader("Choose an Image...",type='JPG')
if uploaded_file is not None:
    img=Image.open(uploaded_file)
    st.image(img,caption='Uploaded Image')
    
    if st.button('PREDICT'):
        st.write('Result..')
        #test_image=img.load_img(img,target_size=(256,256))
        img=img.resize((256, 256))
        test_image = image.img_to_array(img)
        test_image = np.expand_dims(test_image, axis = 0)
        prediction = reloaded_model.predict(test_image,batch_size=32)
        st.write("Predicted label : ", class_names[np.argmax(prediction[0])])
        confidence = round(100 * (np.max(prediction[0])), 2)
        st.write('Confidence : ',confidence)

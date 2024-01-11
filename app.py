import streamlit as st
import tensorflow as tf





@st.cache_data
def load_model():
    model=tf.keras.models.load_model('currency.hdf5')
    return model
model=load_model()

st.write("""
       #INDIAN CURRENCY NOTES IDENTIFICATION
""")

file=st.file_uploader("Please upload an CURRENCY image",type=['jpg','png'])

import cv2
from PIL import Image,ImageOps
import numpy as np

def predict_function(img,model):
    size=(64,64)
    image=ImageOps.fit(img,size,Image.ANTIALIAS)
    img_arr=np.asarray(image)
    img_scaled=img_arr/255
    img_reshape=np.reshape(img_scaled,[1,64,64,3])
    prediction=model.predict(img_reshape)
    output=np.argmax(prediction)
    if(output==3):
        return 'The note is Two Thousand Rupees Note(2000)'
    elif(output==2):
        return 'The note is Ten Rupees Note(10)'
    elif(output==1):
        return 'The note is Twenty Rupees Note(20)'
    elif(output==4):
        return 'The note is Five Hundered Rupees Note(500)'
    elif(output==5):
        return 'The note is Hundered Rupees Note(100)'
    elif(output==7):
        return 'The Image is Background Image'
    elif(output==0):
        return 'The note is Fifty Rupees Note(50)'
    elif(output==6):
        return 'The note is Two Hundered Rupees Note(200)'

    

if file is None:
    st.text('Please upload an image file')
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    result=predict_function(image,model)
    st.success(result)
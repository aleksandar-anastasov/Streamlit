# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 10:34:06 2021
​
@author: Ana
"""

import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Model is cached when loaded first time in order to use the object next time
@st.cache(allow_output_mutation=True)
def load_model(path = 'best_model_mobile.pt'):
    '''
    Loads pretrained model into memory.
    
    Returns
    -------
    A Keras model instance.
    '''
    return tf.keras.models.load_model(path)

def url_to_image(image):
    '''
    Reads image from url and performs preprocessing.
     
    Parameters
    ----------
    url: string
      Image url.  
    Returns
    -------
    image_resized: image
      Preprocessed image.
    image : image
      Original image.
    '''
    
    image = np.array(image)
    # Resize the image to required input shape by the model
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized  = cv2.resize(image_gray, (224,224))
    # Scale image pixels from 0 to 1 
    image_resized = image_resized/255.0
    
    return image_resized, image 


def get_prediction(image, model, class_names):
    '''
    Predicts image class.
    
    Parameters
    ----------
    image : image object
    model : deep learning model
        
    Returns
    -------
    class_names : list
      Predicted class.
    '''
    # When training the model we have batch dimension
    image = np.expand_dims(image, axis=0)
    # Predict a class for the image
    prediction = np.argmax(model.predict(image))
      
    return class_names[prediction]


def main():
    # Disable streamlit comments and warnings
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.title('BeautifAI Image Classifier')
    st.text('Please provide an Image for classification:')
    # Define class labels
    class_names = ['indoor selfie', 'outdoor selfie', 'indoor pose', 'outdoor pose', 'no human']
    # Some models can be larger, so it takes time to load - so we just print a notification 
    with st.spinner('Loading pretrained model into memory ...'):
      model = load_model()
    
    uploaded_file = st.file_uploader('Choose a File: ', type =['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_resized, image = url_to_image(image)
        st.write('Predicted class: ')
        with st.spinner('Classifying ...'):
          label = get_prediction(image_resized, model, class_names)
          st.write(label)
        st.write("")
        st.image(image, caption='Classifying Image', use_column_width=True)
        
if __name__ == '__main__':
    main()
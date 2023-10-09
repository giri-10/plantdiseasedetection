import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.applications.vgg19 import preprocess_input


st.set_page_config(page_title="Plant Disease Detection") 
model = tf.keras.models.load_model("best_model.h5")  


def main():
    st.title("Plant Disease Detection")
    st.write("Upload plant image")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True, width = 10)


        img_array = np.array(image)
        img_array = tf.image.resize(img_array, (256, 256)) 
        img_array = preprocess_input(img_array)


        pred = np.argmax(model.predict(np.expand_dims(img_array, axis=0)))
        
        ref  = {
			    0: 'Apple scab',
			    1: 'Apple Black rot',
			    2: 'Apple Cedar',
			    3: 'Apple Healthy',
			    4: 'Blueberry Healthy',
			    5: 'Cherry Powdery mildew',
			    6: 'Cherry Healthy',
			    7: 'Corn Gray leaf spot',
			    8: 'Corn Common rust',
			    9: 'Corn Northern Leaf Blight',
			    10: 'Corn healthy',
			    11: 'Grape Black_rot',
			    12: 'Grape Esca (Black Measles)',
			    13: 'Grape Leaf blight (Isariopsis Leaf Spot)',
			    14: 'Grape healthy',
			    15: 'Orange Haunglongbing (Citrus greening)',
			    16: 'Peach Bacterial spot',
			    17: 'Peach healthy',
			    18: 'Pepper,bell Bacterial_spot',
			    19: 'Pepper,bell healthy',
			    20: 'Potato Early blight',
			    21: 'Potato Late blight',
			    22: 'Potato healthy',
			    23: 'Raspberry healthy',
			    24: 'Soybean healthy',
			    25: 'Squash Powdery_mildew',
			    26: 'Strawberry Leaf scorch',
			    27: 'Strawberry healthy',
			    28: 'Tomato Bacterial spot',
			    29: 'Tomato Early blight',
			    30: 'Tomato Late blight',
			    31: 'Tomato Leaf Mold',
			    32: 'Tomato Septoria leaf spot',
			    33: 'Tomato Spider mites Two-spotted spider mite',
			    34: 'Tomato Target Spot',
			    35: 'Tomato Yellow Leaf Curl Virus',
			    36: 'Tomato mosaic virus',
			    37: 'Tomato healthy'
              } 

        
        st.write("Predicted Disease:")
        st.write(ref[pred])

        

if __name__ == "__main__":
    main()


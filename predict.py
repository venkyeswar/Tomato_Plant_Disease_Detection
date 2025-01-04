import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
def load_model(model_path):
    interpreter  =tf.lite.Interpreter(model_path = model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image):
    image_array = img_to_array(image)/255 
    return np.expand_dims(image_array,axis=0)
def make_prediction(model,img_array):
    model.set_tensor(model.get_input_details()[0]["index"],img_array)
    model.invoke()
    output  = model.get_tensor(model.get_output_details()[0]["index"])
    return (np.argmax(output),np.max(output))
def predict(model_path,image_path):
    model = load_model(model_path)
    img_array = preprocess_image(image_path)
    label_index,confidence = make_prediction(model,img_array)
    labels = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 
              'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
              'Tomato___Septoria_leaf_spot', 
              'Tomato___Spider_mites Two-spotted_spider_mite',
             'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
             'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
    prediction = labels[label_index]
    return (prediction,confidence)




from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2 as cv
# Load the model
model = load_model('keras_model.h5')

# def rescaleImage(frame , scale=0.75):
#     width = int(frame.shape[1] * scale)
#     height = int(frame.shape[1] * scale)
#     demension = (width, height)
#     return cv.resize(frame, demension, interpolation=cv.INTER_AREA)
# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.



data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# Replace this with the path to your image
image = Image.open('text.jpg').convert('RGB')
# image = image.resize((224, 224))
#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

#turn the image into a numpy array
image_array = np.asarray(image)
# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
# Load the image into the array
data[0] = normalized_image_array

# run the inference
prediction = model.predict(data)
val = prediction.argmax()
def predict(val):
    if(val==0):
        print("plastic")
    elif(val==1):
        print("cardboard")
    elif(val==2):
        print("metal")
    elif(val==3):
        print("paper")
    else:
        print("default")
        
if __name__ == '__main__':
    predict(val)
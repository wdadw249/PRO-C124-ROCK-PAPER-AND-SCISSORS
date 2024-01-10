import cv2
import tensorflow as tf
import numpy as np

model = tf.keeras.models.load_model("keras_model.h5")

vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()
    img = cv2.resize(frame,(224,224))
    test_image = np.array(img,dtype=np.float32)
    test_image = np.expand_dims(test_image,axis=0)
    normalized_image = test_image/255.0
    prediction = model.predict(normalized_image)
    print("Prediction: ", prediction)
    cv2.imshow("Rock Sciccors Papers",frame)
    key = cv2.waitKey(1)
    if key == 32:
        break
    
cv2.destroyAllWindows()


import cv2
import tensorflow as tf

categories = ["Dog", "Cat"]
filepath = 'cat.jpg'

def prepare(filepath):
    img_size = 100
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (img_size,img_size))

    return img_resized.reshape(-1, img_size, img_size, 1)

model = tf.keras.models.load_model("32x3-CNN.model")

prediction = model.predict([prepare(filepath)])

print("This is a {}!".format(categories[int(prediction[0][0])]))
pic = cv2.imread(filepath)
pic = cv2.resize(pic, (400,400))

cv2.rectangle(pic, (0,0), (80,60), (255,255,255), -1)
cv2.putText(pic, categories[int(prediction[0][0])], (10,40), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 1)

while 1:
    cv2.imshow("result", pic)
    key = cv2.waitKey(0)
    if key == 27:
        break

cv2.destroyAllWindows()
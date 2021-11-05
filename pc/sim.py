import tensorflow as tf
import scipy.misc
import model
import cv2
from subprocess import call
import os
import time

sess = tf.InteractiveSession()
saver = tf.train.Saver()
for filename in os.listdir("save/"):
    print(filename)

saver.restore(sess, "save/model.ckpt")

#img = cv2.imread('C:/Users/liam/Desktop/Capstone/_PC/steering_wheel_image.jpg',0)
#rows,cols = img.shape

#smoothed_angle = 0

filelist = []
img_angles = []

for filename in os.listdir("images/"):
    filelist.append("images/" + filename)
    filenameNoExt = os.path.splitext(filename)[0]
    index, angle= filenameNoExt.split("-")
    img_angles.append(angle)
 
i = 0
smoothed_angle = 0

while(cv2.waitKey(10) != ord('q')):
    full_image = scipy.misc.imread(filelist[i], mode="RGB")
    image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0
    output = (model.y.eval(session=sess, feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180.0 / scipy.pi)
    steeringAngle = 128 + int(output*128/180)
    smoothed_angle += 0.2 * pow(abs((steeringAngle - smoothed_angle)), 2.0 / 3.0) * (steeringAngle - smoothed_angle) / abs(steeringAngle - smoothed_angle)


    full_image = cv2.resize(full_image,None,fx=2.0,fy=2.0)
    cv2.putText(full_image, "NN / DATA",(250,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(full_image, str(int(smoothed_angle)) + "/" + img_angles[i],(250,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
    time.sleep(0.2)
    i += 1


cv2.destroyAllWindows()

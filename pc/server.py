import sys
import threading
import socketserver
import numpy as np
import cv2
import pygame
from pygame.locals import *
import socket
import time
import scipy.misc
import tensorflow as tf
import model

HOST = "0.0.0.0"
VPORT = 8000
CPORT = 8005

can_run = True

steering_angle = 128
speed = 0

record = False
drive_self = False

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt")

lock = threading.Lock()

class ControlStreamHandler(socketserver.StreamRequestHandler):

    def handle(self):
        #control stream is incharge of globals
        global steering_angle
        global speed
        global can_run
        global record
        global drive_self

        #init pygame to capture controller input
        pygame.init()

        pygame.joystick.init()

        joystick = pygame.joystick.Joystick(0)
        joystick.init()

        smoothed_angle = 0
        
        try:
            while can_run:
                    for event in pygame.event.get():  # User did something
                        if event.type == pygame.QUIT or joystick.get_button(4):  # If user clicked close
                            can_run = False  # Flag that we are done so we exit this loop
                            break
                        if event.type == pygame.JOYBUTTONDOWN:
                            if joystick.get_button(6) and not drive_self: # button with squares
                                record = not record
                            elif joystick.get_button(7) and not record: # menu button
                                drive_self = not drive_self
                            elif joystick.get_button(0) and speed < 16: #A button
                                speed += 1
                            elif joystick.get_button(1) and speed > 0: #B button
                                speed -= 1
                            elif joystick.get_button(2): #Y button
                                speed = 0

                    axis = ((joystick.get_axis(0) + 1.0) / 2.0)

                    #scale inputs to bytes
                    scaled_axis = int(axis*256)
                    scaled_speed = int(speed*8)
                    
                    #commands will be sent over bytearray
                    move_cmd = bytearray()

                    with lock:
                        #if not drive self use value from user, otherwise hope that the global is taking care of it
                        if not drive_self:
                            steering_angle = scaled_axis
                    
                        if steering_angle > 118 and steering_angle < 138:
                            steering_angle = 128

                        smoothed_angle += 0.2 * pow(abs((steering_angle - smoothed_angle)), 2.0 / 3.0) * (steering_angle - smoothed_angle) / abs(steering_angle - smoothed_angle)

                        move_cmd.append(int(smoothed_angle))

                    #append scaled speed as second byte in byte array
                    move_cmd.append(scaled_speed)

                    #wfile used to write stream
                    self.wfile.write(move_cmd)

        except Exception as e:
            print('ControlStream Error: '+ str(e))

        finally:
            sys.exit()


class VideoStreamHandler(socketserver.StreamRequestHandler):

    def handle(self):
        global steering_angle

        stream_bytes = b' '
        image_num = 0
        #img record interval, count to 3 and reset in loop (should be 30fps/3 = 10fps)
        rec_interval = 0
        try:
            while can_run:
                #rfile is used to read stream
                stream_bytes += self.rfile.read(1024)

                first = stream_bytes.find(b'\xff\xd8')
                last = stream_bytes.find(b'\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]
                    image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                    #scale image and save shape data to use for display
                    scaled_image = cv2.resize(image,None,fx=3.0,fy=3.0)
                    image_h, image_w, image_c = scaled_image.shape
                    
                    #if recording save, the original image and draw notification
                    if record:
                        cv2.putText(scaled_image,"REC",((image_w-70),(30)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                        if rec_interval == 3:
                            cv2.imwrite(('images/{:>07}-{}.jpg').format(image_num,steering_angle), image)
                            cv2.circle(scaled_image,((image_w-85),(20)), 10, (0,0,255), -1)
                            image_num += 1
                    
                    #white text color
                    text_color = (255,255,255)
                    
                    #if self driving mode send image into model and set steering angle based on that, make steering angle green to signify auto mode
                    if drive_self:
                        feed_image = scipy.misc.imresize(image[-150:], [66, 200]) / 255.0
                        output = (model.y.eval(session=sess, feed_dict={model.x: [feed_image], model.keep_prob: 1.0})[0][0] * 180.0 / scipy.pi)
                        with lock:
                            steering_angle = 128 + int(output*128/180)
                        text_color = (57,255,20)

                    cv2.putText(scaled_image,str(speed),(image_w-20,(image_h-142)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),thickness=1)
                    cv2.rectangle(scaled_image,(image_w-10,(image_h-10)),(image_w-42,(image_h-138)),(255,255,255),thickness=2)
                    cv2.rectangle(scaled_image,(image_w-10,(image_h-10)),(image_w-42,(image_h-10-(speed*8))),(255,255,255),cv2.FILLED)

                    cv2.putText(scaled_image,str(steering_angle),(242,(image_h-46)),cv2.FONT_HERSHEY_SIMPLEX,0.4,text_color,thickness=1)
                    cv2.rectangle(scaled_image,(10,(image_h-10)),(266,(image_h-42)),(255,255,255),thickness=2)
                    cv2.line(scaled_image,(10 + steering_angle,(image_h-8)),(10 + steering_angle,(image_h-44)),text_color,thickness=2)

                    cv2.imshow('Car', scaled_image)

                    if rec_interval == 3:
                        rec_interval = 0
                    else:
                        rec_interval += 1
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        except Exception as e:
            print('VideoStream Error: '+ str(e))

        finally:
            cv2.destroyAllWindows()
            sys.exit()

def start_video_stream(host, port):
    v = socketserver.TCPServer((host, port), VideoStreamHandler)
    v.serve_forever()
    
def start_control_stream(host, port):
    c = socketserver.TCPServer((host, port), ControlStreamHandler)
    c.serve_forever()

#Start Video Server
video_thread = threading.Thread(target=start_video_stream, args=(HOST, VPORT))
video_thread.start()

#Start Control Server
control_thread = threading.Thread(target=start_control_stream, args=(HOST, CPORT))
control_thread.start()

video_thread.join()
control_thread.join()

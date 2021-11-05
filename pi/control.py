from multiprocessing import Process
from rrb3 import *
import sys import argv
import socket
import time
import io
import struct
import picamera

HOST = argv[1]
VPORT = 8000
CPORT = 8005

rr = RRB3(12, 6)

def video_stream():
    class SplitFrames(object):
        def __init__(self, connection):
            self.connection = connection
            self.stream = io.BytesIO()
            self.count = 0

        def write(self, buf):
            if buf.startswith(b'\xff\xd8'):
                # Start of new frame; send the old one's length
                # then the data
                size = self.stream.tell()
                if size > 0:
                    self.connection.write(struct.pack('<L', size))
                    self.connection.flush()
                    self.stream.seek(0)
                    self.connection.write(self.stream.read(size))
                    self.count += 1
                    self.stream.seek(0)
            self.stream.write(buf)

    video_socket = socket.socket()
    video_socket.connect((HOST, VPORT))
    connection = video_socket.makefile('wb')
    res = (320,240)
    start = time.time()
    try:
        output = SplitFrames(connection)
        with picamera.PiCamera(resolution=res, framerate=30) as camera:
            time.sleep(2)
            camera.start_recording(output, format='mjpeg')
            camera.wait_recording(600)
            camera.stop_recording()

            connection.close()
            video_socket.close()

    except socket.error:
        print('Connection Interrupted ...')

    finally:
        finish = time.time()
        print('Sent %d images in %d seconds at %.2ffps' % (output.count, finish-start, output.count / (finish-start)))

Process(target=video_stream).start()

control_socket = socket.socket()
control_socket.connect((HOST,CPORT))

try:
    while True:
        control_data = control_socket.recv(1024)
        control_bytes = bytearray(control_data)

        if not control_bytes:
            break

        direction = int(control_bytes[0])  # between 0 and 256
        speed = int(control_bytes[1])  # between 0 and 128

        direction = (direction - 128)/3

        right_spd = (speed + direction)/float(128)
        left_spd = (speed - direction)/float(128)

        if(right_spd < 0):
            if speed  > 32:
                right_spd = 0.05
            else:
                #backwards
                right_dir = 1
        else:
            #forwards
            right_dir = 0

        if(left_spd < 0):
            if speed > 32:
                left_spd = 0.05
            else:
                #backwards
                left_dir = 1
        else:
            #forwards
            left_dir = 0

        if rr.get_distance() < 10.0:
            rr.stop()
            rr.set_motors(1, 1, 1, 1)
            time.sleep(0.5)
        else:      
            rr.set_motors(abs(right_spd), right_dir, abs(left_spd), left_dir)

finally:
    rr.stop()
    control_socket.close()
    GPIO.cleanup()

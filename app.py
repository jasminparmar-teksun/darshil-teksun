'''
Tensorflow CNN_LSTM Model for Exercise Detection - INSPIRE AI 
author - Darshil Modi
Date - 20 Aug 2021
'''

# necessary imports
from flask import Flask
from flask import request
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPool2D, BatchNormalization
from keras.layers import TimeDistributed, GRU, Dense, Dropout, LSTM, Flatten
from keras.layers import LeakyReLU
from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import tensorflow as tf
import boto3
import cv2
import csv
import os
from subprocess import call
import json
import datetime
import numpy as np
import paho.mqtt.client as mqtt
from os.path import dirname, abspath
import ssl
import json
from glob import glob
import time
from threading import Thread
import random
import threading
import logging
import flask
import watchtower


app = Flask(__name__)                                # Flask app for creating API
logging.basicConfig(level=logging.INFO)    
app = flask.Flask("loggable")


@app.route("/", methods=['POST', 'GET'])
def home():
    return "Hello Darshil"                           # Text to be displayed on webpage and prediction model will run in backend


@app.route("/predictions")
def rundetection():                                  # http headers and ip address for threading and ECS Clustering
    global ip_address, headers_list
    global STREAM_NAME
    global tr
    global n
    status = "started"
    ip_address = request.remote_addr
    headers_list = request.headers.getlist("HTTP_X_FORWARDED_FOR")
    n = random.randint(0, 500)
    STREAM_NAME = request.args['streamname']
    time.sleep(5)
    tr = TestThreading()


    return str(n)


class TestThreading(object):                         # threading 
    global thread

    def __init__(self, interval=1):
        thread = threading.Thread(target=self.run, args=())

        self.interval = interval

        thread.daemon = True
        # print("execution started")
        thread.start()

    def run(self):
        predictions(n, STREAM_NAME)
        # print("here running properly")
        os._exit(0)


def predictions(mm01, streamname):
    
    '''
     This function contains deep learning model consisting of CNN and LSTM layers. CNN is used for extracting useful
    features from the image pixels and LSTM is used to remember sequence of frames. Model Input shape is (12,200,200,3)
    Parameters required are as follows -
    mm01 : thread_id
    streamname : Kinesis stream name
    '''

    output = ''
    app_output = ''
    confidence = 0.00

    SIZE = (200, 200)
    CHANNELS = 3
    NBFRAME = 12
    BS = 10
    
    thread_id = str((threading.get_ident()))

    shape = (NBFRAME,) + SIZE + (CHANNELS,)  # (5, 112, 112, 3)

    model = Sequential()

    model.add(TimeDistributed(Conv2D(16, (5, 5)), input_shape=shape))
    model.add(TimeDistributed(LeakyReLU(alpha=0.05)))
    model.add(TimeDistributed(MaxPooling2D(
        (2, 2), strides=(4, 4), padding='same')))

    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same')))
    model.add(TimeDistributed(LeakyReLU(alpha=0.05)))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same')))
    model.add(TimeDistributed(LeakyReLU(alpha=0.05)))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same')))
    model.add(TimeDistributed(LeakyReLU(alpha=0.05)))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same')))
    model.add(TimeDistributed(LeakyReLU(alpha=0.05)))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same')))
    model.add(TimeDistributed(LeakyReLU(alpha=0.05)))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    # extract features and dropout
    model.add(TimeDistributed(Flatten()))
    model.add(Dropout(0.3))

    # input to LSTM
    model.add(LSTM(128, return_sequences=False, dropout=0.3))

    # classifier with sigmoid activation for multilabel
    model.add(Dense(20, activation='softmax'))

    # load model checkpoint
    model.load_weights('checkpoint_8_07/cp-030.pkl')  

    optimizer = optimizers.Adam(0.0005)
    model.compile(optimizer, 'categorical_crossentropy', metrics=['acc'])

    file1 = open("stream_payload.txt", "w")
    file1.write("\n first line  " + str(datetime.datetime.now()))
    STREAM_NAME = streamname
    kvs = boto3.client("kinesisvideo", region_name='us-west-2', aws_access_key_id="AKIA5QENINP63ROHPSVN",
                       aws_secret_access_key="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")
    # Grab the endpoint from GetDataEndpoint
    file1.write("\n get endpoint " + str(datetime.datetime.now()))
    endpoint = kvs.get_data_endpoint(
        APIName="GET_HLS_STREAMING_SESSION_URL",
        StreamName=STREAM_NAME)['DataEndpoint']

    file1.write("\n endpoint received" + str(datetime.datetime.now()))
    # print(endpoint)
    file1.write("\n get url  " + str(datetime.datetime.now()))
    # # Grab the HLS Stream URL from the endpoint
    kvam = boto3.client("kinesis-video-archived-media", endpoint_url=endpoint, aws_access_key_id="AKIA5QENINP63ROHPSVN",
                        aws_secret_access_key="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", region_name='us-west-2')

    class BlowerCloud_AWS:
        """This class is used to connect with MQTT broker on AWS. More details can
        be found from here --> https://www.eclipse.org/paho/clients/python/docs/
        """

        def __init__(self):
            self.mqttc = mqtt.Client()
            self.sTopic = ''
            self.pTopic = ''

            self.PORT = 8883
            self.KEEPALIVE_INTERVAL = 10

            self.MQTT_HOST = "atl0ryxpkp276-ats.iot.us-west-2.amazonaws.com"
            self.CA_ROOT_CERT_FILE = r"certificates/AmazonRootCA1.pem"
            self.THING_CERT_FILE = r"certificates/e899f91bb1-certificate.pem.crt"
            self.THING_PRIVATE_KEY = r"certificates/e899f91bb1-private.pem.key"

            self.mqttc.on_connect = self.on_connect
            self.mqttc.on_disconnect = self.on_disconnect
            self.mqttc.on_publish = self.on_publish
            self.mqttc.on_subscribe = self.on_subscribe
            self.mqttc.on_message = self.on_message

            self.mqttc.will_set("mytopic/publish/disconnect", payload=None, qos=1,
                                retain=False)

        def connect(self, sTopic):
            """ This function is used to provide SSL certificates and connect with
            MQTT host.
            :param  sTopic  topic that need to be subscribed
            :return None
            """
            self.sTopic = sTopic
            # try:
            self.mqttc.tls_set(self.CA_ROOT_CERT_FILE,
                               certfile=self.THING_CERT_FILE,
                               keyfile=self.THING_PRIVATE_KEY,
                               cert_reqs=ssl.CERT_REQUIRED,
                               tls_version=ssl.PROTOCOL_TLSv1_2,
                               ciphers="ECDHE-RSA-AES128-GCM-SHA256"
                               )
            self.mqttc.connect(self.MQTT_HOST, self.PORT,
                               self.KEEPALIVE_INTERVAL
                               )
            return True
            # except:
            # print("Connection with MQTT brocker failed.\n")
            # return None

        def on_message(self, mosq, obj, msg):
            """ This function is used to capture MQTT messages from MQTT broker.
            :param  mosq:   client
            :param  obj:    user_data
            :param  msg:    message
            :return None
            """
            # print("\nTopic: " + str(msg.topic))
            # print("QoS: " + str(msg.qos))
            # print("Payload: " + str(msg.payload))

        '''
        def disconnect(self):
            print("going to disconnect")
            self.mqttc.disconnect()
        '''

        def MonitorConnection(self):
            """ It monitors whole mqtt connections and receives incoming messages
            from broker.
            :param  None
            :return None
            """
            # print("in MonitorConnection")
            self.mqttc.loop_start()

        def PublishData(self, topic, data):
            """ It is used to publish data on specific topic.
            :param  topic:  topic on which data needs to be published
            :param  data:   data which needs to be published
            :return None
            """
            print("in PublishData")
            data = json.dumps(data)
            self.mqttc.publish(self.pTopic + topic, data, qos=0)

        def on_publish(self, client, userdata, mid):
            """ It is a callback of publish message.
            :param  client:     client
            :param  userdata:   a user provided object that will be passed to the
                                on_message callback when a message is received
            :param  mid:        message ID for the publish request
            :return None
            """
            print('published\n')
            pass

        def on_subscribe(self, mosq, obj, mid, granted_qos):
            """ It is a callback to successfully subscribe topic.
            :param  mosq:           client
            :param  obj:            user_data
            :param  mid:            message_id
            :param  granted_qos:    quality of service
            :return None
            """
            print('Subscribed %s with Qos : %s\n' %
                  (self.sTopic, str(granted_qos)))

        def on_connect(self, mqtt, mosq, obj, rc):
            """ It is a callback to successfully subscribe topic.
            :param  mqtt:   class instance
            :param  mosq:   client
            :param  obj:    user_data
            :param  rc:     connection result
            :return None
            """

            print('connected\n')
            self.mqttc.reconnect_delay_set(min_delay=1, max_delay=30)
            self.mqttc.subscribe(self.sTopic, 0)

        def on_disconnect(self, mqtt, mosq, rc):
            """ It is a callback to mqtt connection disconnect event.
            :param  mqtt:   instance to client
            :param  mosq:   client
            :param  rc:     the connection result
            :return None
            """
            print('connection lost\n')
            self.mqttc.reconnect()

            
    # test on AWS KINESIS - LIVE and ON_DEMAND 
    
    i = 1
    while (True):
        try:
            url = kvam.get_hls_streaming_session_url(
                StreamName=STREAM_NAME,
                Expires=43200,
                
                '''
                if you wish to test on live stream comment the TimestampRange attribute and change PlaybackMode attribute 
                to 'LIVE'. If you wish to test on prerecorded AWS Stream, write the timestamps in TimestampRange attribute 
                and change PlaybackMode attribute to 'ON_DEMAND'
                '''
                
                HLSFragmentSelector={
                    "FragmentSelectorType": "PRODUCER_TIMESTAMP",

                    # "TimestampRange": {
                    #        "EndTimestamp": '13 Feb 2021 06:50:50 UTC',
                    #         "StartTimestamp": '13 Feb 2021 06:47:46 UTC'
                    #    }
                },
                PlaybackMode="LIVE"
            )['HLSStreamingSessionURL']
            
            handler = watchtower.CloudWatchLogHandler()
            app.logger.addHandler(handler)
            logging.getLogger("tensorflow").addHandler(handler)

        except:
            logger = logging.getLogger("tensorflow")
            logger.addHandler(watchtower.CloudWatchLogHandler())
            logger.info("No Stream with thread_id %s" %(thread_id))
            break
            
 

        cap = cv2.VideoCapture(url)                                     # capture frames from kinesis URL
    
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')                        # initiating video writer to save output video

        images = []

        cloud = BlowerCloud_AWS()
        connectionStatus = cloud.connect("Inspire_AI/AI_Camera/"+streamname)

        counter = 0

        sliding_window = []
        confidence_window = []

        ret, frame = cap.read()
        frame = cv2.resize(frame, (200, 200))
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        output = 'fetching results..'                                    # initializing output variables
        final_output = 'processing..'                                    # initializing output variables

        while (cap.isOpened()):

            counter = counter + 1

            if counter % 13 == 0:                                        # once 13 frames are collected, pass frames to model
                images = np.array(images)
                images = np.expand_dims(images, axis=0)
                
                # exercise labels mapped with their one-hot codes 

                exercise_dict = {0: 'Abdominal Crunch With Bar', 1: 'Alternating Bicep Curl', 2: 'Flat Chest Fly',
                                 3: 'Flat Chest Press', 4: 'Incline Chest Fly', 5: 'Incline Chest Press', 6: 'Lunge With Bar Behind Back',
                                 7: 'no_exercise', 8: 'Pullup', 9: 'Rope Curls', 10: 'Seated Lat Pulldown', 11: 'Seated Shoulder Press',
                                 12: 'Side laterals', 13: 'Seated Tricep Extension', 14: 'Single Arm Bent Over Row', 15: 'Standing Core Rotation',
                                 16: 'Straight Arm Core Rotation With Bar', 17: 'Standing Flys', 18: 'Straight Arm Pressdown', 19: 'Tricep Pressdown'}

                prediction = model.predict(images / 255)                  # model output
                pred_ans = np.argmax(np.round(prediction, 2))             # finding class with max confidence
                confidence = prediction[0][pred_ans]                      # finding confidence score
                init_output = exercise_dict[pred_ans]                     # finding exercise name from one-hot code
   

                if init_output == 'no_exercise':                          # only consider label if it crosses defined threshold
                    output = init_output
                if init_output == 'Single Arm Bent Over Row':
                    if confidence > 0.989:
                        output = init_output
                    else:
                        output = 'no_exercise'
                else:
                    if confidence > 0.965:
                        output = init_output

                if len(sliding_window) < 6:                               # consider the label that appears more than 3 times/6
                    sliding_window.append(output)
                    confidence_window.append(confidence)
           
                    freq = {}
                    for items in sliding_window:
                        freq[items] = sliding_window.count(items)
      
                    keys = [k for k, v in freq.items() if v > 2]
                  
                    if len(keys) > 0:
                        final_output = keys[-1]

                else:
                    sliding_window.append(output)
                    confidence_window.append(confidence)
                    freq = {}
                    for items in sliding_window:
                        freq[items] = sliding_window.count(items)
                    keys = [k for k, v in freq.items() if int(v) > 2]
                   
                    if len(keys) > 0:
                        final_output = keys[-1]
                    
                    sliding_window.pop(0)
                    confidence_window.pop(0)

                if app_output != 'processing..':

                    app_output = {'label': final_output,'thread_id': thread_id}    
                    print(app_output)                                # final output after sliding window and threshold
                    logger = logging.getLogger("tensorflow")
                    logger.addHandler(watchtower.CloudWatchLogHandler())
                    logger.info("Label is %s and thread_id is %s" %(final_output, thread_id))
                    topic = 'Inspire_AI/AI_Camera/'+streamname
                    cloud.PublishData(topic, app_output)             # publish labels on MQTT CLOUD

                images = []

            else:                                                    # append frames in list 'images' until 12 frames collected

                ret, frame = cap.read()  
                result = frame.copy()
                frame = cv2.resize(frame, (200, 200))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(prev_gray, gray)                  # for absolute difference between 2 consecutive frames
                diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2RGB)
                images.append(diff)
                prev_gray = gray

    return url


if __name__ == "__main__":

    app.run(host="0.0.0.0", port=int(
        os.environ.get("PORT", 80)), threaded=True)

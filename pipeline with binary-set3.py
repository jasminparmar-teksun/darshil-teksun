'''
Tensorflow CNN_LSTM Model for Exercise Detection - INSPIRE AI 
author - Darshil Modi
Date - 20 Aug 2021
'''

# necessary imports
from tensorflow.keras.layers import Conv2D,MaxPooling2D, GlobalMaxPool2D, BatchNormalization
from tensorflow.keras.layers import TimeDistributed, GRU, Dense, Dropout, LSTM, Flatten
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
import tensorflow as tf
import boto3
import numpy
import cv2
import csv
from  subprocess import call
import json
import datetime
import numpy as np
import paho.mqtt.client as mqtt
from os.path import dirname, abspath
import ssl
import json
from glob import glob
import time


# defining input shape of the model
SIZE = (200, 200)
CHANNELS = 3
NBFRAME = 12
BS = 10

shape=(NBFRAME,) + SIZE + (CHANNELS,) # (5, 112, 112, 3)




def model(output, checkpoint):
    
    '''
    This function contains deep learning model consisting of CNN and LSTM layers. CNN is used for extracting useful
    features from the image pixels and LSTM is used to remember sequence of frames.
    input shape : (number of frames, image width, image height, channel) - eg. (12,200,200,3)
    output : number of classes - (no of exercise that we wish to classify)
    checkpoint : relative path of checkpoint file (.pkl)
    '''

    model = Sequential()

    model.add(TimeDistributed(Conv2D(16, (5, 5)), input_shape=shape))
    model.add(TimeDistributed(LeakyReLU(alpha=0.05)))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(4, 4), padding='same')))

    model.add(TimeDistributed(Conv2D(32, (3,3), padding='same')))
    model.add(TimeDistributed(LeakyReLU(alpha=0.05)))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(64, (3,3), padding='same')))
    model.add(TimeDistributed(LeakyReLU(alpha=0.05)))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(128, (3,3), padding='same')))
    model.add(TimeDistributed(LeakyReLU(alpha=0.05)))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(64, (3,3), padding='same')))
    model.add(TimeDistributed(LeakyReLU(alpha=0.05)))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(32, (3,3), padding='same')))
    model.add(TimeDistributed(LeakyReLU(alpha=0.05)))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))


    # extract features and dropout 
    model.add(TimeDistributed(Flatten()))
    model.add(Dropout(0.3))

    # input to LSTM
    model.add(LSTM(256, return_sequences=False, dropout=0.3))

    # classifier with sigmoid activation for multilabel
    model.add(Dense(output, activation='softmax'))
    
    model.load_weights(checkpoint)
    
    optimizer = optimizers.Adam(0.0005)
    model.compile(optimizer,'categorical_crossentropy', metrics=['acc'] )
    
    return model


#calling model function with 30 classes and trained model checkpoint path
model_exercise = model(30,'./final_results/checkpoints/set3_17Aug/cp-040.pkl')



#boto3 kinesis stream fetching - enter the kinesis streamname in Stream_NAME (case-sensitive)

file1 = open("stream_payload.txt","w")
file1.write("\n first line  "+str(datetime.datetime.now()))
STREAM_NAME = "TeksunAITeam"
kvs = boto3.client("kinesisvideo")
# Grab the endpoint from GetDataEndpoint
file1.write("\n get endpoint "+str(datetime.datetime.now()))
endpoint = kvs.get_data_endpoint(
    APIName="GET_HLS_STREAMING_SESSION_URL",
    StreamName=STREAM_NAME)['DataEndpoint']

file1.write("\n endpoint received"+str(datetime.datetime.now()))
print(endpoint)
file1.write("\n get url  "+str(datetime.datetime.now()))
# # Grab the HLS Stream URL from the endpoint
kvam = boto3.client("kinesis-video-archived-media", endpoint_url=endpoint)



# code for MQTT AWS - passing predictions to the iothub

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
        self.CA_ROOT_CERT_FILE = r"/home/darshilmodi/IF_video_pipeline/final_results/certificates/AmazonRootCA1.pem"
        self.THING_CERT_FILE = r"/home/darshilmodi/IF_video_pipeline/final_results/certificates/e899f91bb1-certificate.pem.crt"
        self.THING_PRIVATE_KEY = r"/home/darshilmodi/IF_video_pipeline/final_results/certificates/e899f91bb1-private.pem.key"

        #self.MQTT_HOST = "a1jzj5nr2pv53i-ats.iot.us-east-1.amazonaws.com"
        #self.CA_ROOT_CERT_FILE = glob(dirname(abspath(__file__))
        #                              + "/key/AmazonRootCA*.pem")[0]
        #self.THING_CERT_FILE = glob(dirname(abspath(__file__))
        #                            + "/key/*certificate*")[0]
        #self.THING_PRIVATE_KEY = glob(dirname(abspath(__file__))
        #                              + "/key/*private*")[0]

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
            #try:
            self.mqttc.tls_set( self.CA_ROOT_CERT_FILE,
                                certfile = self.THING_CERT_FILE,
                                keyfile = self.THING_PRIVATE_KEY,
                                cert_reqs = ssl.CERT_REQUIRED,
                                tls_version = ssl.PROTOCOL_TLSv1_2,
                                ciphers = "ECDHE-RSA-AES128-GCM-SHA256"
                               )
            self.mqttc.connect( self.MQTT_HOST, self.PORT,
                                self.KEEPALIVE_INTERVAL
                                )
            return True
            #except:
            print("Connection with MQTT brocker failed.\n")
            #return None

    def on_message(self, mosq, obj, msg):
        """ This function is used to capture MQTT messages from MQTT broker.
        :param  mosq:   client
        :param  obj:    user_data
        :param  msg:    message
        :return None
        """
        print("\nTopic: " + str(msg.topic))
        print("QoS: " + str(msg.qos))
        print("Payload: " + str(msg.payload))

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
        #print("in MonitorConnection")
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
        print('Subscribed %s with Qos : %s\n'%(self.sTopic, str(granted_qos)))

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
        
cloud = BlowerCloud_AWS()
connectionStatus = cloud.connect("Inspire_AI/AI_Camera/tensorflow")



# TESTING ON AWS KINESIS STREAM - LIVE OR ON_DEMAND

i=1
while(True):
    try:
        url = kvam.get_hls_streaming_session_url(
        StreamName=STREAM_NAME,
        Expires = 43200 ,
            
        '''
        if you wish to test on live stream comment the TimestampRange attribute and change PlaybackMode attribute 
        to 'LIVE'. If you wish to test on prerecorded AWS Stream, write the timestamps in TimestampRange attribute 
        and change PlaybackMode attribute to 'ON_DEMAND'
        '''
            
        HLSFragmentSelector= { 
                  "FragmentSelectorType": "PRODUCER_TIMESTAMP",
                    
#                   "TimestampRange": { 
#                          "EndTimestamp": '18 Aug 2021 06:59:10 UTC',
#                           "StartTimestamp": '18 Aug 2021 06:27:27 UTC'  
#                      }
                       },
        PlaybackMode="LIVE"
        )['HLSStreamingSessionURL']
    except:
        print("no stream")
        
    
    #capturing video frames from HSL Stream
    
    cap = cv2.VideoCapture(url)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    
    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4)) 
   
    size = (frame_width, frame_height)

    images = []
    
    #writer to save the output video file
    writer = cv2.VideoWriter('output'+str(datetime.datetime.now())+'.avi', fourcc, 10, size) 
    
    #establishing connection with AWS MQTT
    cloud = BlowerCloud_AWS()
    connectionStatus = cloud.connect("Inspire_AI/AI_Camera/AWSLabel")
    
    counter = 0
    
    sliding_window = []
    confidence_window = []
    
    output = 'fetching results..'
    final_output = 'processing..'
    confidence = 'processing..'
    app_output = ''
    
    ret, frame = cap.read()
    frame = cv2.resize(frame, (200, 200))
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    while (cap.isOpened()): 

        counter = counter + 1
        
        # if 13 frames collected..
        if counter % 13 == 0 :
            images = np.array(images)
            images = np.expand_dims(images, axis=0) #for matching input shape of the model


            # exercise names with their one-hot encoding codes 
            exercise_dict = {0: 'abdominalcrunchwithbar', 1: 'abdominalcrunchwithrope', 2: 'alternatingbicepcurl', 3: 'deadlift', 4: 'flatchestfly', 5: 'flatchestpress',
                            6: 'inclinechestfly', 7: 'inclinechestpress', 8: 'inner thigh', 9: 'leg curl', 10: 'leg kickback',
                            11: 'no move', 12: 'outer thigh', 13: 'pullup', 14: 'ropecurls', 15: 'seated biceps curl', 
                            16: 'seated low row', 17: 'seatedlatpulldown', 18: 'seatedshoulderpress', 19: 'seatedsidelaterals', 
                            20: 'seatedtricepextension', 21: 'single_leg_lunge', 22: 'singlearmbentoverrow', 23: 'squats', 
                            24: 'standing chest press punch', 25: 'standingcorerotation', 26: 'standingcorerotationwithbar',
                            27: 'standingflys', 28: 'straightarmpressdown', 29: 'triceppressdown'}
    
    
        
                           
            final_prediction = model_exercise.predict(images/255) #model prediction
            pred_ans = np.argmax(np.round(final_prediction,2)) 
            confidence = final_prediction[0][pred_ans] # confidence score of the prediction
            init_output = exercise_dict[pred_ans] # fetching name of one-hot code from above exercise_dict

            #only consider label if confidence crosses the mentioned threshold
            
            if init_output == 'no_exercise':
                output = init_output

            if init_output == 'Single Arm Bent Over Row':
                if confidence > 0.989:
                    output = init_output
                else:
                    output = 'no_exercise'
            else:
                if confidence > 0.95:
                    output = init_output


        # sliding window to deal with flickering outputs. consider the label that appears maximum time out of 5 labels
        
        if len(sliding_window) < 6:
            sliding_window.append(output)
            confidence_window.append(confidence)
            
            freq = {}
            for items in sliding_window:
                freq[items] = sliding_window.count(items)
            
            keys = [k for k, v in freq.items() if v > 2]
            if len(keys)> 0:
                final_output = keys[-1]

        else:
            sliding_window.append(output)
            confidence_window.append(confidence)
            
            freq = {}
            for items in sliding_window:
                freq[items] = sliding_window.count(items)
            
            keys = [k for k, v in freq.items() if int(v) > 2]
            
            if len(keys)> 0:
                final_output = keys[-1] 
            sliding_window.pop(0)
            confidence_window.pop(0)

        if app_output != 'processing..':

            app_output = {'label':final_output}
            print(final_output)                                # final output will be sent to MQTT

        images = []

        else:                                                  # Append frames until counter counts 12 frames

        ret, frame = cap.read()  
        result = frame.copy()
        frame = cv2.resize(frame, (200, 200))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)                     # for absolute difference of current frame with previous frame
        diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2RGB)
        cv2.imshow('input',diff)
        images.append(diff)
        prev_gray = gray

        # printing predictions with confidence score on output video
        
        result = cv2.putText(result,final_output, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        result = cv2.putText(result,str(confidence), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        writer.write(result)
        cv2.imshow('result',result)


    if cv2.waitKey(25) & 0xFF == ord('q'): 
        break

cap.release() 
cv2.destroyAllWindows()



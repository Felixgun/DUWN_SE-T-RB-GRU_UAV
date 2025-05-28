# Set your parameters here
ID = "0870"
seq_length = 35  # Length of sequence
filename = "test13des3"
anchor_num = 3
numb_tag = 1
model_folder = "run11des"
RASP_IP = '192.168.134.208'

box_length=10

xy_coord = [0, 0]


import sklearn
import sys
import os

# Paths from terminal Python environment
paths_to_add = [
    '/usr/local/lib/python3.8/site-packages',
    '/home/t2508',
    '/usr/lib/aarch64-linux-gnu/',
    '/home/t2508/.local/lib/python3.8/site-packages/scikit_learn.libs/'
]

# Add each path to the sys.path in Jupyter
for path in paths_to_add:
    if path not in sys.path:
        sys.path.append(path)

# Verify the updated sys.path
print(sys.path)
os.environ['LD_PRELOAD']='/usr/lib/aarch64-linux-gnu/libgomp.so.1'
os.environ['LD_PRELOAD']='/home/t2508/.local/lib/python3.8/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0'

def T2A(id):
    print('t2a: ',id)
    '''
    id = 0 : anchor num 1
    '''
    i0 = ["DW17A0", {"x":10,"y":0,"z":1.3,"quality":100},False,"17a0"]
    i1 = ["DW1E91", {"x":0,"y":0,"z":3.1,"quality":100},False,"1e91"]
    i2 = ["DWD0A8", {"x":10,"y":9,"z":3.1,"quality":100},True,"d0a8"]
    i3 = ["DW9A90", {"x":0,"y":9,"z":1.3,"quality":100},False,"9a90"]
    i4 = ["DW04CF", {"x":0,"y":18,"z":3.1,"quality":100},False,"04cf"]
    i5 = ["DW1E89", {"x":10,"y":18,"z":1.3,"quality":100},False,"1e89"]

    if id == 0:   ii = i0
    elif id == 1: ii = i1
    elif id == 2: ii = i2
    elif id == 3: ii = i3
    elif id == 4: ii = i4
    elif id == 5: ii = i5

    payload = {"configuration":{"label": ii[0],
                               "nodeType":"ANCHOR",
                               "ble":False,
                               "leds":True,
                               "uwbFirmwareUpdate":False,
                               "anchor":{"initiator":ii[2],
                                         "position":ii[1],
                                         "routingConfig":"ROUTING_CFG_OFF"}}}
    return payload , ii[3]

def A2T(id):
    print('a2t: ', id)
    '''
    id = 0 : anchor num 1
    '''
    i0 = ["DW17A0", {"x":10,"y":0,"z":1.3,"quality":100},True,"17a0"]
    i1 = ["DW1E91", {"x":0,"y":0,"z":3.1,"quality":100},True,"1e91"]
    i2 = ["DWD0A8", {"x":10,"y":9,"z":3.1,"quality":100},True,"d0a8"]
    i3 = ["DW9A90", {"x":0,"y":9,"z":1.3,"quality":100},True,"9a90"]
    i4 = ["DW04CF", {"x":0,"y":18,"z":3.1,"quality":100},True,"04cf"]
    i5 = ["DW1E89", {"x":10,"y":18,"z":1.3,"quality":100},True,"1e89"]

    if id == 0:   ii = i0
    elif id == 1: ii = i1
    elif id == 2: ii = i2
    elif id == 3: ii = i3
    elif id == 4: ii = i4
    elif id == 5: ii = i5

    payload = {"configuration":{"label":ii[0],
                              "nodeType":"TAG",
                              "ble":False,
                              "leds":False,
                              "uwbFirmwareUpdate":False,
                              "tag":{"stationaryDetection":False,
                                     "responsive":True,
                                     "locationEngine":True,
                                     "nomUpdateRate":100,
                                     "statUpdateRate":100}}}
    # client.publish("dwm/node/{}/downlink/config".format(payload), json.dumps(ii[3]))
    return payload , ii[3]

#---------------------------------------------------------------------------------
def change_rule(x, y):
    ''' 
        rule_flag=1: Inside overlap, AreaNum not be changed.
        rule_flag=0: Outside overlap, can start to change AreaNum.
    '''
    global rule_flag
    global AreaNumRec
    global AreaNumTrue
    e= 1
    
    # Equations for the boundaries between areas
    # For 'a' and 'b': y = 0.9x
    # For 'c' and 'd': y = -0.9x + 18
    # For 'b' and 'c': y = 9
    # print('rule_flag = ', rule_flag)
    # print('AreaNumRec[:2] = ', AreaNumRec[:2])
            
    # The boundary between 'a' and 'b' is y = 0.9x
    if y > 0.9 * x - e and y < 0.9 * x + e:
        if 'a' in AreaNumRec[:2] or 'b' in AreaNumRec[:2]: 
            rule_flag = 1
        else:
            rule_flag = 0

    # The boundary between 'b' and 'c' is the horizontal line y = 9
    elif y > 9 - e and y < 9 + e:
        if 'c' in AreaNumRec[:2] or 'b' in AreaNumRec[:2]:
            rule_flag = 1
        else:
            rule_flag = 0

    # The boundary between 'c' and 'd' is y = -0.9x + 18
    elif y > -0.9 * x + 18 - e and y < -0.9 * x + 18 + e:
        if 'c' in AreaNumRec[:2] or 'd' in AreaNumRec[:2]: 
            rule_flag = 1
        else:
            rule_flag = 0
    else:
        rule_flag = 0

    if rule_flag == 0:
        if AreaNumTrue == 'a':
            if y > 0.9 * x:  # Crossing from 'a' to 'b'
                AreaNumTrue = 'b'
                print("change status a to b")
                rule_flag = 1
        elif AreaNumTrue == 'b':  
            if y < 0.9 * x:  # Crossing from 'b' to 'a'
                AreaNumTrue = 'a'
                print("change status b to a")
                rule_flag = 1
            elif y > 9:  # Crossing from 'b' to 'c'
                AreaNumTrue = 'c'
                print("change status b to c")
                rule_flag = 1
        elif AreaNumTrue == 'c':  
            if y < 9:  # Crossing from 'c' to 'b'
                AreaNumTrue = 'b'
                print("change status c to b")
                rule_flag = 1
            elif y > -0.9 * x + 18:  # Crossing from 'c' to 'd'
                AreaNumTrue = 'd'
                print("change status c to d")
                rule_flag = 1
        elif AreaNumTrue == 'd':
            if y < -0.9 * x + 18:  # Crossing from 'd' to 'c'
                AreaNumTrue = 'c'
                print("change status d to c")
                rule_flag = 1
        if AreaNumTrue != AreaNumRec[0]:
            AreaNumRec.insert(0, AreaNumTrue)
#---------------------------------------------------------------------------------

def reset_anchor(reset_index, AreaNumTrue, received_indices, requested_indices):

    print('Received indices: ', received_indices)
    print('Requested indices: ', requested_indices)
    
    client = mqtt.Client()
    client.connect(RASP_IP, 1883, 60) 

    received_indices = [i for i in received_indices.split(',') if i.isdigit()]
    # Determine wrong received indices
    wrong_received_indices = [i for i in received_indices if i not in requested_indices]

    remained_indices = [i for i in requested_indices if i not in received_indices]

    print('wrong_received_indices: ', wrong_received_indices)
    print('remained_indices: ', remained_indices)
    # Send A2T to wrong received indices
    for i in wrong_received_indices:
        
        p1, id1 = A2T(int(i))
        client.publish("dwm/node/{}/downlink/config".format(id1), json.dumps(p1))

    # Send T2A to requested indices
    for i in remained_indices:
        p1, id1 = T2A(int(i))
        client.publish("dwm/node/{}/downlink/config".format(id1), json.dumps(p1))



def activate_allanchors():
    print("activate all anchors")
    p1,id1 = T2A(0)
    client.publish("dwm/node/{}/downlink/config".format(id1), json.dumps(p1))
    p1,id1 = T2A(1)
    client.publish("dwm/node/{}/downlink/config".format(id1), json.dumps(p1))
    p1,id1 = T2A(2)
    client.publish("dwm/node/{}/downlink/config".format(id1), json.dumps(p1))
    p1,id1 = T2A(3)
    client.publish("dwm/node/{}/downlink/config".format(id1), json.dumps(p1))
    p1,id1 = T2A(4)
    client.publish("dwm/node/{}/downlink/config".format(id1), json.dumps(p1))
    p1,id1 = T2A(5)
    client.publish("dwm/node/{}/downlink/config".format(id1), json.dumps(p1))

    
    
def f1score(y_true, y_pred):
    # Convert predictions to one-hot encoding
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.one_hot(y_pred, depth=tf.shape(y_true)[-1])
    
    # Calculate True Positives, False Positives, and False Negatives
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    # Calculate Precision and Recall
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    # Calculate F1 Score
    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

    
import numpy as np
from collections import namedtuple
import paho.mqtt.client as mqtt
import base64
import json
import os
from threading import Thread
import threading
import tensorflow as tf
import time
from tensorflow.keras.models import load_model
from collections import deque
import joblib
import matplotlib.pyplot as plt
from pynput import keyboard

global reset_index, AreaNumTrue, received_indices, input_index,error_count

AreaNumRec = ['a']   # The area number before recording
AreaNumNow = 'a'   # Determined after judging by location
AreaNumTrue = 'a'  # Determined by anchor number
input_index={'a':['0','1','2'],'b':['3','2','1'],'c':['3','2','4'],'d':['5','4','2']}

reset_index={'a':[0,1,2, 3,4,5],'b':[1,2,3, 4,5,0],'c':[2,3,4, 5,0,1],'d':[2,4,5, 0,1,3]}

index_1,index_2,index_3 = None,None,None   # raw data to model input
rule_flag = 0 # is to change_rule function


client = mqtt.Client()
client.connect(RASP_IP, 1883, 60) 
activate_allanchors()
# reset_anchor(reset_index,AreaNumTrue)

print('pass0')


# Global variables
data = ''
data_m = ''
time_n = 0
numb_data = 1
d = ''
data_list = []
data_lock = threading.Lock()
new=0
error_count=0


print('pass1')
#==============================================================================================
# Constants

# model = load_model(f'{model_folder}/gesture_model.h5', custom_objects={'f1score': f1score})
scaler = joblib.load(f'{model_folder}/scaler.pkl')
encoder = joblib.load(f'{model_folder}/encoder.pkl')
print('pass2')
# Load the TensorRT optimized model
trt_model = tf.saved_model.load(f'{model_folder}-trt')


# Function to run inference
def run_inference(model, input_tensor):
    infer = model.signatures['serving_default']
    return infer(tf.convert_to_tensor(input_tensor, dtype=tf.float32))


print('warming up...')
start=time.time()
# Dummy input for warming up the model
dummy_input = np.random.random((1, seq_length, 3)).astype(np.float32)

# Warm up the model
run_inference(trt_model, dummy_input)
print('warm up done!')
print('warming duration : ', time.time()-start)

def preprocess_data(data):
    # Reshape and normalize data
    data = data.reshape(1, seq_length, 4)
    data = scaler.transform(data[0]).reshape(1, seq_length, 4)
    return data
    
def predict_gesture(window_data):
    global error_count
    error_count=0
    # predictions = []
    # Preprocess and reshape window data
    # window = deque(maxlen=seq_length)
    # window.append(window_data[0])
    window_data = np.array(window_data[0]).reshape(1, seq_length, 3)
    window_data = scaler.transform(window_data[0]).reshape(1, seq_length, 3)

    pred = run_inference(trt_model, window_data)

    # Extract the output tensor
    output_key = list(pred.keys())[0]  # Adjust this if necessary
    pred_array = pred[output_key].numpy()  # Convert the tensor to a numpy array
    
    pred_label = encoder.inverse_transform([np.argmax(pred_array)])
    # predictions.append(pred_label[0])

    return pred_label[0]
    
# Initialize a queue with fixed size equal to the sequence length
window = deque(maxlen=seq_length)


print('pass3')
#==============================================================================

# Flag to indicate when to exit the loop
exit_flag = False

def on_press(key):
    global exit_flag
    try:
        if key.char == 'q':  # If 'q' is pressed, set the exit flag
            exit_flag = True
            print("Exit key pressed")
            return False  # Stop the listener
    except AttributeError:
        pass  # Ignore special keys (e.g., Shift, Ctrl)

# Start the listener in a separate thread
listener = keyboard.Listener(on_press=on_press)
listener.start()


# Define the on_message callback
def on_message(client, userdata, message):
    global data, data_m, numb_data, d, data_list,data_lock, new, error_count, xy_coord
    global reset_index, AreaNumTrue, received_indices, input_index, error_count
    global last_msg_time
    last_msg_time=time.time()
    
    # data_m = base64.b64decode(json.loads(message.payload.decode("utf-8"))["data"])
    # data_m = data_m.decode("utf-8").replace("\n", '')

    # Decode the entire payload from Base64
    raw_data = base64.b64decode(json.loads(message.payload.decode("utf-8"))["data"])
    data_text = raw_data.decode("utf-8").replace("\n", '')
    # print(data_text)
    topic = message.topic.split("/")[2]

    # Split the data at ';'
    parts = data_text.split(';')
    if len(parts) < 2:
        print("Received data in unexpected format.")
        return

    # First part is assumed to be CSV of integers
    pos_data = parts[0].split(',')
    # print(pos_data)

    # Second part contains index, encoded data pairs
    encoded_segments = parts[1].split(',')

    base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    decoded_message=''
    gesture_input=''
    received_indices=''
    if len(pos_data) > 1:
        xy_coord=[0,0]
        i = 0
        while i < len(pos_data):
            index = pos_data[i]
            if i < len(pos_data):
                encoded_data = pos_data[i]

                # Check for four-character encoding and remove duplicated characters
                if len(encoded_data) == 4 :
                    print(encoded_data)
                    if encoded_data[0] == encoded_data[1]:
                        print('dup1')
                        encoded_data = encoded_data[1:]  # Remove the first of the duplicated characters
                        print(encoded_data)
                    elif encoded_data[1] == encoded_data[2]:
                        print('dup2')
                        encoded_data = list(encoded_data)
                        encoded_data[2] = encoded_data[3]
                        encoded_data = ''.join(encoded_data[:3])
                        # encoded_data = encoded_data[:3]
                        print(encoded_data)

                    elif encoded_data[2] == encoded_data[3]:
                        print('dup3')
                        encoded_data = encoded_data[:3]
                        print(encoded_data)

                
                num = 0
                # Reverse the order of characters as per the encoding logic
                for char in reversed(encoded_data):
                    num = (num << 6) + base64_chars.index(char)
                num = num-130000
                decoded_message += f"{num},"
                xy_coord[i]=num
            i += 1
        decoded_message = decoded_message.rstrip(',')
        # print(xy_coord)
        change_rule(xy_coord[0]/1000,xy_coord[1]/1000)

    if len(encoded_segments) > 1:
        decoded_message += ";"
        i = 0
        while i < len(encoded_segments):
            index = encoded_segments[i]
            if i + 1 < len(encoded_segments):
                encoded_data = encoded_segments[i + 1]
                # Check for four-character encoding and remove duplicated characters
                if len(encoded_data) == 4 :
                    print(encoded_data)
                    if encoded_data[0] == encoded_data[1]:
                        print('dup1')
                        encoded_data = encoded_data[1:]  # Remove the first of the duplicated characters
                        print(encoded_data)
                    elif encoded_data[1] == encoded_data[2]:
                        print('dup2')
                        encoded_data = list(encoded_data)
                        encoded_data[2] = encoded_data[3]
                        encoded_data = ''.join(encoded_data[:3])
                        # encoded_data = encoded_data[:3]
                        print(encoded_data)

                    elif encoded_data[2] == encoded_data[3]:
                        print('dup3')
                        encoded_data = encoded_data[:3]
                        print(encoded_data)
                    
                num = 0
                # Reverse the order of characters as per the encoding logic
                for char in reversed(encoded_data):
                    num = (num << 6) + base64_chars.index(char)
                num = num-130000
                if num>30000 or num<-30000:
                    print(encoded_data)
                    num=0
                decoded_message += f"{index},{num},"
                received_indices+= f"{index},"
                gesture_input += f"{index},{num},"
            i += 2
        decoded_message = decoded_message.rstrip(',')
        received_indices = received_indices.rstrip(',')
        gesture_input = gesture_input.rstrip(',')

    # print(f"Received: {decoded_message}")
    # print(f"gesture_input: {gesture_input}")
    # print(f"process: {(time.perf_counter()-start)*1_000_000}")

    new1 = gesture_input.split(",")
    
    try:
        count0 = new1.index(input_index[AreaNumTrue][0])
        count1 = new1.index(input_index[AreaNumTrue][1])
        count2 = new1.index(input_index[AreaNumTrue][2])
        d = new1[count0+1] + ' ' + new1[count1+1] + ' ' + new1[count2+1] 
        new=1

        if ID == topic:# and numb_data <= num:
            data = data + d + '\n'
            len_data = len(data.split('\n')[0].split(' '))
            if anchor_num*numb_tag == len_data:
                numb_data += 1
                with data_lock:
                    data_list.append(d)
                    if len(data_list) > seq_length:
                        data_list.pop(0)
                    data = ''
                    d = ''
            else:
                print(f'Length mismatch: expected {anchor_num*numb_tag}, got {len_data}')
    except ValueError:
        error_count+=1
        print(f"Received: {decoded_message}")
        print(f"Request index: {input_index[AreaNumTrue]}")
        print(f"error_count: {error_count}")
        print("-----Bridge node receive error-----\n")
        # if error_count>10:
        #     # reset_anchor(reset_index,AreaNumTrue)
        #     # reset_anchor(reset_index, AreaNumTrue, received_indices, input_index[AreaNumTrue])
        #     error_count=0

# Setup and start the MQTT client
client = mqtt.Client()
client.connect(RASP_IP, 1883, 2)
client.on_message = on_message
client.subscribe(f"dwm/node/{ID}/uplink/data")
print('pass4')
# Start the client loop in a separate thread
def start_client():
    client.loop_start()

client_thread  = Thread(target=start_client)
client_thread .start()
print('mqtt thread start')

print('pass5')


# Initialize the box to store the last box_length=15 prediction results
box = deque(maxlen=box_length)
# Initialize the trigger state
trigger_state = 0
all_data = []  # List to store all data for logging
frame_number = 0  # Initialize frame number
thres= 7
detection='none'
gesture='0'

last_msg_time=time.time()
last_show_time = time.time()



#==================================DRONE================================
from dronekit import connect , VehicleMode , LocationGlobalRelative 
import time 
from datetime import datetime
from pymavlink import mavutil 

vehicle = connect('/dev/ttyACM0',rate=30, wait_ready=False)
vehicle.mode = VehicleMode('GUIDED')
def arm():
  #  while not vehicle.is_armable:
 #       print(" Waiting for vehicle to initialise...")
#        time.sleep(1)

    print("Arming motors")
    # Copter should arm in GUIDED mode
    vehicle.armed = True
    while not vehicle.armed:
        print(" Waiting for arming...")
        time.sleep(1)

def takeoff(aTargetAltitude):
    print("Taking off!")
    vehicle.simple_takeoff(aTargetAltitude)  # Take off to target altitude

    # Wait until the vehicle reaches a safe height before processing the goto
    #  (otherwise the command after Vehicle.simple_takeoff will execute
    #   immediately).
    while True:
        print(" Altitude: ", vehicle.rangefinder.distance)
        # Break and return from function just below target altitude.
        if vehicle.rangefinder.distance >= aTargetAltitude * 0.95:
            print("Reached target altitude")
            break
        time.sleep(1)

def arm_and_takeoff(aTargetAltitude):
    """
    Arms vehicle and fly to aTargetAltitude.
    """
    arm()
    takeoff(aTargetAltitude)

def goto_position_target_local_ned(forward, right, down):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED, # Frame
        0b000000000000, # type_mask (only positions enabled)
        forward, right, down, # x, y, z positions (or North, East, Down in the MAV_FRAME_BODY_NED frame
        0, 0, 0, # x, y, z velocity in m/s  (not used)
        0, 0, 0, # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)    # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink) 
    # send command to vehicle
    vehicle.send_mavlink(msg)

def condition_yaw(heading,direction=1, relative=False):
    if relative:
        is_relative = 1 #yaw relative to direction of travel
    else:
        is_relative = 0 #yaw is an absolute angle
    # create the CONDITION_YAW command using command_long_encode()
    msg = vehicle.message_factory.command_long_encode(
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_CMD_CONDITION_YAW, #command
        0, #confirmation
        heading,    # param 1, yaw in degrees
        0,          # param 2, yaw speed deg/s
        direction,          # param 3, direction -1 ccw, 1 cw
        is_relative, # param 4, relative offset 1, absolute angle 0
        0, 0, 0)    # param 5 ~ 7 not used
    # send command to vehicle
    vehicle.send_mavlink(msg)
    
arm()
motor_takeoff=0
time.sleep(2)

pos_x=0
pos_y=0
pos_z=0
angle=0
travel_dist=1

#=======================================================================



while not exit_flag:

    
    with data_lock:
        if (len(data_list) == seq_length):

                # Predict the gesture
            data_array = np.array([np.array(item.split(), dtype=float) for item in data_list])
            window.append(data_array)
            if new==1:
                new=0
                ti0=time.time()
                gesture = predict_gesture(window)
                # print(gesture)
                ti=time.time() - ti0
                # Append the predicted gesture to the box
                box.append(gesture)
                # print(box)

                ad = f'{pos_x} {pos_y} {pos_z} {angle} {xy_coord[0]} {xy_coord[1]} {ti:.3f} {gesture} {frame_number} ' + data_list[-1]
                all_data.append(ad)  # Add the processed data to all_data list for logging

                # Count occurrences of each gesture in the box
                counts = {label: box.count(label) for label in ['idle', 'A', 'B', 'X', 'Z', 'eight', 'nine']}
                # Check for trigger conditions
                if counts['idle'] > thres:
                    trigger_state = 0  # Reset trigger state if more than 10 idle gestures are in the box
                    detection='idle'
                    
                if trigger_state == 0 and any(counts[label] > thres for label in ['A', 'B', 'X', 'Z', 'eight', 'nine']):
                    # Process non-idle gestures if trigger state is 0
                    if counts['A'] > thres:
                        print('gesture "A" triggered')
                        detection='A'
                        if motor_takeoff==0:
                            motor_takeoff=1
                            takeoff(0.7)
                            pos_y+=0.7
                        else:
                            print('increase altitude 0.5 meters')
                            goto_position_target_local_ned(0, 0, -0.5)
                            pos_y+=0.5
                            
                        
                    elif counts['B'] > thres:
                        print('gesture "B" triggered')                        
                        detection='B'
                        if pos_y>1:
                            print('decrease altitude 0.5 meters')
                            goto_position_target_local_ned(0, 0, 0.5)
                            pos_y-=0.5
                        else:
                            print('landing')
                            vehicle.mode = VehicleMode("LAND")
                            motor_takeoff=0
                            pos_y=0
                    elif counts['X'] > thres:
                        print('gesture "X" triggered')
                        detection='X'
                        print(f'move forward {travel_dist} meters')
                        goto_position_target_local_ned(travel_dist, 0, 0)
                        pos_z+=travel_dist*np.round(np.cos(np.radians(angle)),5)
                        pos_x+=travel_dist*np.round(np.sin(np.radians(angle)),5)
                    elif counts['Z'] > thres:
                        print('gesture "Z" triggered')
                        detection='Z'
                        print(f'move backward {travel_dist} meters')
                        goto_position_target_local_ned(-travel_dist, 0, 0)
                        pos_z-=travel_dist*np.round(np.cos(np.radians(angle)), 5)
                        pos_x-=travel_dist*np.round(np.sin(np.radians(angle)), 5)
                    elif counts['eight'] > thres:
                        print('gesture "eight" triggered')
                        detection='eight'
                        print('rotate clockwise 90 degree')
                        condition_yaw(90,direction=1, relative=True)
                        angle+=90
                    elif counts['nine'] > thres:
                        print('gesture "nine" triggered')
                        detection='nine'
                        print('rotate counter-clockwise 90 degree')
                        condition_yaw(90,direction=-1, relative=True)
                        angle-=90
                    trigger_state = 1  # Change trigger state to prevent re-triggering
                    client.connect(RASP_IP, 1883, 2)
                    client.subscribe(f"dwm/node/{ID}/uplink/data")

    
    if error_count>10:
        reset_anchor(reset_index, AreaNumTrue, received_indices, input_index[AreaNumTrue])
        error_count=0


    last_msg_diff=time.time()-last_msg_time
    if last_msg_diff>5:
        print(last_msg_diff)
        print('no new data for 5 seconds')
        last_msg_time=time.time()

    if time.time()-last_show_time>1:
        print(gesture)
        print('sub-region: ',AreaNumTrue) 
        last_show_time=time.time()



    # exit
    # if keyboard.is_pressed('q'):  # Listen for 'q' key press to break the loop
        # Save all collected and processed data to a text file
with open(f"{filename}.txt", "w") as file:
    for line in all_data:
        file.write(f"{line}\n")
print("Data saved to file")
activate_allanchors()
client.loop_stop()

vehicle.mode = VehicleMode("LAND")
motor_takeoff=0
time.sleep(3)
vehicle.close()
print("program closed")
        # break

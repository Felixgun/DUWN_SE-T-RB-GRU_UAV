# Set your parameters here
ID = "0870" #tag ID
seq_length = 30  # Length of sequence
filename = "test"
anchor_num = 3
numb_tag = 1
model_folder = "run9"
RASP_IP = '192.168.155.208'

import matplotlib


xy_coord = [0, 0]

# Function to create the map with the current position
def mymap(x, y):
    pillar_x1 = [-0.45, 0.45, 0.45, -0.45, -0.45]
    pillar_y1 = [0.35, 0.35, -0.35, -0.35, 0.35]
    pillar_x2 = [-0.45, 0.45, 0.45, -0.45, -0.45]
    pillar_y2 = [6.65, 6.65, 5.95, 5.95, 6.65]
    A1 = [4, -3.15]
    A2 = [-4, -3.15]
    A3 = [0.45, 0.35]
    A4 = [-0.45, -0.35]
    A5 = [4, 3.15]
    A6 = [-4, 3.15]
    A7 = [0.45, 6.65]
    A8 = [-0.45, 5.95]
    A9 = [4, 9.45]
    A10 = [-4, 9.45]

    plt.figure(figsize=(5, 8))
    plt.plot(pillar_x1, pillar_y1, 'k-'), plt.plot(pillar_x2, pillar_y2, 'k-')
    plt.plot(A1[0], A1[1], 'r^', markersize=13, label='Anchor')
    plt.plot(A2[0], A2[1], 'r^', markersize=13)
    plt.plot(A3[0], A3[1], 'r^', markersize=13)
    plt.plot(A4[0], A4[1], 'r^', markersize=13)
    plt.plot(A5[0], A5[1], 'r^', markersize=13)
    plt.plot(A6[0], A6[1], 'r^', markersize=13)
    plt.plot(A7[0], A7[1], 'r^', markersize=13)
    plt.plot(A8[0], A8[1], 'r^', markersize=13)
    plt.plot(A9[0], A9[1], 'r^', markersize=13)
    plt.plot(A10[0], A10[1], 'r^', markersize=13)
    # plt.plot(12, 12, 'ks', mfc="None", markersize=13, label='Pillar')
    plt.plot(x, y, 'bo', markersize=13, label='Current Position')  # Plot current position
    plt.xlabel('X(m)', fontsize=15, fontweight="bold"), plt.ylabel('Y(m)', fontsize=15, fontweight="bold")
    plt.legend()
    plt.grid()
    
    # Convert the plot to an image
    fig = plt.gcf()
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    plt.close(fig)  # Close the figure to release memory
    return img

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
import cv2
import time
from tensorflow.keras.models import load_model
from collections import deque
import joblib
import matplotlib.pyplot as plt

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

model = load_model(f'{model_folder}/gesture_model.h5', custom_objects={'f1score': f1score})
scaler = joblib.load(f'{model_folder}/scaler.pkl')
encoder = joblib.load(f'{model_folder}/encoder.pkl')
print('pass2')

# def preprocess_data(data):
#     # Reshape and normalize data
#     data = data.reshape(1, seq_length, 3)
#     data = scaler.transform(data[0]).reshape(1, seq_length, 3)
#     return data

def predict_gesture(window_data):
    global error_count
    error_count=0
    # Preprocess and reshape window data
    # window = deque(maxlen=seq_length)
    # window.append(window_data[0])
    window_data = np.array(window_data[0]).reshape(1, seq_length, 3)
    window_data = scaler.transform(window_data[0]).reshape(1, seq_length, 3)

    # Make a prediction
    pred = model.predict(window_data, verbose=False)
    pred_label = encoder.inverse_transform([np.argmax(pred)])
    return pred_label[0]
# Initialize a queue with fixed size equal to the sequence length
window = deque(maxlen=seq_length)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter(f'{filename}.mp4', fourcc, 20.0, (940, 480))
out = cv2.VideoWriter(f'{filename}.mp4', fourcc, 20.0, (640,  480))
# OpenCV camera setup
cap = cv2.VideoCapture(0)  # Use 0 for the default camera
print('pass3')
#==============================================================================

# Define the on_message callback
def on_message(client, userdata, message):
    global data, data_m, numb_data, d, data_list,data_lock, new, error_count, xy_coord
    global reset_index, AreaNumTrue, received_indices, input_index, error_count
    
    
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

thread = Thread(target=start_client)
thread.start()
print('pass5')
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 0, 0)  # Blue color
thickness = 2
text_x = 50
text_y = 50
text_x2 = 350
text_y2 = 50
text_x3 = 350
text_y3 = 75

# Define properties for displaying the data list
data_text_x = 20  # X-coordinate for data text (left side of the frame)
data_text_y_start = 70  # Starting Y-coordinate for data text
data_text_y_offset = 10  # Vertical space between lines
# Define the box coordinates
box_coords4 = ((data_text_x, data_text_y_start-10), (data_text_x + 90, data_text_y_start + seq_length*10))
# Create a filled rectangle (box) with a specific transparency (alpha)
box_color = (255, 255, 255) 
alpha = 0.4  # Transparency factor

# Initialize the box to store the last 15 prediction results
box = deque(maxlen=15)
# Initialize the trigger state
trigger_state = 0
all_data = []  # List to store all data for logging
frame_number = 0  # Initialize frame number
thres= 7
detection='none'
gesture='0'
last_frame_time = time.time()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    frame=cv2.flip(frame,1)
    with data_lock:
        if (len(data_list) == seq_length):

                # Predict the gesture
            data_array = np.array([np.array(item.split(), dtype=float) for item in data_list])
            window.append(data_array)
            if new==1:
                new=0
                ti0=time.time()
                gesture = predict_gesture(window)
                ti=time.time() - ti0
                # Append the predicted gesture to the box
                box.append(gesture)
                # print(box)

                ad = f'{xy_coord[0]} {xy_coord[1]} {ti:.3f} {gesture} {frame_number} ' + data_list[-1]
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
                    elif counts['B'] > thres:
                        print('gesture "B" triggered')
                        detection='B'
                    elif counts['X'] > thres:
                        print('gesture "X" triggered')
                        detection='X'
                    elif counts['Z'] > thres:
                        print('gesture "Z" triggered')
                        detection='Z'
                    elif counts['eight'] > thres:
                        print('gesture "eight" triggered')
                        detection='eight'
                    elif counts['nine'] > thres:
                        print('gesture "nine" triggered')
                        detection='nine'
                    trigger_state = 1  # Change trigger state to prevent re-triggering

    
    if error_count>10:
        reset_anchor(reset_index, AreaNumTrue, received_indices, input_index[AreaNumTrue])
        error_count=0
        
    # Create a copy of the frame to draw the box
    overlay = frame.copy()
    
    # Draw the filled rectangle on the overlay
    cv2.rectangle(overlay, box_coords4[0], box_coords4[1], box_color, cv2.FILLED)

    # Blend the overlay with the original frame
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    for idx, data_disp in enumerate(data_list):
        cv2.putText(frame, data_disp, (data_text_x, data_text_y_start + idx * data_text_y_offset), 
                    font, 0.3, (0, 0, 0), 1)
    # Define text and box properties
    # text = f"detection: {detection}"
    text = f"gesture: {gesture}"
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    box_coords = ((text_x, text_y + 5), (text_x + text_size[0], text_y - text_size[1] - 5))
    

    text3 = f"sub-area: {AreaNumTrue}"
    text_size3 = cv2.getTextSize(text3, font, font_scale, thickness)[0]
    box_coords2 = ((text_x2, text_y2 + 5), (text_x2 + text_size3[0], text_y2 - text_size3[1] - 5))
    box_coords3 = ((text_x3, text_y3 + 5), (text_x3 + text_size3[0], text_y3 - text_size3[1] - 5))

    # Draw the box
    cv2.rectangle(frame, box_coords[0], box_coords[1], (0, 255, 0), cv2.FILLED)
    # Display the gesture on the frame
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, thickness)
    
    # Draw the box
    cv2.rectangle(frame, box_coords2[0], box_coords2[1], (0, 255, 0), cv2.FILLED)
    
    # Draw the box
    cv2.rectangle(frame, box_coords3[0], box_coords3[1], (0, 255, 0), cv2.FILLED)
    # Display the gesture on the frame
    cv2.putText(frame, text3, (text_x3, text_y3), font, font_scale, font_color, thickness)

    # FPS Limiter ===============
    # current_time = time.time()
    # fps0 = current_time - last_frame_time
    # if fps0<0.08:
    #     time.sleep(0.08-fps0)
    #============================
    
    #FPS count ==================
    # current_time = time.time()
    # fps0 = current_time - last_frame_time
    # fps = 1.0 / fps0
    # last_frame_time = current_time
    # fps_text = f"FPS: {fps:.2f}"  # Format the FPS to two decimal places
    #============================
    
    fps_text = f"frame: {frame_number}"
    
    # Display the fps on the frame
    cv2.putText(frame, str(fps_text), (text_x2, text_y2), font, font_scale, font_color, thickness)
    '''
    # Create the map image
    map_img = mymap(xy_coord[0] / 1000, xy_coord[1] / 1000)

    # Resize the map image to match the video frame size
    map_img = cv2.resize(map_img, (300, 480))

    # Combine the video frame and the map image side by side
    combined_img = np.hstack((frame, map_img))
    '''
    combined_img = frame

    # Write the combined image to the output video
    out.write(combined_img)

    # Display the combined image
    cv2.imshow('Frame with Map', combined_img)
    frame_number += 1


    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Save all collected and processed data to a text file
        with open(f"{filename}.txt", "w") as file:
            for line in all_data:
                file.write(f"{line}\n")
        print("Data saved to file")
        activate_allanchors()
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        client.loop_stop()
        break

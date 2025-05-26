from dronekit import connect , VehicleMode , LocationGlobalRelative 
import time 
from datetime import datetime
from pymavlink import mavutil 

vehicle = connect('/dev/ttyACM0',rate=30, wait_ready=True)
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
time.sleep(3)
print("takeoff")
takeoff(0.7)
time.sleep(5)

print("forward")
goto_position_target_local_ned(3, 0, 0)
time.sleep(10)

print("counter-clockwise")
condition_yaw(90,direction=-1, relative=True)
time.sleep(5)



print("forward")
goto_position_target_local_ned(3, 0, 0)
time.sleep(10)


print("clockwise")
condition_yaw(90,direction=1, relative=True)
time.sleep(5)


print("backward")
goto_position_target_local_ned(-3, 0, 0)
time.sleep(10)

print("counter-clockwise")
condition_yaw(90,direction=-1, relative=True)
time.sleep(5)


print("backward")
goto_position_target_local_ned(-3, 0, 0)
time.sleep(10)

print("clockwise")
condition_yaw(90,direction=1, relative=True)
time.sleep(5)

vehicle.mode = VehicleMode("LAND")

time.sleep(1)

vehicle.close()

print("Completed")

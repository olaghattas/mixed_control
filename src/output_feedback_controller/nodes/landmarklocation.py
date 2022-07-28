#!/usr/bin/env python

#import the dependencies 
import rospy
from gazebo_msgs.srv import GetModelState, GetModelStateRequest

position_landmark_inworld_matrix = {}   
used_apriltags = [1,2,3,4,5,6,7,8,9,11,12,13,14,15]

rospy.wait_for_service('/gazebo/get_model_state')
get_model_srv = rospy.ServiceProxy('/gazebo/get_model_state',GetModelState)
model = GetModelStateRequest()
for id in used_apriltags:
    model.model_name = 'apriltag'+str(id)
    result = get_model_srv(model)
    position_landmark_inworld_matrix[id] = result.pose
print(position_landmark_inworld_matrix)
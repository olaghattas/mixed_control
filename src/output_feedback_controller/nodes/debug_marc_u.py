#!/usr/bin/env python

#import the dependencies 
import rospy
from geometry_msgs.msg import Pose, Point, Twist
from nav_msgs.msg import Odometry
import transformation_utilities as tu
import numpy as np
import os

def marc_u(apriltag_distance, K_matrix,position_landmark_inworld_matrix,selected_aptag):
    _idx = 0
    u = 0
    print('K_matrix',K_matrix)
    for landmark_id in position_landmark_inworld_matrix.keys():
        k = K_matrix[:,_idx*2:_idx*2+2]
       
        if landmark_id == selected_aptag:
            for _z in range(K_matrix.shape[1]/2):
                # print('_z',_z)
                # print('K_matrix[:,_z*2:_z*2+2]',K_matrix[:,_z*2:_z*2+2])
                u += K_matrix[:,_z*2:_z*2+2].dot(apriltag_distance)
        else:
            # We compute another term of the bias:
            u += np.dot(k,position_landmark_inworld_matrix[landmark_id][:2,3] - position_landmark_inworld_matrix[selected_aptag][:2,3]).reshape((2,-1))
            print('lj-li',position_landmark_inworld_matrix[landmark_id][:2,3] - position_landmark_inworld_matrix[selected_aptag][:2,3])
        _idx += 1

        # Finally, go back to global coordinates:
        # u = self.transform_matrix[:2,:2].dot(u.flatten())
    u = np.concatenaate((u.flatten(),[0]))
    print('marc',u)

def ola_u(K_gains,selected_id,selected_aptag,position_landmark_inworld_matrix,dist_vector_state):
    ### eq u = sum_j [K_j(l_i - x)] + sum_j [K_j(l_j - li)]
    
    # apriltag 0 and 10 were not used 
    if selected_id < 10:
        split_idx = selected_id - 1
    else:
        split_idx = selected_id

    ## j is != i where i is the selected apriltag
    K_j = np.hstack((K_gains[:,:split_idx],K_gains[:,split_idx+2:]))

    u_1 = None
    for i in range(np.size(K_j,1)/2):
        if u_1 is None:
            u_1 = (K_j[:,i*2:i*2+2]*dist_vector_state)
        else:
            u_1  = np.hstack((u_1,(K_gains[:,i*2:i*2+2]*dist_vector_state)))
    u_1 = u_1.sum(axis = 1)

    ### 2nd term in u
    x_i = position_landmark_inworld_matrix[selected_id][0,3]
    y_i = position_landmark_inworld_matrix[selected_id][1,3]
    lj_li = None

    for key, value in position_landmark_inworld_matrix.items():

        if key is not selected_id:
            x = value[0,3] - x_i
            y = value[1,3] - y_i
            # print("x",x,"y",y)
            if lj_li is None:
                lj_li = np.vstack((x,y))
                
            else:
                lj_li = np.vstack((lj_li,x,y))
                   
    u_2 = np.dot(K_j,lj_li).sum(axis=1)
    print('k_j',K_j,'lj-li',lj-li)
    u = u_1 + u_2
    print('u_ola',u)

def qt_to_matrix( position_landmark_inworld):
    position_landmark_inworld_matrix = {}
    for id in position_landmark_inworld.keys():
            pose = Pose()
            pose.position.x = position_landmark_inworld[id][0]
            pose.position.y = position_landmark_inworld[id][1]
            pose.position.z = position_landmark_inworld[id][2]
            pose.orientation.x = position_landmark_inworld[id][3]
            pose.orientation.y = position_landmark_inworld[id][4]
            pose.orientation.z = position_landmark_inworld[id][5]
            pose.orientation.w = position_landmark_inworld[id][6]

            position_landmark_inworld_matrix[id] = tu.msg_to_se3(pose)

    return position_landmark_inworld_matrix

def read_matrix(csv_dir):
    with open(csv_dir,'r') as f:
        return np.genfromtxt(f,delimiter=',')

position_landmark_inworld = {1:[-8.24439, -5.8941, 0.5, 0, 0, 0, 1.0],
                            2:[-8.27337, -1.68115, 0.5, 0, 0, 0, 1.0],
                            3:[-8.23126, 1.44051, 0.5, 0, 0, 0, 1.0],
                            4:[-5.65321, 3.03627, 0.5, 0, 0, -0.717185376228,0.696882440678],
                            5:[-2.52459, 3.22243, 0.5, 0, 0, -0.699616049545, 0.714518987305],
                            6:[1.31312, -2.15597, 0.5, 0, 0, -0.999693206373, 0.0247687935147],
                            7:[1.45869, -6.20846, 0.5, 0, 0,0.99999418143,0.00341132017855],
                            8:[-2.46765, -8.80024, 0.5, 0, 0, 0.705986429932,0.708225360145],
                            9:[-5.70343, -8.65199, 0.5, 0, 0, 0.692161786123,0.721742379129],
                            11:[-4.10288, -5.19704, 0.5, 0, 0, 0.999988854557, 0.00472130925446],
                            12:[-4.10288, -2.02095, 0.5, 0, 0,-0.999944861574, 0.0105011337853],
                            13:[-1.98548, -2.07486, 0.5, 0, 0, 0, 1.0],
                            14:[-1.93215, -4.18616, 0.5, 0, 0, 0, 1.0],
                            15:[1.22551, 1.92208, 0.5, 0, 0, 0.999930365151,0.0118010528658]
                            }

home_dir = os.environ["HOME"]
shared_path = os.environ["HOME"]+"/catkin_ws/src/output_feedback_controller/csv/"
K_gains_path= shared_path + "K_gains_new.csv"
K_gains = read_matrix(K_gains_path)

# K_added_path= shared_path + "K_added_new.csv"
# K_added = read_matrix(K_added_path)
k_index = 2 #seen aptag
K_gains = K_gains[k_index*2:k_index*2+2,:]
# distance from aptag_robot dummy
aptag_dista = np.array([[2],[5]] )
position_landmark_inworld_matrix = qt_to_matrix(position_landmark_inworld)
marc_u(aptag_dista,K_gains,position_landmark_inworld_matrix,k_index)
ola_u(K_gains,k_index, position_landmark_inworld_matrix[k_index],position_landmark_inworld_matrix,aptag_dista)
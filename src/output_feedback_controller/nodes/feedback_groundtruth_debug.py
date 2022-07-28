#!/usr/bin/env python

#import the dependencies
from __future__ import print_function
from re import U
from types import NoneType

from pytz import NonExistentTimeError
 
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from apriltag_ros.msg import AprilTagDetectionArray
import transformation_utilities as tu
import numpy as np
import tf2_ros 
import tf2_geometry_msgs
import os
from std_msgs.msg import Float64MultiArray,Float64
import pandas as pd
from gazebo_msgs.srv import GetModelState, GetModelStateRequest


class Jackal:
    def __init__(self,K_gains,K_added): 
        self.pub=rospy.Publisher('u_input',Float64MultiArray,queue_size=1)
        self.linear_vel =  rospy.Subscriber("/linear_vel", Float64, self.linear_vel_callback)
        self.sub_img_detec =  rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self.apriltag_callback)
        self.vel = Twist()
        self.sub_odom =  rospy.Subscriber("/my_odom", Odometry, self.odom_callback)
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.apriltags_list = list()
        self.K_gains = np.array(K_gains)
        self.K_added = np.array(K_added)
        self.k_index = 0
        self.selected_apriltag = None
        self.selected_aptag_id = None
        self.position_landmark_inworld = {1:[-8.24439, -5.8941, 0.5, 0, 0, 0,],
                                          2:[-8.27337, -1.68115, 0.5, 0, 0, 0],
                                          3:[-8.23126, 1.44051, 0.5, 0, 0, 0],
                                          4:[-5.65321, 3.03627, 0.5, 0, 0, -1.59951],
                                          5:[-2.52459, 3.22243, 0.5, 0, 0, -1.54972],
                                          6:[1.31312, -2.15597, 0.5, 0, 0, -3.09205],
                                          7:[1.45869, -6.20846, 0.5, 0, 0, 3.13477],
                                          8:[-2.46765, -8.80024, 0.5, 0, 0, 1.56763],
                                          9:[-5.70343, -8.65199, 0.5, 0, 0, 1.52896],
                                          11:[-4.49279, 0.886215, 0.5, 0, 0, 0],
                                          12:[-4.10288, -2.02095, 0.5, 0, 0, -3.12059],
                                          13:[-1.98548, -2.07486, 0.5, 0, 0, 0],
                                          14:[-1.93215, -4.18616, 0.5, 0, 0, 0],
                                          15:[1.22551, 1.92208, 0.5, 0, 0, 3.11799]
                                         }
        self.position_landmark_inworld_matrix = {}
        self.x_position = None
        self.y_position = None
        self.used_apriltags = [1,2,3,4,5,6,7,8,9,11,12,13,14,15]
    
    def get_rot_matrix_aptags(self):
        rospy.wait_for_service('/gazebo/get_model_state')
        get_model_srv = rospy.ServiceProxy('/gazebo/get_model_state',GetModelState)
        model = GetModelStateRequest()
        for id in self.used_apriltags:
            model.model_name = 'apriltag'+str(id)
            result = get_model_srv(model)
            self.position_landmark_inworld_matrix[id] = tu.msg_to_se3(result.pose)
        # print(self.position_landmark_inworld_matrix)

    def odom_callback(self,msg):
        if msg:
            self.x_position = msg.pose.pose.position.x
            self.y_position = msg.pose.pose.position.y
            # print(self.x_position,self.y_position)

    def dist_in_world_coor(self,at,at_id):
        # at is tag.pose.pose.pose
           
        # measured displacement of aptag in robot coordinates 
        disp_robcoor= at[:3,3]

        # measured orientation of apriltag wrt robot 
        orientation_aptag_wrt_rob = at[:3,:3]

        #orientation of aptag wrt world frame
        orientation_aptag_wrt_world=self.euler_to_rotmatrix(self.position_landmark_inworld[at_id][3:])
        # print('orientation_aptag_wrt_world', orientation_aptag_wrt_world)

        # rotation matrix of robot in world coordinates WRAT * JRAT-1 = WRJ
        rotation = np.dot(orientation_aptag_wrt_world,np.linalg.inv(orientation_aptag_wrt_rob))
        
        # disp is the measured displacement of the apriltag wrt robot in world coordinates: WRJ * JTAT 
        disp = np.dot(rotation, disp_robcoor)


        return disp[:2]
  

    def apriltag_callback(self,msg):
        if msg.detections:
            # '''If there's an AprilTag in the image'''
            min_distance = np.inf
            for at in msg.detections:
                dist = np.linalg.norm([at.pose.pose.pose.position.x, at.pose.pose.pose.position.y, at.pose.pose.pose.position.z])
                if dist < min_distance:
                    min_distance = dist
                    selected_aptag_id = at.id[0]
                    selected_apriltag = at.pose.pose

                #change frame from camera to baselink
                source_frame = "front_realsense_gazebo"
                transform = self.tfBuffer.lookup_transform("base_link", source_frame, rospy.Time(0), rospy.Duration(1.0))
                pose_transformed = tf2_geometry_msgs.do_transform_pose(selected_apriltag, transform)
                # print("posetrans",selected_aptag_id,pose_transformed)

                self.selected_apriltag = tu.msg_to_se3(pose_transformed.pose)
                # print('msgquat',pose_transformed.pose)
                self.selected_aptag_id = selected_aptag_id 
                # print('sel aptag',self.selected_apriltag)
                # print('sel aptag id',self.selected_aptag_id)
        else:
            rospy.logwarn("Can't find an AprilTag in the image!")
            self.selected_apriltag = None
            self.selected_aptag_id = None


## to update the k_index
    def linear_vel_callback(self,msg):
        if msg:
            threshold = 0.02
            if msg< threshold:
                self.k_index += 1

    
    def euler_to_rotmatrix(self,euler_angle):
        #########
        #   The full rotation matrix for the elemental rotation order yaw-pitch-roll 
        #   (u, v, w) are the three Euler angles (roll, pitch, yaw), corresponding to rotations around the x, y and z axes 
        #   c and s are shorthand for cosine and sine
        #########
        euler_angle = np.array(euler_angle).T
        c_u = np.cos(euler_angle[0])
        s_u = np.sin(euler_angle[0])
        c_v = np.cos(euler_angle[1])
        s_v = np.sin(euler_angle[1])
        c_w = np.cos(euler_angle[2])
        s_w = np.sin(euler_angle[2])
        matrix = np.array([[c_v*c_w, s_u*s_v*c_w-c_u*s_w, s_u*s_w+c_u*s_v*c_w],\
        [c_v*s_w, c_u*c_w+s_u*s_v*s_w, c_u*s_v*s_w-s_u*c_w],\
        [-s_v, s_u*c_v, c_u*c_v]])
        return matrix

## compute u using only one apriltag then check how you can include multipe apriltags

    def compute_u(self): # n = num of aptags seen K=[2n,?] y = [2n,1]
        ### eq u = sum_j [K_j(l_i - x)] + sum_j [K_j(l_j - li)]
        if self.selected_aptag_id is not None and self.x_position is not None and self.y_position is not None:
            selected_id = self.selected_aptag_id
            selected_aptag = self.selected_apriltag
            
            # print("selected_id",selected_id)

            K_gains = self.K_gains[self.k_index*2:self.k_index*2+2,:]
            # print('kgains',K_gains)
            # K_added = self.K_added[self.k_index*2:self.k_index*2+2] will add it later

            # apriltag 0 and 10 were not used 
            if self.selected_aptag_id < 10:
                split_idx = selected_id - 1
            else:
                split_idx = selected_id

            ## j is != i where i is the selected apriltag
            K_j = np.hstack((K_gains[:,:split_idx],K_gains[:,split_idx+2:]))
        
            ### 1st term in u
            # dist is li-x and self.apriltag represents li
            dist_vector_in_world_odom = np.array([[self.position_landmark_inworld[selected_id][0] - self.x_position],[self.position_landmark_inworld[selected_id][1] - self.y_position]])
            WRAT= self.euler_to_rotmatrix(self.position_landmark_inworld[selected_id][3:])
            # print("WRAT",WRAT)
            # print('landmatr',self.position_landmark_inworld_matrix)
            from_get_state = self.position_landmark_inworld_matrix[selected_id][:3,:3]
            # print("from_get_state", from_get_state)
            ori = np.dot(WRAT, np.linalg.inv(selected_aptag[:3,:3].T))
            ori_sttae = np.dot(from_get_state, np.linalg.inv(selected_aptag[:3,:3].T))

            # # Rotate the relative distance (which is apriltag-robot):
            # apriltag_distance = np.dot(ori, selected_aptag[:3,3])[:2].reshape((2,1)); 
            dist_vetor_state = np.dot(ori_sttae, selected_aptag[:3,3])[:2].reshape((2,1)); 
            dist_vector_in_world_aptag = np.dot(ori, selected_aptag[:3,3])[:2].reshape((2,1)); 
            dist_vector_in_world_aptag_old = self.dist_in_world_coor(selected_aptag, selected_id) 
            print("dist_odom",dist_vector_in_world_odom)
            print("dist_aptag",dist_vector_in_world_aptag)
            print("dist_aptagold",dist_vector_in_world_aptag_old)
            print("dist_state",dist_vetor_state)
            # print('id',selected_id)
            u_1 = None
            for i in range(np.size(K_j,1)/2):
                if u_1 is None:
                    u_1 = (K_j[:,i*2:i*2+2]*dist_vector_in_world_odom)
                    # print('u_1',u_1)
                    # print('u1',u_1)
                    # print('K_j[:,i]',K_j[:,i].size)
                    # print('ola',ola.size)
                else:
                    u_1  = np.hstack((u_1,(K_j[:,i*2:i*2+2]*dist_vector_in_world_odom)))
                    
                    # print('u15',u_1)
            u_1 = u_1.sum(axis = 1)

            ### 2nd term in u

            lj_li = self.lj_li(selected_id)
            u_2 = np.dot(K_j,lj_li).sum(axis=1)
            u = u_1 + u_2
            # print("u1",u_1)
            # print("u2",u_2)
            self.pub.publish(data = u)
            # print('u',u)
    
    def lj_li(self,aptag_id): 
        x_i = self.position_landmark_inworld[aptag_id][0]
        y_i = self.position_landmark_inworld[aptag_id][1]
        dist_array = None
        for key, value in self.position_landmark_inworld.items():
            # print("key", key)
            if key is not aptag_id:
                # print("key",key)
                # print("aptag_id",aptag_id)
                # print(self.selected_aptag_id)
                x = value[0] - x_i
                y = value[1] - y_i
                # print("x",x,"y",y)
                if dist_array is None:
                    dist_array = np.vstack((x,y))
                    # print(value[0])
                    # print(key)
                    # print(dist_array)
                else:
                    dist_array = np.vstack((dist_array,x,y))
                    # print(dist_array)
                    # print(value[0])
                    # print(key)
        return dist_array
        
    def to_tf(self,pos,ori):
        return np.block([[np.array(ori),pos.reshape((-1,1))],[0,0,0,1]])

def read_matrix(csv_dir):
    with open(csv_dir,'r') as f:
        return np.genfromtxt(f,delimiter=',')

home_dir = os.environ["HOME"]


if __name__ == "__main__":


    rospy.init_node("u_control")
    
    shared_path = os.environ["HOME"]+"/catkin_ws/src/output_feedback_controller/csv/"
    K_gains_path= shared_path + "K_gains.csv"
    K_gains = read_matrix(K_gains_path)

    K_added_path= shared_path + "K_added.csv"
    K_added = read_matrix(K_added_path)


# ### TO DOO ARRAY WITH LANDMARK POSITION WRT WORLD
    #  jackal = Jackal(K_gains)
    jackal = Jackal(K_gains,K_added)
    jackal.get_rot_matrix_aptags()
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        jackal.compute_u()
        # jackal.get_rot_matrix_aptags()
    r.sleep()   




 


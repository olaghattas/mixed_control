#!/usr/bin/env python

#import the dependencies

import rospy
from geometry_msgs.msg import Twist, Pose, TwistStamped
from nav_msgs.msg import Odometry
from apriltag_ros.msg import AprilTagDetectionArray
import transformation_utilities as tu
import numpy as np
import tf2_ros 
import tf2_geometry_msgs
import os
from std_msgs.msg import Float64MultiArray,Float64, Header
import pandas as pd


class Jackal:
    def __init__(self,K_gains,K_added): 
        self.pub=rospy.Publisher('u_input', TwistStamped, queue_size=1)
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
        self.position_landmark_inworld_matrix = {}
        self.x_position = None
        self.y_position = None
        self.used_apriltags = [1,2,3,4,5,6,7,8,9,11,12,13,14,15]
        self.position_landmark_inworld_matrix = {}
        self.position_landmark_inworld = {1:[-8.24439, -5.8941, 0.5, 0, 0, 0, 1.0],
                                          2:[-8.27337, -1.68115, 0.5, 0, 0, 0, 1.0],
                                          3:[-8.23126, 1.44051, 0.5, 0, 0, 0, 1.0],
                                          4:[-5.65321, 3.03627, 0.5, 0, 0, -0.717185376228,0.696882440678],
                                          5:[-2.52459, 3.22243, 0.5, 0, 0,  -0.699616049545, 0.714518987305],
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
    
    # def get_rot_matrix_aptags(self):
    #     rospy.wait_for_service('/gazebo/get_model_state')
    #     get_model_srv = rospy.ServiceProxy('/gazebo/get_model_state',GetModelState)
    #     model = GetModelStateRequest()
    #     for id in self.used_apriltags:
    #         model.model_name = 'apriltag'+str(id)
    #         result = get_model_srv(model)
    #         self.position_landmark_inworld_matrix[id] = result.pose
    #     print(self.position_landmark_inworld_matrix)

        # print(self.position_landmark_inworld_matrix)

    def odom_callback(self,msg):
        # just to compare
        if msg:
            self.x_position = msg.pose.pose.position.x
            self.y_position = msg.pose.pose.position.y
            # print(self.x_position,self.y_position)

    def apriltag_callback(self,msg):
        if msg.detections:
            # '''If there's an AprilTag in the image it selectes the one closest to the apriltag '''
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
                self.selected_aptag_id = selected_aptag_id 
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

## compute u using only one apriltag then check how you can include multipe apriltags

    def compute_u(self): # n = num of aptags seen K=[2n,?] y = [2n,1]
        ### eq u = sum_j [K_j(l_i - x)] + sum_j [K_j(l_j - li)]
        if self.selected_aptag_id is not None and self.x_position is not None and self.y_position is not None:
            selected_id = self.selected_aptag_id
            selected_aptag = self.selected_apriltag
            
            # print("selected_id",selected_id)

            K_gains = self.K_gains[self.k_index*2:self.k_index*2+2,:]
            # print("from_get_state", from_get_state)
            ori_state = np.dot(self.position_landmark_inworld_matrix[selected_id][:3,:3], np.linalg.inv(selected_aptag[:3,:3]))
          
            # # Rotate the relative distance (which is apriltag-robot):
            # apriltag_distance = np.dot(ori, selected_aptag[:3,3])[:2].reshape((2,1)); 
            dist_vector_state = np.dot(ori_state, selected_aptag[:3,3])[:2].reshape((2,1)); 
            print('dist_state',dist_vector_state)
            print('dist_od', self.x_position , self.y_position)
            
            _idx = 0
           
            u = 0
            for landmark_id in self.position_landmark_inworld_matrix.keys():
                k = K_gains[:,_idx*2:_idx*2+2]
            
                if landmark_id == selected_id:
                    for _z in range(K_gains.shape[1]/2):
                        # print('_z',_z)
                        # print('K_matrix[:,_z*2:_z*2+2]',K_matrix[:,_z*2:_z*2+2])
                        u += K_gains[:,_z*2:_z*2+2].dot(dist_vector_state)
                        # print('u_1111',u_11)
                else:
                   
                    u += np.dot(k,self.position_landmark_inworld_matrix[landmark_id][:2,3] - self.position_landmark_inworld_matrix[selected_id][:2,3]).reshape(2,-1)
                    
# print('lj-li',self.position_landmark_inworld_matrix[landmark_id][:2,3] - self.position_landmark_inworld_matrix[selected_aptag][:2,3])
                _idx += 1
          
            u_1 = None

            for i in range(np.size(K_gains,1)/2):
                if u_1 is None:
                    u_1 = K_gains[:,i*2:i*2+2].dot(dist_vector_state)
                    
                    # print('u_1',u_1)
                    # print('u1',u_1)
                    # print('K_j[:,i]',K_j[:,i].size)
                    # print('ola',ola.size)
                else:
                    u_1  = np.hstack((u_1,K_gains[:,i*2:i*2+2].dot(dist_vector_state)))

            u_1 = u_1.sum(axis = 1)
            print('u',u_1)
            ### 2nd term in u

            lj_li = self.lj_li(selected_id)
            u_2 = np.dot(K_gains,lj_li).sum(axis=1)
            u = u_1 + u_2

            msg = TwistStamped()
            msg.header.stamp = rospy.Time.now()
            msg.twist.linear.x = u[0]
            msg.twist.linear.y = u[1]

    def qt_to_matrix( self ):  
        for id in self.position_landmark_inworld.keys():
                pose = Pose()
                pose.position.x = self.position_landmark_inworld[id][0]
                pose.position.y = self.position_landmark_inworld[id][1]
                pose.position.z = self.position_landmark_inworld[id][2]
                pose.orientation.x = self.position_landmark_inworld[id][3]
                pose.orientation.y = self.position_landmark_inworld[id][4]
                pose.orientation.z = self.position_landmark_inworld[id][5]
                pose.orientation.w = self.position_landmark_inworld[id][6]

                self.position_landmark_inworld_matrix[id] = tu.msg_to_se3(pose)

    def lj_li(self,aptag_id): 
        x_i = self.position_landmark_inworld_matrix[aptag_id][0,3]
        y_i = self.position_landmark_inworld_matrix[aptag_id][1,3]
        # print('matrix',self.position_landmark_inworld_matrix[aptag_id])
        # print('x_i',x_i)
        dist_array = None
        for key, value in self.position_landmark_inworld_matrix.items():
            # print("aptag_id",aptag_id)
            # print(self.selected_aptag_id)
            x = value[0,3] - x_i
            y = value[1,3] - y_i
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
        

def read_matrix(csv_dir):
    with open(csv_dir,'r') as f:
        return np.genfromtxt(f,delimiter=',')

home_dir = os.environ["HOME"]


if __name__ == "__main__":


    rospy.init_node("u_control")
    
    shared_path = os.environ["HOME"]+"/catkin_ws/src/output_feedback_controller/csv/"
    K_gains_path= shared_path + "K_gains_new.csv"
    K_gains = read_matrix(K_gains_path)

    K_added_path= shared_path + "K_added_new.csv"
    K_added = read_matrix(K_added_path)


# ### TO DOO ARRAY WITH LANDMARK POSITION WRT WORLD
    #  jackal = Jackal(K_gains)
    jackal = Jackal(K_gains,K_added)
    jackal.qt_to_matrix()
    # jackal.get_rot_matrix_aptags()
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        jackal.compute_u()
        # jackal.get_rot_matrix_aptags()
    r.sleep()   




 


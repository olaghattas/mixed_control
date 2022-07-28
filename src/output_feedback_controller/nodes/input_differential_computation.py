#!/usr/bin/env python

#import the dependencies
 
import rospy
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from apriltag_ros.msg import AprilTagDetection, AprilTagDetectionArray
import transformation_utilities as tu
import numpy as np
import tf2_ros
import tf.transformations 
import tf2_geometry_msgs
# import os
from std_msgs.msg import Float64MultiArray, Float64


class Input_Differential:
    def __init__(self):
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.pub_linear_vel = rospy.Publisher('/linear_vel', Float64 ,queue_size=1)
        self.pub_vel = rospy.Publisher('/cmd_vel',Twist,queue_size=1)
        self.sub_u_input = rospy.Subscriber("/u_input",Float64MultiArray, self.feedback_control_callback)
        self.sub_img_detec =  rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self.apriltag_callback)
        self.vel = Twist()
        self.position_landmark_inworld = {1:[-8.24439, -5.8941, 0.5, 0, -0, 0,],
                                    2:[-8.27337, -1.68115, 0.5, 0, -0, 0],
                                    3:[-8.23126, 1.44051, 0.5, 0, -0, 0],
                                    4:[-5.65321, 3.03627, 0.5, 0, 0, -1.59951],
                                    5:[-2.52459, 3.22243, 0.5, 0, 0, -1.54972],
                                    6:[1.31312, -2.15597, 0.5, 0, 0, -3.09205],
                                    7:[1.45869, -6.20846, 0.5, 0, -0, 3.13477],
                                    8:[-2.46765, -8.80024, 0.5, 0, -0, 1.56763],
                                    9:[-5.70343, -8.65199, 0.5, 0, -0, 1.52896],
                                    11:[-4.49279, 0.886215, 0.5, 0, -0, 0],
                                    12:[-4.10288, -2.02095, 0.5, 0, 0, -3.12059],
                                    13:[-1.98548, -2.07486, 0.5, 0, -0, 0],
                                    14:[-1.93215, -4.18616, 0.5, 0, -0, 0],
                                    15:[1.22551, 1.92208, 0.5, 0, -0, 3.11799]
                                    }
        self.at_transf = None
        self.selected_id = None
        self.u = None
    
    def apriltag_callback(self,msg):
        if msg.detections:
            # print(msg.detections)
            # '''If there's an AprilTag in the image'''
            min_distance = np.inf
            
            selected_apriltag = []
            for at in msg.detections:
                dist = np.linalg.norm([at.pose.pose.pose.position.x, at.pose.pose.pose.position.y, at.pose.pose.pose.position.z])
                if dist < min_distance:
                    min_distance = dist
                    self.selected_id = at.id[0]
                    selected_apriltag = at.pose.pose

                #change frame from camera to baselink
                source_frame = "front_realsense_gazebo"
                transform = self.tfBuffer.lookup_transform("base_link", source_frame, rospy.Time(0), rospy.Duration(1.0))
                pose_transformed = tf2_geometry_msgs.do_transform_pose(selected_apriltag, transform)
                '''Now add the apriltag to seen apriltags_list'''
                self.at_transf = tu.msg_to_se3(pose_transformed.pose)
                # print(self.at_transf)
        else:
            self.selected_id = None


    def to_tf(self,pos,ori):
        return np.block([[np.array(ori),pos.reshape((-1,1))],[0,0,0,1]])

    def feedback_control_callback(self,msg):
        if msg.data:
            # print("msg",msg)
            u = np.array(msg.data)
            print('u',u)
        else:
            self.u = None
            
    def compute_differntial_control(self):
        at_transf = self.at_transf
        selected_id = self.selected_id
        if self.u is not None and at_transf is not None:
            
            orientation = self.robot_pose(at_transf,selected_id)
            print('orien',orientation)

            alpha = 0.5
            beta = 0.1
            print('u',self.u)
            print('ori',orientation)

            # ux = alpha/np.linalg.norm(self.u)
            # ux = np.dot(ux,orientation/np.linalg.norm(self.u))[2]
            # ux = np.dot(ux,np.array([[np.cos(orientation)],[np.sin(orientation)]]).T)
            ux = alpha * np.dot(self.u, orientation[:2])
            ux = ux/np.linalg.norm(self.u)
            print('ux',ux)

            # wz = beta/np.linalg.norm(self.u)
            # wz =np.dot(wz,np.array([[0],[0],[1]]).T)
            # ,axis=0
         
            cross = np.cross(orientation,self.u/np.linalg.norm(self.u))[2]
            print('cross',cross)
            wz = beta*cross
            print('wz',wz)

            # adjust the velocity message
            self.vel.angular.z = wz

            self.vel.linear.x = ux
            #publish it
            self.pub_vel.publish(self.vel)
            self.pub_linear_vel.publish(ux)


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

## Get the orientation from different apriltags this only gets from the closest one
    def robot_pose(self,at_transf,selected_id):
        # Finally, robot pose wrt. the world: WTR = WTAT & (RTAT)-1
        # landmarks are in euler angles so chnage them to rotation matrix
        WRAT= self.euler_to_rotmatrix(self.position_landmark_inworld[selected_id][3:])
        # print('ori',ori)
        #Get the transformation using the totf function 
        #WTAT = self.to_tf(np.array(self.position_landmark_inworld[self.selected_id][:3]),ori)

        # print('wtat',WTAT)
        #pose of the robot in the world
        # ori = np.dot(WRAT, np.linalg.inv(at_transf[:3,:3]))
        ori = np.dot(WRAT, np.linalg.inv(self.at_transf[:3,:3].T))

        # print('pose',self.pose)

        # Update the orientation vector of the robot from the seen AprilTag
        # because we only want the X vector
        
        orientation = ori[:3,0].flatten()
        orientation[2] = 0
        # Finally, update the orientation vector
        orientation /= np.linalg.norm(orientation)
        return orientation


if __name__ == "__main__":

    rospy.init_node("u_sparse_control")

    #  jackal = Jackal(K_gains)
    jackal = Input_Differential()
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        jackal.compute_differntial_control()
	r.sleep() 
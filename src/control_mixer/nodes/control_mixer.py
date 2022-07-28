#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import numpy as np


class ControlMixer:
    def __init__(self):
        self.sub_opflow_control = rospy.Subscriber('/cmd_vel_opticalflow',Twist,self.callback_opflow)
        self.sub_feedback_control = rospy.Subscriber('/cmd_vel_feedbackcontrol',Twist,self.callback_feedback)
        self.sub_img_detec =  rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self.detection_callback)
        self.pub_vel = rospy.Publisher('/cmd_vel',Twist, queue_size=1)
        self.vel = Twist()
        self.opticalflow = False
        self.feedback= False
        self.linear_x_opticalflow = None
        self.angular_z_opticalflow = None
        self.linear_x_feedback = None
        self.angular_z_feedback = None

    def detection_callback(self,msg):
        if msg.detections:
            self.opticalflow = False
            self.feedback= True

        else:
            self.opticalflow = True
            self.feedback= False
		

    def callback_opflow(self, msg):
        self.linear_x_opticalflow = msg.linear.x
        self.angular_z_opticalflow = msg.angular.z

    
    def callback_feedback(self, msg):
        self.linear_x_feedback = msg.linear.x
        self.angular_z_feedback = msg.angular.z

    def publish_control(self):
        if self.opticalflow:
            # adjust the velocity message
            self.vel.linear.x = self.linear_x_opticalflow
            self.vel.angular.z = self.angular_z_opticalflow
            
            #publish it
            self.pub_vel.publish(self.vel)
        elif self.feedback:
            # adjust the velocity message
            self.vel.linear.x = self.linear_x_feedback
            self.vel.angular.z = self.angular_z_feedback
            #publish it
            self.pub_vel.publish(self.vel)

if __name__=="__main__":
    
	#initialise the node
	rospy.init_node("mixer", anonymous=True)
	cont_mix= ControlMixer()
	#while the node is still on
	r = rospy.Rate(10)
	while not rospy.is_shutdown():
		cont_mix.publish_control()
		r.sleep()
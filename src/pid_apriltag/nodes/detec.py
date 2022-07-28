#!/usr/bin/env python

#import the dependencies
 
import rospy
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from apriltag_ros.msg import AprilTagDetection, AprilTagDetectionArray
import numpy as np
import tf2_ros
import tf.transformations 
import tf2_geometry_msgs


class PID:
    def __init__(self, Kp=0, Ki=0, Kd=0):
        '''
        '''
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    def proportional_control(self, error):
		return self.Kp * error

    def integral_control(self, error, dt):
        return self.Ki * error * dt

    def derivative_control(self, error, previous_error, dt):
        return self.Kd * (error - previous_error)/dt



class Jackal:
	def __init__(self):
		self.pub=rospy.Publisher('/cmd_vel',Twist,queue_size=1)
		self.sub_img_detec =  rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self.detection_callback)
		self.pose_x = None
		self.pose_y = None
		self.pose_z = None
	
		self.spin = True
		self.move = False
		self.vel = Twist()
		self.prev_error = 0
		self.tfBuffer = tf2_ros.Buffer()
		self.listener = tf2_ros.TransformListener(self.tfBuffer)
		self.saved_time = rospy.Time.now()
		self.detected = False
		self.dist_to_tag = 0

		# pid parameters
		kp = rospy.get_param("/gain/kp")
		ki = rospy.get_param("/gain/ki")
		kd = rospy.get_param("/gain/kp")
		self.controller = PID(kp,kd,ki)
		
	def update_status(self):
		# print("dist_to_tag", self.dist_to_tag)
		if not self.detected :
			self.move = False
			self.spin = True 
		elif self.detected and self.dist_to_tag < 1.55: # limit set baseed on trials
			self.move = False
			self.spin = True
		else:
			self.spin = False
			self.move = True

	def on_apriltag_detection(self):
		if self.pose_x is not None and self.pose_y is not None and self.pose_z is not None:
			self.dist_to_tag = self.get_dist() 

	def detection_callback(self,msg):
		if msg.detections:
			source_frame = "front_realsense_gazebo"
			transform = self.tfBuffer.lookup_transform("base_link", source_frame, rospy.Time(0), rospy.Duration(1.0))
			# print("trans",transform)
			#.pose.powse to get to Posestamped 
			pose_transformed = tf2_geometry_msgs.do_transform_pose(msg.detections[0].pose.pose, transform)
			#print("pose_transf",pose_transformed)
			self.pose_x= pose_transformed.pose.position.x
			self.pose_y= pose_transformed.pose.position.y
			self.pose_z= pose_transformed.pose.position.z
			self.detected = True
			self.on_apriltag_detection()
			
		else:
			self.pose_x = None
			self.pose_y = None
			self.pose_z = None
			self.detected = False

	def spinning(self):
		# adjust the velocity message
		self.vel.angular.z = 1
		self.vel.linear.x = 0
		#publish it
		self.pub.publish(self.vel)

	def velocity_control(self, error, dt, prev_error):
		max_vel = 4
		mv_p = self.controller.proportional_control(error)
		mv_i = self.controller.integral_control(error, dt)
		mv_d = self.controller.derivative_control(error, prev_error, dt)
	
		desired_vel = np.clip( mv_p + mv_i + mv_d, -max_vel, max_vel)
		return desired_vel
	
				
	def get_dist(self):
		return np.linalg.norm([self.pose_x, self.pose_y, self.pose_z])

	def move_towards_tag(self):
		current_error = self.pose_y
		print("current_error",current_error)
	
		if current_error is not None:
			
			self.vel.linear.x = 1
			current_time = rospy.Time.now()
			dt = (current_time - self.saved_time).to_sec()
			pid_output = self.velocity_control(current_error, dt, self.prev_error)
			# print("pid_output ", pid_output)
			self.vel.angular.z = pid_output 
			self.saved_time = current_time

			self.pub.publish(self.vel)
			self.prev_error = current_error


if __name__=="__main__":
    
	#initialise the node
	rospy.init_node("ap_tag", anonymous=True)
	jack = Jackal()
	#while the node is still on
	r = rospy.Rate(10)
	while not rospy.is_shutdown():
		jack.update_status()
		if jack.spin:
			jack.spinning()
		elif jack.move:
			jack.move_towards_tag()
		r.sleep()

	

	
#! /usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from gazebo_msgs.srv import GetModelState, GetModelStateRequest

rospy.init_node('odom_pub')

odom_pub = rospy.Publisher('/my_odom', Odometry, queue_size=10)

# connect to the service to ask proxy for info

rospy.wait_for_service('/gazebo/get_model_state')
get_model_srv = rospy.ServiceProxy('/gazebo/get_model_state',GetModelState)

#instance of odometry message that is empty to fill later
odom = Odometry()
header = Header() # contais timestamp and sequence
header.frame_id = '/odom'

model = GetModelStateRequest()
model.model_name = 'jackal'

r = rospy.Rate(2)

while not rospy.is_shutdown():
    result = get_model_srv(model)
    odom.pose.pose = result.pose
    odom.twist.twist = result.twist

    header.stamp = rospy.Time.now()
    odom.header = header
    
    odom_pub.publish(odom)

    r.sleep()

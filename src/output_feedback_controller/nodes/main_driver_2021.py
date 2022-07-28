#!/usr/bin/env python

from __future__ import print_function

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Pose, Point, Twist
from nav_msgs.msg import Odometry
import transformation_utilities as tu
import so3_utilities as so
from creates_iros.msg import AprilTagDetectionArray
import tf.transformations as tr

import copy
import os

# Check if I'm on a Create:
SIMULATION = True
try:
    import RPi.GPIO as GPIO
    SIMULATION = False
except:
    print("Not in a real create. Simulating instead!")
    import matplotlib.pyplot as plt

def read_matrix(csv_dir):
    with open(csv_dir,'r') as f:
        return np.genfromtxt(f,delimiter=',')

def write_matrix(csv_dir,mat):
    with open(csv_dir,'w') as f:
        np.savetxt(f,mat,delimiter=',')

class Landmark(object):
    def __init__(self,id,position,orientation=[0,0,0],displaced=np.zeros((3,))):
        
        self.id = id

        self.position = position 

        try:
            self.orientation = so.exp3(orientation)
        except:
            self.orientation = orientation

        self.to_tf(self.position,self.orientation)

        # We need to store the transformation between the seen landmark and the
        # algorithm landmark:
        self.visual_to_landmark(displaced)

    def to_tf(self,pos,ori):
        self.tf = np.block([[np.array(ori),pos.reshape((-1,1))],[0,0,0,1]])

    def dot(self,matrix):
        assert type(matrix) == np.ndarray
        return self.tf.dot(matrix)

    def visual_to_landmark(self,displaced):
        # Displacement is the position of the _seen_ landmark,
        # and not the _algorithm_ landmark:
        displacement = np.block([[self.orientation,displaced.reshape((3,-1))],[0,0,0,1]])
        
        if self.id == 12:
            displacement[:3,:3] = np.array([[0,-1,0],[0,0,1],[-1,0,0]]).T
        elif self.id == 13:
            displacement[:3,:3] = np.array([[1,0,0],[0,0,1],[0,-1,0]]).T
        
        
        self.displacement = np.linalg.inv(displacement).dot(self.tf)

class Create(object):

    def __init__(self,K_gains,landmarks,obstacle_coordinates,transform_matrix=np.eye(4),start_point = None,orientation_=[1,0],sequence_idx=0):

        self.transform_matrix = transform_matrix
        self.obstacle_coordinates = obstacle_coordinates

        # One matrix with all gains concatenated
        self.K_gains = K_gains

        # Save landmark transformations wrt. mocap:
        self.landmark_poses = landmarks

        # if the seen landmark is 12 or 13...
        # Transforms are 0->13 and 3->12
        self.landmark_transforms = {13:(0,np.linalg.inv(np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,52],[0,0,0,1]]))),
                                    12:(3,np.linalg.inv(np.array([[0,0,1,5],[0,1,0,0],[-1,0,0,-6],[0,0,0,1]])))}
                                    
        self.landmark_transforms[13][1][:3,3] *= 1e-3
        self.landmark_transforms[12][1][:3,3] *= 1e-3
        

        # Position and speed variables:
        self.start_point = copy.deepcopy(start_point)
        self.position = copy.deepcopy(start_point)
        self.orientation = orientation_
        self.linear_speed = np.inf
        self.angular_speed = np.inf
        self.speed = np.inf

        # robot to camera transform for the apriltags
        self.cam_to_robot = (np.array([[0,0,1,6.5],[-1,0,0,3],[0,-1,0,-1],[0,0,0,1]]))
        self.cam_to_robot[:3,3] *= 1e-2

        ### Landmarks and cells
        # Used to select what landmarks to trust
        self.cell_to_landmark = {'L1':[0,13,6,7,16], 'L2':[1,5,4,14],'L3':[11,2,10,15],'L4':[3,12,8,9]}
        #~ self.K_to_cell = {2:'L1',3:'L2',4:'L2',5:'L2',6:'L3',7:'L3',8:'L3',9:'L3',10:'L3',11:'L4',12:'L4',13:'L1',14:'L4'}
        self.K_to_cell = {2:'L1',3:'L2',4:'L2',5:'L2',6:'L2',7:'L3',8:'L3',9:'L3',10:'L3',11:'L4',12:'L4',13:'L1',14:'L4'}

        # For current trajectory, what K's will it visit?
        all_sequences = [[4,3],[7,6,4,3],[9,11,12,13],[13]]
        #~ sequence_idx = 0
        
        self.sequence_id = sequence_idx
        self.K_sequence = all_sequences[sequence_idx]
        self.K_index = 0

        self.current_cell = self.K_to_cell[self.K_sequence[self.K_index]]
        
        self.rate = rospy.Rate(5)

        self.apriltag = None
        
        # Odometry will be (x,y,theta)
        self.odometry = None
        
        # Save trajectory!
        root_dir = os.environ["HOME"]
        
        self.trajectory_file = root_dir + "/catkin_ws/src/creates_iros/csv/trajectory_sequence_"+str(sequence_idx)+"_deformed.csv"
        self.mocap_file = root_dir + "/catkin_ws/src/creates_iros/csv/mocap_trajectory_"+str(sequence_idx)+".csv"
        self.odometry_file = root_dir + "/catkin_ws/src/creates_iros/csv/odometry_trajectory_"+str(sequence_idx)+".csv"
        
        self.mocap_trajectory = list()
        self.odometry_trajectory = list()  
        
           
        
        # subscribers and publishers
        self.apriltag_sub = rospy.Subscriber("/tag_detections",AprilTagDetectionArray,self.apriltag_callback)
        
        if not SIMULATION:
            self.vrpn_sub = rospy.Subscriber("/create1/pose",PoseStamped,self.vrpn_callback)
            self.odom_sub = rospy.Subscriber("/odom",Odometry,self.odometry_callback)

        self.vel_pub = rospy.Publisher("/cmd_vel",Twist,queue_size=10)   


    def odometry_callback(self,msg):
        
        if self.odometry is not None:

            self.odometry[0] = self.start_point[0,0] + msg.pose.pose.position.x
            self.odometry[1] = self.start_point[1,0] + msg.pose.pose.position.y

            q = msg.pose.pose.orientation
            orientation_matrix = tr.quaternion_matrix([q.x,q.y,q.z,q.w])[:3,:3]
            theta = np.arctan2(orientation_matrix[1,0],orientation_matrix[0,0])
            self.odometry[2] = self.odometry[2] + theta

        self.linear_speed = np.linalg.norm(np.array([msg.twist.twist.linear.x,msg.twist.twist.linear.y,msg.twist.twist.linear.z]))
        self.angular_speed = np.linalg.norm(np.array([msg.twist.twist.angular.x,msg.twist.twist.angular.y,msg.twist.twist.angular.z]))


    def apriltag_callback(self,msg):
        self.apriltags = []
        
        if msg.detections:
            '''If there's an AprilTag in the image'''
            min_distance = np.inf
            selected_id = 0
            selected_apriltag = None
            for at in msg.detections:
                
                if at.id[0] == 8:
                    continue
                
                tmp_point = at.pose.pose.pose.position

                ''' ADD the FIX for apriltag distance!'''
                gamma = 1.04
                tmp_point.x *= gamma
                tmp_point.y *= gamma
                tmp_point.z *= gamma

                dist = np.linalg.norm([tmp_point.x, tmp_point.y, tmp_point.z])
                if dist < min_distance:
                    min_distance = dist
                    selected_id = at.id[0]
                    selected_apriltag = at.pose.pose.pose
                    selected_apriltag.position = tmp_point
            
                self.apriltag = (selected_id,self.cam_to_robot.dot(tu.msg_to_se3(selected_apriltag).\
                    dot(self.landmark_poses[selected_id].displacement)))
            
            
            #~ rospy.logwarn("I'm seeing AprilTag {}".format(self.apriltag[0]))
            
        else:
            #~ rospy.logwarn("Can't find an AprilTag in the image!")
            pass

    def vrpn_callback(self,msg):

        self.position = np.array([[msg.pose.position.x,msg.pose.position.y]]).T
        
        
        q = msg.pose.orientation 
        
        self.tf = tr.quaternion_matrix([q.x,q.y,q.z,q.w]); 
        self.tf[:3,3] = np.array([msg.pose.position.x,msg.pose.position.y,msg.pose.position.z])
        
        self.orientation = self.tf[:3,0].flatten(); self.orientation[2] = 0
        self.orientation /= np.linalg.norm(self.orientation)

        # Angle of rotation around Z axis:
        theta = np.arctan2(self.orientation[1],self.orientation[0])
        
        if self.start_point is None:
            self.start_point = np.array([[msg.pose.position.x,msg.pose.position.y]]).T
            self.position = np.array([[msg.pose.position.x,msg.pose.position.y]]).T
            self.odometry = np.array([msg.pose.position.x,msg.pose.position.y,theta]).T

    def get_position(self):
        return self.position.T
    
    def get_speed(self):
        return self.speed

    def draw_landmarks(self):
        # Re-mark current cell landmarks:
        curr_landmark_poses = []
        for _id in self.cell_to_landmark[self.current_cell]:
            if _id != 12 and _id != 13:
                lndmrk = self.landmark_poses[_id]
                curr_landmark_poses.append(list(lndmrk.tf[:2,3].flatten()))
        curr_landmark_poses = np.array(curr_landmark_poses).T
        ax = self.fig.axes[0]
        self.curr_landmarks = ax.plot(curr_landmark_poses[0,:],curr_landmark_poses[1,:],'co',markersize=self.markersize)
        plt.draw(); plt.pause(0.01)
        
        
    def compute_control_input(self):
        # Current cell:
        cell = self.current_cell
        
        orientation = None
        
        if self.apriltag:
        
            seen_id = self.apriltag[0]
            
            # We may need to switch to another apriltag:
            new_apriltag = seen_id
            
            if seen_id == 12:
                new_apriltag = 3
                
            elif seen_id == 13:
                new_apriltag = 0
                
            self.apriltag = (new_apriltag, self.apriltag[1].dot(np.linalg.inv(self.landmark_poses[seen_id].tf).dot(self.landmark_poses[new_apriltag].tf)))
            seen_id = self.apriltag[0]
                
                
            # Get rotation!
            if seen_id != 5 and seen_id != 8 and seen_id != 9: 
                # Because we don't have their rotations, they are inside the deformed obstacle at random orientations
                apriltag = self.apriltag[1]
                rospy.logwarn("Used apriltag {} FOR ROTATION".format(seen_id))
                # Full rotation matrix from AprilTag measured rotation:
                at_orientation = (self.landmark_poses[seen_id].tf[:3,:3]).dot(apriltag[:3,:3].T)

                # Finally, update the orientation vector of the robot from the seen AprilTag:
                orientation = at_orientation[:3,0].flatten(); orientation[2] = 0; orientation /= np.linalg.norm(orientation)
                
                rospy.logwarn("Seeing apriltag {}, in cell {}".format(self.apriltag[0], self.cell_to_landmark[cell]))
            
        if self.apriltag and (seen_id != 8 and seen_id != 5 and seen_id != 9) and seen_id in self.cell_to_landmark[cell]:
            ''' This is for when we SEE apriltags. No need to use MoCap, yay! '''
            # Apriltag used for reference:
            apriltag = self.apriltag[1]

            '''For now, the following just for plotting'''
            # 3x3 rotation matrix, orientation robot->world:
            at_orientation = (self.landmark_poses[seen_id].tf[:3,:3]).dot(apriltag[:3,:3].T)
            #~ at_orientation = copy.deepcopy(self.tf[:3,:3])
            
            # Rotate the relative distance (which is apriltag-robot):
            apriltag_distance = (at_orientation.dot(apriltag[:3,3])[:2]).reshape((2,1)); 
            
            orientation = at_orientation[:3,0].flatten(); orientation[2] = 0; orientation /= np.linalg.norm(orientation)
            
            
            print(self.orientation.flatten(),orientation)
            real_apriltag_distance = np.linalg.inv(self.tf).dot(self.landmark_poses[seen_id].tf) # apriltag->robot
            # Now we need to rotate this distance by the REAL orientation (which is self.tf[:3,:3])
            real_apriltag_distance = (self.tf[:3,:3].dot(real_apriltag_distance[:3,3]))[:2].reshape((2,-1));
            
            # Compare apriltag measurements with real MoCap measurements:
            print("Distance with AprilTag: {}".format(apriltag_distance.flatten()))
            print("Real distance: {}".format(real_apriltag_distance.flatten()))
            
            source = "Apriltag"

        else:
            rospy.logwarn("No AprilTag detected!")
            ''' This is for when the camera of the Create wants to "die" '''

            if orientation is None:
                orientation = copy.deepcopy(self.orientation)
            
            # If we don't see an AprilTag (i.e. if we don't enter the previous 'if'), return current orientation and
            # old control input:
            rospy.logerr("Returning old control input!")
            return self.old_u, orientation
            
            seen_id = self.cell_to_landmark[cell][0]
            
            apriltag_distance = np.linalg.inv(self.tf).dot(self.landmark_poses[seen_id].tf) # apriltag->robot
            
            # We need to add one transformation: from robot to world (THAT'S when we use global rotation).
            # transform_matrix is the identity now (it goes from local to global coordinates: local->global)
            apriltag_distance = self.transform_matrix[:3,:3].T.dot(self.tf[:3,:3]).dot(apriltag_distance[:3,3]).flatten(); apriltag_distance = apriltag_distance[:2].reshape((2,1))
            
            source = "MoCap"
        
        print("Using the source %s."%(source))

        ''' In the following lines, Proposition 1 of the paper is applied '''
        
        # We need the -2, as real K indices start with one and we didn't copy the null K [1]!
        k_index = self.K_sequence[self.K_index]-2 
        
        # Obtain all landmarks in the cell for error computation:
        _idx = 0 # current landmark index (in the loop)
        u = 0 # The control input
        K_matrix = copy.deepcopy(self.K_gains[k_index*2:k_index*2+2,:])
        K_matrix = K_matrix[~np.isnan(K_matrix)].reshape((2,-1))

        for landmark_id in self.cell_to_landmark[cell]:
            k = K_matrix[:,_idx*2:_idx*2+2]
            if landmark_id == 12 or landmark_id == 13:
                continue
            if landmark_id == seen_id:
                for _z in range(K_matrix.shape[1]/2):
                    u += K_matrix[:,_z*2:_z*2+2].dot(apriltag_distance)
            else:
                # We compute another term of the bias:
                u += k.dot(self.transform_matrix[:2,:2].T).dot((self.landmark_poses[landmark_id].tf[:2,3]-self.landmark_poses[seen_id].tf[:2,3]).reshape((2,-1)))
            _idx += 1
            
        # Finally, go back to global coordinates:
        u = self.transform_matrix[:2,:2].dot(u.flatten())
        u = np.concatenate((u.flatten(),[0]))
        
        # Clear apriltag variable:
        self.apriltag = ()

        return (u, orientation)

    def compute_simulated_input(self):
        cell = self.current_cell
        position = self.position
        k_index = self.K_sequence[self.K_index]-2 

        landmarks = []
        for landmark_id in self.cell_to_landmark[cell]:
            if landmark_id == 12 or landmark_id == 13:
                continue
            new_landmark = self.landmark_poses[landmark_id].tf
            landmarks.append(list(new_landmark[:2,3].flatten()))
            
        landmarks = np.array(landmarks).T
        K_gains = self.K_gains[k_index*2:k_index*2+2,:]; K_gains = K_gains[~np.isnan(K_gains)].reshape((2,-1))

        error = landmarks - position.reshape((2,1)).dot(np.ones((1,landmarks.shape[1])))
        error = self.transform_matrix[:2,:2].T.dot(error)

        old_u = (self.transform_matrix[:2,:2].dot(K_gains).\
            dot(error.flatten('F'))).flatten()
        old_u = np.concatenate((old_u,[0]))

        return old_u
    
    def navigate(self):

        while (not rospy.is_shutdown() and self.position is None) and not SIMULATION:
            rospy.loginfo("Waiting for mocap to send robot position...")
            self.rate.sleep()

        if SIMULATION:
            plt.ion()
            self.fig = plt.figure()
            
            self.draw_environment()
            # Re-mark current cell landmarks:
            self.markersize = 10
            self.draw_landmarks()
            rospy.loginfo("Current K: %d"%(self.K_sequence[self.K_index]))
            
        if not rospy.is_shutdown():
            rospy.loginfo("Navigating!")
            
        self.old_u = np.zeros(3,)

        while not rospy.is_shutdown():


            if SIMULATION:
                u = self.compute_simulated_input()
                orientation = np.concatenate((self.orientation,[0]))
            else:                
                # raise NotImplementedError("\nImplement:\nWaaay smaller speeds\nOdometry when not seeing apriltag\nAnything else?")
                u,orientation = self.compute_control_input()
                if self.position is not None and self.odometry is not None:
                    #~ print("Real position: ",self.position[:2,0]," Odom position: ", self.odometry[:2])
                    
                    # Store current position to our self.trajectory list:
                    self.mocap_trajectory.append([self.position[0,0],self.position[1,0]])
                    # Store odometry to trajectory:
                    self.odometry_trajectory.append([self.odometry[0],self.odometry[1]])
                    
                self.old_u = copy.deepcopy(u)
                
            # OR send /cmd_vel instead!
            vel_msg = Twist()

            
            linear_coef = 0.4/5
            angular_coef = 5.0/10 # fast, but slow enough so that we see landmarks
            
            linear_vel_threshold = 0.02*2/5
            
            linear_vel = linear_coef * u.dot(orientation)
            
            '''The logic of behind the velocity control'''
            if linear_vel < 0:
                # if goal is behind us:
                linear_vel = 0
                angular_vel = angular_coef * np.sign(np.cross(orientation,u/np.linalg.norm(u))[2])
            elif linear_vel < linear_vel_threshold * 0.5:
                # if I'm close to the goal, don't chatter!
                self.vel_pub.publish(Twist())
                try:
                    self.K_index += 1
                    self.current_cell = self.K_to_cell[self.K_sequence[self.K_index]]
                    rospy.loginfo("Next K: %d"%(self.K_sequence[self.K_index]))
                    if SIMULATION:
                        # Delete previously marked landmarks:
                        self.curr_landmarks.pop(0).remove()
                        # Re-draw landmarks:
                        self.draw_landmarks()
                except: 
                    rospy.loginfo("Finished experiment!") 
                    return 
                continue
            else:
                # In any normal case:
                if linear_vel < linear_vel_threshold:
                    angular_vel = 1 * angular_coef * np.cross(orientation,u)[2]
                    print(angular_vel)
                else:
                    angular_vel = 1 * angular_coef * np.cross(orientation,u/np.linalg.norm(u))[2]
                linear_vel = linear_vel/np.linalg.norm(u)
                
                
                
            
            if not SIMULATION:
                vel_msg.linear.x = linear_vel
                vel_msg.angular.z = angular_vel
                self.vel_pub.publish(vel_msg)
                
                dummy_msg = Twist()
                #~ dummy_msg.linear.x = 0.05
                #~ self.vel_pub.publish(dummy_msg)
            
            else:

                position = self.position
                
                # Pyplot stuff:
                ax = self.fig.axes[0]
                ax.plot(position[0],position[1],'bo')
                plt.draw(); plt.pause(0.01)

                # Update next position using linear and angular velocities
                dt = 1.0/10
                self.position = position + linear_vel*dt*np.array([np.cos(np.arctan2(orientation[1],orientation[0])),\
                                                                   np.sin(np.arctan2(orientation[1],orientation[0]))])

                if abs(angular_vel) > 1e-10:
                    self.orientation
                    v = np.cross(angular_vel*dt*np.array([0,0,1]),orientation)[:2]
                    self.orientation = orientation[:2]*np.cos(np.linalg.norm(v)) + v/np.linalg.norm(v) * np.sin(np.linalg.norm(v)) * \
                        np.linalg.norm(orientation)

                # Update simulated speed:
                self.linear_speed = linear_vel
                self.angular_speed = angular_vel


                                    
            self.rate.sleep()

    def on_rospy_shutdown(self):
        rospy.logwarn("Stopping create.")
        rospy.Rate(1).sleep()
        if SIMULATION:
            plt.close(self.fig)
        else:
            # Save trajectories to our files (careful not to overwrite them):
            save_traj = True
            if os.path.exists(self.trajectory_file):
                rospy.logerr("Trajectory already exists! Save it? (Y/N): %s")
                save_traj = raw_input()
                save_traj = 'y' in save_traj.lower() and 'n' not in save_traj.lower()
                
            if save_traj:
                write_matrix(self.trajectory_file,np.array(self.mocap_trajectory))
                #~ write_matrix(self.mocap_file,np.array(self.mocap_trajectory))
                #~ write_matrix(self.odometry_file,np.array(self.odometry_trajectory))
                rospy.loginfo("Trajectory saved to file in:\n%s"%(self.trajectory_file))
            else:
                rospy.logwarn("Trajectory NOT saved.")
            
        self.vel_pub.publish(Twist())

    def draw_environment(self):
        plt.show()
        ax = self.fig.add_subplot(111)
        
        ax.plot(self.obstacle_coordinates[:,0],self.obstacle_coordinates[:,1],'r-')
        ax.plot([self.obstacle_coordinates[-1,0],self.obstacle_coordinates[0,0]],\
            [self.obstacle_coordinates[-1,1],self.obstacle_coordinates[0,1]],'r-')

        ax.axis('equal')

        for i in self.landmark_poses.keys():
            if i != 12 and i != 13:
                landmark = self.landmark_poses[i]
                ax.plot(landmark.position[0],landmark.position[1],'go')

        plt.draw()
        plt.pause(0.001)

if __name__ == "__main__":

    ZERO_TAG_OFFSET = np.array([0,-0.327]) 


    rospy.init_node("creates_iros_node")

    root_dir = os.environ["HOME"]+"/catkin_ws/src/creates_iros/csv/"
    K_dir = root_dir + "K_gains_deformed.csv"
    landmark_positions_dir = root_dir + "landmark_positions_deformed.csv"
    landmark_orientations_dir = root_dir + "landmark_orientations_deformed.csv"
    transform_matrix_dir = root_dir + "transform_matrix.csv"
    obstacle_coordinates_dir = root_dir + "global_obstacle_coordinates.csv"

    visual_landmarks_dir = root_dir + "landmark_visual_positions.csv"

    try:
        visual_landmarks = read_matrix(visual_landmarks_dir)
    except:
        raise NotImplementedError("We don't have coordinates for visual landmarks yet!")

    K_gains = read_matrix(K_dir)
    landmark_positions = read_matrix(landmark_positions_dir)
    landmark_orientations = read_matrix(landmark_orientations_dir)
    transform_matrix = read_matrix(transform_matrix_dir) # From local to global
    obstacle_coordinates = read_matrix(obstacle_coordinates_dir)

    # Transform everything to local:
    #~ print("Transformation matrix:\n",transform_matrix)

    landmarks = {}
    for i in range(landmark_positions.shape[0]):
        #~ if i == 0:
            #~ landmark_positions[i,:2] += ZERO_TAG_OFFSET
        if not np.any(np.isnan(visual_landmarks[i,:])):
            landmarks[i] = Landmark(i,landmark_positions[i,:]*1e-2,(landmark_orientations[i*3:i*3+3,:]),\
                visual_landmarks[i,:])
        else:
            landmarks[i] = Landmark(i,landmark_positions[i,:]*1e-2,(landmark_orientations[i*3:i*3+3,:]),\
                landmark_positions[i,:]*1e-2)

        print(landmarks[i].displacement)



    if SIMULATION:
        start_point = np.array([1.,-3.2]).T
    else:
        start_point = None

    create = Create(K_gains,landmarks,obstacle_coordinates,start_point=start_point,sequence_idx=2)

    rospy.on_shutdown(create.on_rospy_shutdown)

    rospy.loginfo("Create created.")

    create.navigate()

    if SIMULATION:
        rospy.spin()




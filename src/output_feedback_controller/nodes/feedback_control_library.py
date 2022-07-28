#!/usr/bin/env python

#import the dependencies
from __future__ import print_function
from re import U

import rospy
import numpy as np
import pandas as pd

## Compute u using only one apriltag then check how you can include multipe apriltags

def compute_u(selected_aptag_id,selected_apriltag,dist_vector_in_world): # n = num of aptags seen K=[2n,?] y = [2n,1]
    ### eq u = sum_j [K_j(l_i - x)] + sum_j [K_j(l_j - li)]
    if self.selected_aptag_id is not None:
        selected_id = selected_aptag_id
        selected_aptag = selected_apriltag
        print("selected_id",selected_id)

        K_gains = self.K_gains[self.k_index*2:self.k_index*2+2,:]
        # print('kgains',K_gains)
        # K_added = self.K_added[self.k_index*2:self.k_index*2+2] will add it later

        # apriltag 0 and 10 were not used 
        if selected_aptag_id < 10:
            split_idx = selected_id - 1
        else:
            split_idx = selected_id

        ## j is != i where i is the selected apriltag
        K_j = np.hstack((K_gains[:,:split_idx],K_gains[:,split_idx+2:]))
    
        ### 1st term in u
        # dist is li-x and self.apriltag represents li
        # dist_vector_in_world = dist_in_world_coor(selected_aptag, selected_id) 
        # print('id',selected_id)
        u_1 = None
        for i in range(np.size(K_j,1)/2):
            if u_1 is None:
                u_1 = (K_j[:,i*2:i*2+2]*dist_vector_in_world)
                # print('u1',u_1)
                # print('K_j[:,i]',K_j[:,i].size)
                # print('ola',ola.size)
                print(u_1,)
            else:
                u_1  = np.hstack((u_1,(K_j[:,i]*dist_vector_in_world).reshape(-1,1)))
                # print('u15',u_1)
        print('u_1',u_1)
        u_1 = u_1.sum(axis = 1)

        ### 2nd term in u

        lj_li = self.lj_li(selected_id)
        u_2 = np.dot(K_j,lj_li).sum(axis=1)
        u = u_1 + u_2
        # print("u1",u_1)
        # print("u2",u_2)
        self.pub.publish(data = u)
        # print('u',u)






 


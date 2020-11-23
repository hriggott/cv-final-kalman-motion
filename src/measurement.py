'''
Created on Nov 22, 2020

@summary: This is a python file containing functions for position estimation

@author: Dimitri Lezcano

'''

import numpy as np
import numpy.linalg as la
import cv2
import matplotlib.pyplot as plt


def circle_position( img ):
    ''' This is a function to estimate circle position'
    
        @param img: A binarized image of the circle
        
        @return: mu ([x, y]), Q (covariance) if any circles detected, otherwise None
                     
                     
     '''
    # parameters
    min_covar = 1  # minimum covariance 
    
    # perform the HoughCircle transform
    circles = cv2.HoughCircles( img, cv2.HOUGH_GRADIENT, 2, 10, param1 = 100, param2 = 15, minRadius = 5 )
    
    if not isinstance( circles, type( None ) ):
        circles = circles.squeeze()
        
        # calculate the mean
        mu = circles.mean( axis = 0 )  # the mean
        
        Q = np.cov( circles - mu, rowvar = False )[:2, :2]  # the positional covariance
        
        # make sure there is some uncertainty
        if la.norm( Q ) < min_covar:
            Q = np.diag( [min_covar, min_covar] )

        # if
        
        return mu[:2], Q
        
    # if
    
    else:  # no circles found
        return None
    
    # else
    
# circle_position


def main():
    data_dir = '../data/'
    data_file = data_dir + 'circle-constant_acceleration.mp4'
    
    video_cap = cv2.VideoCapture( data_file )
    
    counter = 0
    pts = []
    while video_cap.isOpened():
        counter += 1
        ret, frame = video_cap.read()
        if not ret:
            break 
        
        gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
        _, thresh = cv2.threshold( gray, 10, 255, cv2.THRESH_BINARY )
        
        # get the circle position estimate
        ret = circle_position( gray )
        
        if ret:
            pts.append( np.append( ret[0], counter ) )
            
        # ret
          
        cv2.imshow( 'frame', thresh )
        
        if cv2.waitKey( 1 ) & 0xFF == ord( 'q' ):
            break
        
    # while
    
    video_cap.release()
    
    cv2.destroyAllWindows()
    
    # find the poitnts
    pts = np.array( pts )
    plt.subplot( 2, 1, 1 )
    plt.plot( pts[:, -1], pts[:, 0] )
    plt.title( 'Trajectory x' )
    
    plt.subplot( 2, 1, 2 )
    plt.plot( pts[:, -1], pts[:, 1] )
    plt.title( 'Trajectory y' )
    
    plt.figure()
    dp = np.diff( pts, 1, axis = 0 )
    plt.plot( pts[:, 0], pts[:, 1], '*' )
    plt.quiver( pts[:-1, 0], pts[:-1, 1], dp[:, 0], dp[:, 1], scale = 1, angles = 'xy', scale_units = 'xy' )
    plt.xlabel( 'x' )
    plt.ylabel( 'y' )
    plt.title( 'Trajectory' )
    
    plt.show()
    
# main


if __name__ == '__main__':
    main()
    print( 'Program terminated.' )

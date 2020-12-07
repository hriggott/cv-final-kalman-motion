import measurement as meas
from LKBB import Linear_Kalman_Black_Box

import cv2
import numpy as np
import matplotlib.pyplot as plt


def main( args ):
    ''' main method '''
    data_dir = "../data/"
    video_file = data_dir + args["vid_file"]
    data_file = data_dir + args["data_file"]
    template_file = data_dir + args['template_file']
    
    # load the image template
    template = cv2.imread( template_file, cv2.IMREAD_ANYCOLOR )
    template_gray = cv2.cvtColor( template, cv2.COLOR_BGR2GRAY )
    t_0, t_1 = template.shape[:2]

    # grab the data csv file
    true_data = np.loadtxt( data_file, skiprows = 1, delimiter = ',' )

    # load the image
    video = cv2.VideoCapture( video_file )
    fps = video.get( cv2.CAP_PROP_FPS )
    N = int( video.get( cv2.CAP_PROP_FRAME_COUNT ) )
    width = video.get( cv2.CAP_PROP_FRAME_WIDTH )
    height = video.get( cv2.CAP_PROP_FRAME_HEIGHT )
    print( 'FPS:', fps, '|', '# Frames:', N )
    print( 'Video frame size: %d x %d' % ( width, height ) )
    
    # Kalman set-up
    dt = 1 / fps
    cov_z = 1  # assume static covariance measurement
    cov_dynamics = 0.1

    model = Linear_Kalman_Black_Box( dt, cov_dynamics, cov_z )
    
    # - initial position
    ( x0, y0 ), _ = meas.find_matches( template, video.read()[1] )
#     x0, y0, *_ = true_data[0]
    model.setInitialPosition( x0, y0 )  # start with the true location
    
    # initialize the data arrays
    # - predicted points
    x_predicted = np.zeros( N )
    y_predicted = np.zeros( N )
    
    x_predicted[0] = x0
    y_predicted[0] = y0
    
    # - measured points
    x_measured = x_predicted.copy()
    y_measured = y_predicted.copy()
    
    # perform the Kalman tracking
    i = 1
    while video.isOpened():
        ret, frame = video.read()
        
        if not ret:
            print( 'Error grabbing frame. Stopping tracking...' )
            break
        
        # if
        
        # perform the prediction
        x_pred, y_pred = model.predict()
        x_predicted[i] = x_pred
        y_predicted[i] = y_pred
        
        # perform the measurement
        # - region of interest
        roi = np.array( [[0, 0], [t_0, t_1]] ) + np.array( [y_pred, x_pred] ).reshape( -1, 2 )
        roi_uncertainty = 30 * np.array( [[-1, -1], [1, 1]] )
        roi_uncertain = roi + roi_uncertainty  # region of interest to look for the image
        
        # -- change roi values to fit within the image
        roi_uncertain[roi_uncertain < 0] = 0 
        roi_uncertain[roi_uncertain[:, 0] > frame.shape[0], 0] = frame.shape[0]
        roi_uncertain[roi_uncertain[:, 1] > frame.shape[1], 1] = frame.shape[1]
        roi_uncertain = roi_uncertain.astype( int )
        
        x_meas, y_meas = meas.find_subimg_matches( template, frame, roi_uncertain )
        x_measured[i] = x_meas
        y_measured[i] = y_meas
        
        # posterior state update
        model.update( x_meas, y_meas )
        
        # counter update
        i += 1
        
    # while
    
    video.release()
    
    # grab the fit data
    x_kalman, y_kalman = model.getCorrectedPositions()
    x_kalman = np.array( x_kalman )
    y_kalman = np.array( y_kalman )
    
    fig = plt.figure()
    ax1 = fig.add_subplot( 2, 1, 1 )
    ax1.plot( x_kalman, y_kalman )
    plt.title( 'trajectory' )
    
    ax2 = fig.add_subplot( 2, 1, 2 )
    ax2.plot( x_kalman, label = 'Kalman x' )
    ax2.plot( y_kalman, label = 'Kalman y' )
    ax2.set_xlabel( 'frame #' )
    ax2.set_ylabel( 'x|y' )
    ax2.legend()
    
    plt.show()
    
    print( 'kalman xhapes, (x, y):', x_kalman.shape, y_kalman.shape )

# main


if __name__ == '__main__':
    args = {}
    args['vid_file'] = "testooo.mov"
    args['data_file'] = args['vid_file'].replace( '.mov', '.csv' )
    args['template_file'] = args['vid_file'].replace( '.mov', '_template.png' )

    main( args )
    print( 'Program completed.' )

# if __main__

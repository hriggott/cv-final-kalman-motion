import measurement as meas
from LKBB import Linear_Kalman_Black_Box
import analysis

import cv2, os
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import time


def annotate_video( video_in_file, x, y, x_actual, y_actual, template ):
    video_out_file = video_in_file.replace( '.mov', '_results.mov' )

    print( video_out_file )
    # input video
    video_in = cv2.VideoCapture( video_in_file )
    print( 'Number of in frames;', video_in.get( cv2.CAP_PROP_FRAME_COUNT ) )
    
    # prepare output video
    fourcc = cv2.VideoWriter_fourcc( 'M', 'J', 'P', 'G' )
    fps = video_in.get( cv2.CAP_PROP_FPS )
    height = int( video_in.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
    width = int( video_in.get( cv2.CAP_PROP_FRAME_WIDTH ) )
    video_out = cv2.VideoWriter( video_out_file, fourcc, fps, ( width, height ) )
    
    t_0, t_1 = template.shape[:2]
    i = 0
    
    while video_in.isOpened():
        ret, frame = video_in.read()
        
        if not ret:
            break
        
        # add the actual rectangle
        xa_i = int( x_actual[i] )
        ya_i = int( y_actual[i] )
        x_i = int( x[i] )
        y_i = int( y[i] )
        
        frame = cv2.rectangle( frame, ( xa_i, ya_i ), ( xa_i + t_0, ya_i + t_1 ),
                              ( 0, 0, 255 ), 6 )
        frame = cv2.rectangle( frame, ( x_i, y_i ), ( x_i + t_0, y_i + t_1 ),
                              ( 0, 255, 0 ), 3 )
        
        video_out.write( frame )
        cv2.imshow( 'annotated video:', frame )
        
        if cv2.waitKey( 5 ) & 0xFF == ord( 'q' ):
            break
    
        # update counter
        i += 1
        
    # while
    video_in.release()
    video_out.release()
    cv2.destroyAllWindows()
    
    print( 'Saved movie:', video_out_file )
    
# annotate_video


def annotate_video_real( video_in_file, x, y, template ):
    video_out_file = video_in_file.replace( '.mov', '_results.mov' )

    print( video_out_file )
    # input video
    video_in = cv2.VideoCapture( video_in_file )
    print( 'Number of in frames;', video_in.get( cv2.CAP_PROP_FRAME_COUNT ) )
    
    # prepare output video
    fourcc = cv2.VideoWriter_fourcc( 'M', 'J', 'P', 'G' )
    fps = video_in.get( cv2.CAP_PROP_FPS )
    height = int( video_in.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
    width = int( video_in.get( cv2.CAP_PROP_FRAME_WIDTH ) )
    video_out = cv2.VideoWriter( video_out_file, fourcc, fps, ( width, height ) )
    
    t_0, t_1 = template.shape[:2]
    i = 0
    
    while video_in.isOpened():
        ret, frame = video_in.read()
        
        if not ret:
            break
        
        # add the actual rectangle
        x_i = int( x[i] )
        y_i = int( y[i] )
        
        frame = cv2.circle( frame, ( x_i, y_i ), 8, ( 0, 0, 255 ), -1 )
        
        video_out.write( frame )
        cv2.imshow( 'annotated video:', frame )
        
        if cv2.waitKey( 5 ) & 0xFF == ord( 'q' ):
            break
    
        # update counter
        i += 1
        
    # while
    video_in.release()
    video_out.release()
    cv2.destroyAllWindows()
    
    print( 'Saved movie:', video_out_file )
    
# annotate_video_real


def check_valid_roi( image_shape, roi, template_shape ):
    
    np_roi = np.array( roi )

    # flatten out of range inputs
    np_roi[np_roi < 0] = 0
    np_roi[np_roi[:, 0] >= image_shape[0], 0] = image_shape[0] - 1
    np_roi[np_roi[:, 1] >= image_shape[1], 1] = image_shape[1] - 1
    
    win_size = np.diff( np_roi, axis = 0 )
    
    retval = np.all( win_size > np.array( template_shape ) / 1.25 )  # want a good portion of the template
        
    return retval

# check_valid_roi


def plot_trajecories( x_kalman, y_kalman, x_predicted, y_predicted, x_measured, y_measured,
                     x_actual = None, y_actual = None ):
    fig = plt.figure()
    ax1 = fig.add_subplot( 1, 3, 1 )
    ax1.plot( x_kalman, y_kalman, label = 'kalman' )
    ax1.plot( x_measured, y_measured, label = 'measured' )
    ax1.plot( x_predicted, y_predicted, label = 'predicted' )
    if ( x_actual is not None ) and ( y_actual is not None ):
        ax1.plot( x_actual, y_actual, label = 'actual' )
    ax1.set_xlabel( 'x' )
    ax1.set_ylabel( 'y' )
    plt.title( 'trajectory' )
    
    ax2 = fig.add_subplot( 1, 3, 2 )
    ax2.plot( x_kalman, label = 'Kalman' )
    ax2.plot( x_measured, label = 'measured' )
    ax2.plot( x_predicted, label = 'predicted' )
    if ( x_actual is not None ):
        ax2.plot( x_actual, label = 'actual' )
    ax2.set_xlabel( 'frame #' )
    ax2.set_ylabel( 'x' )
    ax2.legend()
    
    ax3 = fig.add_subplot( 1, 3, 3 )
    ax3.plot( y_kalman, label = 'Kalman' )
    ax3.plot( y_measured, label = 'measured' )
    ax3.plot( y_predicted, label = 'predicted' )
    if ( y_actual is not None ):
        ax3.plot( y_actual, label = 'actual' )
    ax3.set_xlabel( 'frame #' )
    ax3.set_ylabel( 'y' )
    
    fig.tight_layout()
    
    return fig

# plot_trajectories


def mask_roi( image_shape, roi ):
    X, Y = np.meshgrid( np.arange( image_shape[1] ), np.arange( image_shape[0] ) )
    
    x_lo = roi[0][0]
    x_hi = roi[1][0]
    
    y_lo = roi[0][1]
    y_hi = roi[1][1]
    
    mask = ( ( x_lo <= X ) & ( X <= x_hi ) ) & ( ( y_lo <= Y ) & ( Y <= y_hi ) )
    
    return mask

# mask_roi


def main( args ):
    ''' main method '''
    data_dir = "../data/"
    video_file = data_dir + args["vid_file"]
    data_file = data_dir + args["data_file"]
    template_file = data_dir + args['template_file']
    
    # load the image template
    template = cv2.imread( template_file, cv2.IMREAD_ANYCOLOR )
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
    print( 'Video frame size: %d x %d' % ( height, width ) )
    
    # Kalman set-up
    dt = 1 / fps
    cov_z = 20  # assume static covariance measurement
    cov_dynamics = 0.1
    
    t = np.arange( N ) * dt  # time
    model = Linear_Kalman_Black_Box( dt, cov_dynamics, cov_z )
    
    # - initial position
#     ( x0, y0 ), _ = meas.find_matches( template, video.read()[1] )
#     x0, y0, *_ = true_data[0]
#     model.setInitialPosition( x0, y0 )  # start with the true location
    
    # initialize the data arrays
    # - predicted points
    x_predicted = np.zeros( N )
    y_predicted = np.zeros( N )
    
    # - measured points
    x_measured = x_predicted.copy()
    y_measured = y_predicted.copy()
    
    x_kalman = x_predicted.copy()
    y_kalman = y_predicted.copy()
    
    # perform the Kalman tracking
    i = 0
    while video.isOpened():
        ret, frame = video.read()
        
        if not ret:
            print( 'Error grabbing frame. Stopping tracking...' )
            break
        
        # if
        
        if i == 0:
            ( x0, y0 ), _ = meas.find_matches( template, frame )
            model.setInitialPosition( x0, y0 )  # start with the true location
            x_measured[i] = x0
            y_measured[i] = y0
            
            x_kalman[i] = x0
            y_kalman[i] = y0
            
            x_predicted[i] = x0
            y_predicted[i] = y0
            
        # if
        
        else:
            # perform the prediction
            x_pred, y_pred = model.predict()
            x_predicted[i] = x_pred
            y_predicted[i] = y_pred
            
            # perform the measurement
            # - region of interest
            roi = np.array( [[0, 0], [t_0, t_1]] ) + np.array( [y_pred, x_pred] ).reshape( -1, 2 )
            roi_uncertainty = 50 * np.array( [[-1, -1], [1, 1]] )
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
            x_up, y_up = model.getCorrectedPositions()
            x_kalman[i] = x_up[-1]
            y_kalman[i] = y_up[-1]
            
        # else
        
        # counter update
        i += 1
        
    # while
    
    video.release()
    
    # grab the fit data
#     x_kalman, y_kalman = model.getCorrectedPositions()
    x_kalman = np.array( x_kalman )
    y_kalman = np.array( y_kalman )
    
    x_actual = true_data[:, 0]
    y_actual = true_data[:, 1]

    fig_t = plot_trajecories( x_kalman, y_kalman, x_predicted, y_predicted,
                           x_measured, y_measured, x_actual, y_actual )
    fig_t.tight_layout()    
    fig_p, fig_c = analysis.analyzeResults( x_actual, y_actual,
                                            x_measured, y_measured,
                                            x_predicted, y_predicted,
                                            x_kalman, y_kalman, t )

    plt.show()
    
    fig_t.savefig( video_file.replace( '.mov', '_trajectory.png' ) )
    print( 'Saved figure:', video_file.replace( '.mov', '_trajectory.png' ) )
    
    fig_p.savefig( video_file.replace( '.mov', '_pred-err.png' ) )
    print( 'Saved figure:', video_file.replace( '.mov', '_pred-err.png' ) )
    
    fig_c.savefig( video_file.replace( '.mov', '_kalman-err.png' ) )
    print( 'Saved figure:', video_file.replace( '.mov', '_kalman-err.png' ) )
    
    results = np.vstack( ( t, x_measured, y_measured, x_predicted, y_predicted, x_kalman, y_kalman ) ).T
    np.savetxt( data_file.replace( '.csv', '_results.csv' ), results, delimiter = ',',
               header = 'time, x_measured, y_measured, x_predicted, y_predicted, x_kalman, y_kalman' )
    print( 'saved results file:', data_file.replace( '.csv', '_results.csv' ) )

# main


def main_data( args ):
    data_dir = "../data/"
    video_file = data_dir + args["vid_file"]
    data_file = data_dir + args["data_file"]
    template_file = data_dir
    
    # load in the data
    true_data = np.loadtxt( data_file, skiprows = 1, delimiter = ',' )
    result_data = np.loadtxt( data_file.replace( '.csv', '_results.csv' ),
                             skiprows = 1, delimiter = ',' )
    
    # unpack the data
    t = result_data[:, 0]
    # - actual data
    x_actual = true_data[:, 0]
    y_actual = true_data[:, 1]
    
    # - measured data
    x_measured = result_data[:, 1]
    y_measured = result_data[:, 2]
    
    # - predicted data
    x_predicted = result_data[:, 3]
    y_predicted = result_data[:, 4]
    
    # - kalman data
    x_kalman = result_data[:, 5]
    y_kalman = result_data[:, 6]
    
    fig_t = plot_trajecories( x_kalman, y_kalman, x_predicted, y_predicted,
                             x_measured, y_measured, x_actual, y_actual )
    
    fig_p, fig_c = analysis.analyzeResults( x_actual, y_actual, x_measured, y_measured,
                                            x_predicted, y_predicted, x_kalman, y_kalman, t )
    
    plt.show()
    
    fig_t.savefig( video_file.replace( '.mov', '_trajectory.png' ) )
    print( 'Saved figure:', video_file.replace( '.mov', '_trajectory.png' ) )
    
    fig_p.savefig( video_file.replace( '.mov', '_pred-err.png' ) )
    print( 'Saved figure:', video_file.replace( '.mov', '_pred-err.png' ) )
    
    fig_c.savefig( video_file.replace( '.mov', '_kalman-err.png' ) )
    print( 'Saved figure:', video_file.replace( '.mov', '_kalman-err.png' ) )
    
# main_data


def main_real( args ):
    ''' compute the real video tracking '''
    data_dir = args['data_dir']
    video_file = data_dir + args["vid_file"]
    template_file = data_dir + args['template_file']
    N_update = 10 if 'N_update' not in args.keys() else int( args['N_update'] )  # how often prompted for updates 
    N_skip = 1 if 'N_skip' not in args.keys() else int( args['N_skip'] )  # how many frames to skip per iteration (1 doees no skipping)
    N_skip = N_skip if N_skip >= 1 else 1  # floor it to 1
    
    # load the image template
    template = cv2.imread( template_file, cv2.IMREAD_ANYCOLOR )
    template_gray = cv2.cvtColor( template, cv2.COLOR_BGR2GRAY )
    t_0, t_1 = template.shape[:2]
    
    # load the image
    video = cv2.VideoCapture( video_file )
    fps = video.get( cv2.CAP_PROP_FPS )
    N = int( video.get( cv2.CAP_PROP_FRAME_COUNT ) )
    width = video.get( cv2.CAP_PROP_FRAME_WIDTH )
    height = video.get( cv2.CAP_PROP_FRAME_HEIGHT )
    print( 'FPS:', fps, '|', '# Frames:', N )
    print( 'Video frame size: %d x %d' % ( height, width ) )
    
    # Kalman set-up
    dt = 1 / fps
    cov_z = 80  # assume static covariance measurement
    cov_dynamics = 20
    
    t = np.arange( N ) * dt  # time
    model = Linear_Kalman_Black_Box( dt, cov_dynamics, cov_z )
    
    # data storage
    # - predicted positions
    x_predicted = np.zeros( N )
    y_predicted = np.zeros( N )
    
    # - measured points
    x_measured = x_predicted.copy()
    y_measured = y_predicted.copy()
    
    # - Filtered points
    x_kalman = x_predicted.copy()
    y_kalman = y_predicted.copy()
    
    # SIFT set-up
    sift = cv2.SIFT_create()
    obj_kp, obj_desc = sift.detectAndCompute( template_gray, None )
    
    # perform the Kalman tracking
    i = -1
    t0 = time.time()
    print( 'Beginning Kalman filtering' )
    if N_skip > 1:
        print( f"Will skip every {N_skip} frames" )
    print( f'Will update every {N_update} iterations...' )
    while video.isOpened():
        for _ in range( N_skip ):
            ret, frame = video.read()
            # counter update
            i += 1

        # for
            
        if not ret:
            print( 'Error grabbing frame. Stopping tracking...' )
            break
        
        # if
        
        if i == 0:
            x0, y0 , *_ = meas.feature_matching( frame, template, obj_kp, obj_desc, sift, mask = None )
            model.setInitialPosition( x0, y0 )  # start with the true location
            x_measured[i] = x0
            y_measured[i] = y0
            
            x_kalman[i] = x0
            y_kalman[i] = y0
            
            x_predicted[i] = x0
            y_predicted[i] = y0
            
        # if
        
        else:
            # perform the prediction
            x_pred, y_pred = model.predict()
            x_predicted[i] = x_pred
            y_predicted[i] = y_pred
            
            # perform the measurement
            # - region of interest
            roi = np.array( [[0, 0], [t_0, t_1]] ) + np.array( [y_pred, x_pred] ).reshape( -1, 2 )
            roi_uncertainty = 300 * np.array( [[-1, -1], [1, 1]] )
            roi_uncertain = roi + roi_uncertainty  # region of interest to look for the image
            
            # -- change roi values to fit within the image
            roi_uncertain[roi_uncertain < 0] = 0 
            roi_uncertain[roi_uncertain[:, 0] > frame.shape[0], 0] = frame.shape[0]
            roi_uncertain[roi_uncertain[:, 1] > frame.shape[1], 1] = frame.shape[1]
            roi_uncertain = roi_uncertain.astype( int )
            
            # check to see if we get a valid roi
            if check_valid_roi( frame.shape[:2], roi_uncertain, ( t_0, t_1 ) ):
                roi_mask = mask_roi( ( height, width ), roi_uncertain ).astype( np.uint8 )
                
            else:
                roi_mask = None
                
            # else
            
            x_meas, y_meas , *_ = meas.feature_matching( frame, template, obj_kp, obj_desc, sift, mask = roi_mask )
            
            # take previous measurement if none found
            if x_meas is None:
                x_meas = x_measured[i - 1]
            if y_meas is None:
                y_meas = y_measured[i - 1]
                
            x_measured[i] = x_meas
            y_measured[i] = y_meas
            
            # posterior state update
            model.update( x_meas, y_meas )
            x_up, y_up = model.getCorrectedPositions()
            x_kalman[i] = x_up[-1]
            y_kalman[i] = y_up[-1]
            
        # else
        
        # prompt update
        if ( i % N_update == 0 ) and ( i > 0 ):
            t1 = time.time()
            dt = ( t1 - t0 ) / N_update
            t_left = ( N - i ) * dt
            print( f"Percent complete: {i/N*100:.1f}% | Avg. time per iter.: {dt:.3f}s | Time left: {t_left:.1f}s" )
            t0 = t1
            
        # if
        
    # while
    
    # annoate the video stream
    annotate_video_real( video_file, x_kalman, y_kalman, template )
    
    # plot the trajectories
    fig_t = plot_trajecories( x_kalman, y_kalman, x_predicted, y_predicted, x_measured, y_measured )
    
    # saving figures
    fig_t.savefig( video_file.replace( '.mov', '_trajectory.png' ) )
    print( 'Saved figure:', video_file.replace( '.mov', '_trajectory.png' ) )
    
    # saving files
    results = np.vstack( ( t, x_measured, y_measured, x_predicted, y_predicted, x_kalman, y_kalman ) ).T
    np.savetxt( video_file.replace( '.mov', '_results.csv' ), results, delimiter = ',',
               header = 'time, x_measured, y_measured, x_predicted, y_predicted, x_kalman, y_kalman' )
    
    # release the video
    video.release()
    
# main_real


if __name__ == '__main__':
    args = {}
    args['data_dir'] = '../data/real/'
    args['vid_file'] = "hold_square.mov"
#     args['data_file'] = args['vid_file'].replace( '.mov', '.csv' )
    args['template_file'] = args['vid_file'].replace( '.mov', '_template.png' )
    args['N_skip'] = 1

    main_real( args )
    
#     if os.path.exists( args['data_dir'] + args['data_file'].replace( '.csv', '_results.csv' ) ):
#         print( 'results file exist...' )
#         main_data( args )
#    
#     else:
#         main( args )
#     
#     true_data = np.loadtxt( args['data_dir'] + args['data_file'], skiprows = 1, delimiter = ',' )
#     result_data = np.loadtxt( args['data_dir'] + args['data_file'].replace( '.csv', '_results.csv' ),
#                              skiprows = 1, delimiter = ',' )
#     template = cv2.imread( args['data_dir'] + args['template_file'], cv2.IMREAD_ANYCOLOR )
#     x_actual = true_data[:, 0]
#     y_actual = true_data[:, 1]
#     x_kalman = result_data[:, 5]
#     y_kalman = result_data[:, 6]
#     
#     annotate_video( args['data_dir'] + args['vid_file'], x_kalman, y_kalman, x_actual, y_actual, template )
    
    print( 'Program completed.' )

# if __main__

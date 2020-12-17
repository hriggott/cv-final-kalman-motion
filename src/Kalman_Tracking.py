import measurement as meas
from LKBB import Linear_Kalman_Black_Box
import analysis

import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect


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


def plot_trajecories( x_kalman, y_kalman, x_predicted, y_predicted, x_measured, y_measured,
                     x_actual, y_actual ):
    fig = plt.figure()
    ax1 = fig.add_subplot( 1, 3, 1 )
    ax1.plot( x_kalman, y_kalman, label = 'kalman' )
    ax1.plot( x_measured, y_measured, label = 'measured' )
    ax1.plot( x_predicted, y_predicted, label = 'predicted' )
    ax1.plot( x_actual, y_actual, label = 'actual' )
    ax1.set_xlabel( 'x' )
    ax1.set_ylabel( 'y' )
    plt.title( 'trajectory' )
    
    ax2 = fig.add_subplot( 1, 3, 2 )
    ax2.plot( x_kalman, label = 'Kalman' )
    ax2.plot( x_measured, label = 'measured' )
    ax2.plot( x_predicted, label = 'predicted' )
    ax2.plot( x_actual, label = 'actual' )
    ax2.set_xlabel( 'frame #' )
    ax2.set_ylabel( 'x' )
    ax2.legend()
    
    ax3 = fig.add_subplot( 1, 3, 3 )
    ax3.plot( y_kalman, label = 'Kalman' )
    ax3.plot( y_measured, label = 'measured' )
    ax3.plot( y_predicted, label = 'predicted' )
    ax3.plot( y_actual, label = 'actual' )
    ax3.set_xlabel( 'frame #' )
    ax3.set_ylabel( 'y' )
    
    fig.tight_layout()
    
    return fig

# plot_trajectories


if __name__ == '__main__':
    args = {}
    args['data_dir'] = '../data/'
    args['vid_file'] = "two_balls_noisy_background.mov"
    args['data_file'] = args['vid_file'].replace( '.mov', '.csv' )
    args['template_file'] = args['vid_file'].replace( '.mov', '_template.png' )

    if os.path.exists( args['data_dir'] + args['data_file'].replace( '.csv', '_results.csv' ) ):
        print( 'results file exist...' )
        main_data( args )
   
    else:
        main( args )
    
    true_data = np.loadtxt( args['data_dir'] + args['data_file'], skiprows = 1, delimiter = ',' )
    result_data = np.loadtxt( args['data_dir'] + args['data_file'].replace( '.csv', '_results.csv' ),
                             skiprows = 1, delimiter = ',' )
    template = cv2.imread( args['data_dir'] + args['template_file'], cv2.IMREAD_ANYCOLOR )
    x_actual = true_data[:, 0]
    y_actual = true_data[:, 1]
    x_kalman = result_data[:, 5]
    y_kalman = result_data[:, 6]
    
    annotate_video( args['data_dir'] + args['vid_file'], x_kalman, y_kalman, x_actual, y_actual, template )
    
    print( 'Program completed.' )

# if __main__

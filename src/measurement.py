'''
Created on Nov 22, 2020

@summary: This is a python file containing functions for position estimation

@author: Dimitri Lezcano

'''

import numpy as np
import numpy.linalg as la
import cv2
import matplotlib.pyplot as plt
from scipy.signal import correlate2d    


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


def find_coordinate_image( img, param = {} ):
    ''' Helper function for finding template bounding boxes '''
    pts = []

    # used to determine bounding box for ROI, one-time use
    def mouse_cb( event, x, y, flags, param = {} ):
        nonlocal pts
        print( f'x:{x}px y:{y}px' )
        
        # grab the limit
        if isinstance( param, dict ) and 'limit' in param.keys():
            lim = param['limit']
            
        # if
        
        else:
            lim = np.inf
            
        # else
        
        # handle event mouse buttons
        if event == cv2.EVENT_LBUTTONDOWN:
            print( '\n' + 75 * '=' )
            if len( pts ) >= lim:
                pts = [( x, y )]
                print( f"Cleared pts and added ({x}, {y})." )
                
            # if
            
            else:
                pts.append( ( x, y ) )
                print( f"Appended ({x}, {y})." )
                
            # else
            print( 75 * '=', end = '\n\n' )
        # if
        
        elif event == cv2.EVENT_MBUTTONDOWN:
            print( '\n' + 75 * '=' )
            pts = []
            print( "Cleared pts." )
            print( 75 * '=', end = '\n\n' )
            
        # elif
                
    # mouse_cb
    
    cv2.imshow( 'Mouse Coordinate Find', img )
    cv2.setMouseCallback( 'Mouse Coordinate Find',
                         mouse_cb, param = param )  # used to determine bounding box for ROI
    cv2.waitKey( 0 )
    cv2.destroyAllWindows()
    print()
    
    return pts

# find_coordinate_img


def find_matches( template, image, thresh = None ):
    """Find template in image using normalized cross-correlation.
    
    Args:
    - template (3D uint8 array): BGR template image.
    - image (3D uint8 array): BGR search image.
    
    Return:
    - coords (2-tuple or list of 2-tuples): When `thresh` is None, find the best
        match and return the (x, y) coordinates of the upper left corner of the
        matched region in the original image. When `thresh` is given (and valid),
        find all matches above the threshold and return a list of (x, y)
        coordinates.
    - match_image (3D uint8 array): A copy of the original search images where
        all matched regions are marked.
    """
    # compute the correlation heat map
    template_gray = cv2.cvtColor( template, cv2.COLOR_BGR2GRAY )  # grayscale
    image_gray = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )  # grayscale
    heat_map = normxcorr2( template_gray, image_gray, mode = 'same' )
    
    # copy image for mark up
    match_image = image.copy()
    
    # find coordaintes and mark up match_image
    if thresh is None:
        coords = np.unravel_index( np.argmax( heat_map ), heat_map.shape )
        end_coords = tuple( np.array( coords ) + np.array( template.shape[0:2] ) )
        
        # flip the coords
        coords = tuple( reversed( coords ) )
        end_coords = tuple( reversed( end_coords ) )
        
        # draw the template matched
        cv2.rectangle( match_image, coords, end_coords, [0, 255, 0], 2 )
        
    # if
        
    else:
        I, J = np.nonzero( heat_map >= thresh )  # find threshold coordinates
        
        coords = list( zip( J, I ) )  # turn those coordinates to a list of tuples
        
        # draw the template matched
        end_coord_adjust = np.array( template.shape[1::-1] )
        for coord in coords:
            end_coord = tuple( np.array( coord ) + end_coord_adjust )
            
            cv2.rectangle( match_image, coord, end_coord, [0, 255, 0], 2 )
            
        # for
        
    # else
    
    return coords, match_image

# find_matches


def find_subimg_matches( template, image, roi ):
    top_left = roi[0]
    bttm_right = roi[1]
    
    sub_img = image[top_left[0]:bttm_right[0], top_left[1]:bttm_right[1]]
    
    match_image = image.copy()
    
    coords, sub_match_image = find_matches( template, sub_img )
    
    match_image[top_left[0]:bttm_right[0], top_left[1]:bttm_right[1]] = sub_match_image.copy()
    
    updated_coords = ( coords[0] + top_left[1], coords[1] + top_left[0] )
    
    return updated_coords

# find_subimg_matches


def normxcorr2( template, image, mode = 'valid' ):
    """Do normalized cross-correlation on grayscale images.
    
    When dealing with image boundaries, the "valid" style is used. Calculation
    is performed only at locations where the template is fully inside the search
    image.
    
    Heat maps are measured at the top left of the template.
    
    Args:
    - template (2D float array): Grayscale template image.
    - image (2D float array): Grayscale search image.
    
    Return:
    - scores (2D float array): Heat map of matching scores.
    """
    # helper variables
    t_0, t_1 = template.shape
    template_norm = template / np.linalg.norm( template )
    
    if mode == 'same':
#         pad_ax0_l = ( t_0 ) // 2
#         pad_ax1_l = ( t_1 ) // 2
#         
#         pad_ax0_r = pad_ax0_l if t_0 % 2 == 0 else pad_ax0_l + 1
#         pad_ax1_r = pad_ax1_l if t_1 % 2 == 0 else pad_ax1_l + 1
        
        image = np.pad( image, ( ( 0, t_0 ), ( 0, t_1 ) ), constant_values = 0 )
        
    # if        
        
    # begin norm xcorr 2-D
    retval = np.zeros( ( image.shape[0] - t_0, image.shape[1] - t_1 ) )
    for i in range( retval.shape[0] ):
        for j in range( retval.shape[1] ):
            # get the sub image for this point
            sub_img = image[i:i + t_0, j:j + t_1]
            if np.all( sub_img == 0 ):
                continue
            # if
            sub_img_norm = la.norm( sub_img )
            sub_img = sub_img / sub_img_norm
            
            retval[i, j] = ( sub_img * template_norm ).sum() 
            
        # for
    # for
    
    return retval
    
# normxcorr2

def template_correlate( template, image ):
    ''' Returns the top left-point of the bounding box (x, y)'''
    corr = correlate2d( image, template, mode = 'same' )
    
    x, y = np.unravel_index( np.argmax( corr ), image.shape )
    
    return x, y

# template_correlate


#======================== MAIN METHODS =========================
def main():
    data_dir = '../data/'
#     data_file = data_dir + 'circle-constant_acceleration.mp4'
#     data_file = data_dir + 'video-1606755677.mp4'
#     template_file = data_file.replace( '.mp4', '_template.png' )
    
    data_file = data_dir + 'projectile_motion_2balls.mov'
    template_file = data_file.replace( '.mov', '_template.png' )
        
    video_cap = cv2.VideoCapture( data_file )
    
    counter = 0
    pts = []
    
    track = False
    
    # try grabbing the template
    template = cv2.imread( template_file )
    
    # if no template is found, let's make one
    if isinstance( template, type( None ) ):
        # grab the kth-frame
        for k in range( 20 ):
            ret, frame = video_cap.read()
            
        # for
        
        if not ret:
            print( "No frame found." )
            return
        
        # if
        
        roi = find_coordinate_image( frame, param = {'limit': 2} )  # find the bounding box
        
        print( 'roi = ', roi )
        
        # grab the template
        template = frame[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
        print( template.shape )
        
        # show the template
        cv2.imshow( 'template', template )
        cv2.waitKey( 0 )
#         cv2.destroyAllWindows()
        
        # save the template
        cv2.imwrite( template_file, template )
        print( "wrote template:", template_file )
        
        # release and re-grab the video_capture file
        video_cap.release()
        video_cap = cv2.VideoCapture( data_file )
        
        # test correlation
        print( 'testing find_matches...' )
        coords, match_img = find_matches( template, frame, None )
        print( 'completed find_matches.' )
        
        cv2.imshow( 'find_matches frame', match_img )
        
        cv2.waitKey( 0 )
        cv2.destroyAllWindows()
        
    # if
    
    else:
        print( 'template found:', template_file )
        
    # else
    
    # iterate through the video
    while track and video_cap.isOpened():
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
    if len( pts ) > 0:
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
        
    # if
    
# main


def main_normxcorr():
    data_dir = '../data/'
#     data_file = data_dir + 'circle-constant_acceleration.mp4'
#     data_file = data_dir + 'video-1606755677.mp4'
#     template_file = data_file.replace( '.mp4', '_template.png' )
    
    data_file = data_dir + 'toss_ball.mov'
    template_file = data_file.replace( '.mov', '_template.png' )
        
    video_cap = cv2.VideoCapture( data_file )
    
    counter = 0
    pts = []
    
    # try grabbing the template
    template = cv2.imread( template_file )
    
    # if no template is found, let's make one
    if isinstance( template, type( None ) ):
        # grab the kth-frame
        for k in range( 20 ):
            ret, frame = video_cap.read()
            
        # for
        
        if not ret:
            print( "No frame found." )
            return
        
        # if
        
        roi = find_coordinate_image( frame, param = {'limit': 2} )  # find the bounding box
        
        print( 'roi = ', roi )
        
        # grab the template
        template = frame[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
        print( template.shape )
        
        # show the template
        cv2.imshow( 'template', template )
        cv2.waitKey( 0 )
        cv2.destroyAllWindows()
        
        # save the template
        cv2.imwrite( template_file, template )
        print( "wrote template:", template_file )
        
        # test template matching
        print( 'testing find_matches...' )
        coords, match_img = find_matches( template, frame, None )
        print( 'completed find_matches.' )
        
        cv2.imshow( 'find_matches frame', match_img )
        
        cv2.waitKey( 0 )
        cv2.destroyAllWindows()
        
        # release and re-grab the video_capture file
        video_cap.release()
        video_cap = cv2.VideoCapture( data_file )
    # if
    
    else:
        print( 'template found:', template_file )
        
    # else
    
    # iterate through the video
    while video_cap.isOpened():
        counter += 1
        ret, frame = video_cap.read()
        if not ret:
            break 
        
        # get the circle position estimate
        coords, match_img = find_matches( template, frame, None )
        
        if ret:
            coords_cent = np.array( coords ) + np.array( template.shape[:2] ) // 2
            pts.append( np.append( coords_cent.tolist(), counter ) )
            
        # ret
          
        cv2.imshow( 'frame', match_img )
        
        if cv2.waitKey( 1 ) & 0xFF == ord( 'q' ):
            break
        
    # while
    
    video_cap.release()
    
    cv2.destroyAllWindows()
    
    # find the poitnts
    if len( pts ) > 0:
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
        
    # if
    
# main_normxcorr


if __name__ == '__main__':
    main()
    print( 'Program terminated.' )

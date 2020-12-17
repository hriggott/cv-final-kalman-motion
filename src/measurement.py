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
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans, MiniBatchKMeans
from sklearn.neighbors import LocalOutlierFactor
from skimage import feature

# color HSV ranges
COLOR_HSVRANGE_RED = ( ( 0, 50, 50 ), ( 10, 255, 255 ) )
COLOR_HSVRANGE_BLUE = ( ( 110, 50, 50 ), ( 130, 255, 255 ) )
COLOR_HSVRANGE_GREEN = ( ( 40, 50, 50 ), ( 75, 255, 255 ) )
COLOR_HSVRANGE_YELLOW = ( ( 25, 50, 25 ), ( 35, 255, 255 ) )
COLOR_HSVRANGE_TENNIS = ( COLOR_HSVRANGE_YELLOW[0], COLOR_HSVRANGE_GREEN[1] )


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
        
        Q = np.cov( circles - mu, rowvar = False )[:2,:2]  # the positional covariance
        
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


def color_measurement( image, hsv_range, method = 'standard' ):
    ''' Returns the center of mass of a color blob '''
    image_hsv = cv2.cvtColor( image, cv2.COLOR_BGR2HSV )
    
    color_mask = cv2.inRange( image_hsv, hsv_range[0], hsv_range[1] )
    
    # outlier removal 
    if method == 'standard':
        # grab the non-zero coordinates
        X, Y = np.meshgrid( np.arange( image.shape[1] ), np.arange( image.shape[0] ) )
         
        # get the center of mass
        color_mask_norm = color_mask / color_mask.max()
        x_com = np.sum( X * color_mask_norm ) / np.sum( color_mask_norm )
        y_com = np.sum( Y * color_mask_norm ) / np.sum( color_mask_norm )
        
    # if
    
    elif method == 'kmeans':
        pts = np.vstack( np.nonzero( color_mask ) ).T
        cluster = KMeans( n_clusters = 3, max_iter = 20 ).fit( pts )
        max_lbl = np.argmax( np.bincount( cluster.labels_ ) )
        
        pts_keep = pts[cluster.labels_ == max_lbl]
        
        y_com, x_com = np.mean( pts_keep, axis = 0 ) 
        
        color_mask = np.zeros_like( color_mask )
        color_mask[pts_keep[:, 0], pts_keep[:, 1]] = 255
        
    # elif
    
    elif method == 'minibatch-kmeans':
        raise NotImplementedError( f"'{method}' is not implemented yet." )
    
    # elif
        
    elif method == 'meanshift':
        pts = np.vstack( np.nonzero( color_mask ) ).T
        cluster = MeanShift( bandwidth = estimate_bandwidth( pts ) ).fit( pts )
        max_lbl = np.argmax( np.bincount( cluster.labels_ ) )
        
        pts_keep = pts[cluster.labels_ == max_lbl]  # points to keep
        
        pts_com = np.mean( pts_keep, axis = 0 )
        
        y_com = pts_com[0]
        x_com = pts_com[1]
        
        color_mask = np.zeros_like( color_mask )
        color_mask[pts_keep[:, 0], pts_keep[:, 1]] = 255

    # elif
    
    elif method == 'outlier-weight':
        pts = np.vstack( np.nonzero( color_mask ) ).T
        clf = LocalOutlierFactor( n_neighbors = 150, contamination = 'auto' )
        clf.fit_predict( pts )
        
        # perform a weighting
        weights = np.exp( -np.abs( clf.negative_outlier_factor_ + 1 ) )
        
        # weight the color mask
        color_mask_norm = color_mask / color_mask.max()
        color_mask_norm[pts[:, 0], pts[:, 1]] *= weights
        
        # grab the com
        X, Y = np.meshgrid( np.arange( image.shape[1] ), np.arange( image.shape[0] ) )
        x_com = np.sum( X * color_mask_norm ) / np.sum( color_mask_norm )
        y_com = np.sum( Y * color_mask_norm ) / np.sum( color_mask_norm )
        
    # elif
    
    elif method == 'outlier-thresh':
        pts = np.vstack( np.nonzero( color_mask ) ).T
        clf = LocalOutlierFactor( n_neighbors = 150, contamination = 'auto', p = 1 )
        clf.fit_predict( pts )
        
        # threshold points
        thresh = ( np.abs( clf.negative_outlier_factor_ + 1 ) < 0.05 )
        
        pts_keep = pts[thresh]
        
        # get the com
        pts_com = np.mean( pts_keep, axis = 0 )
        
        y_com = pts_com[0]
        x_com = pts_com[1]
        
        color_mask = np.zeros_like( color_mask )
        color_mask[pts_keep[:, 0], pts_keep[:, 1]] = 255
        
    # elif
    
    else:
        raise NotImplementedError( f"'{method}' is not implemented yet." )
    
    # else

    # grab the masked image
    masked_image = cv2.bitwise_and( image, image, mask = color_mask )
    
    return x_com, y_com, color_mask, masked_image 
    
# color_measurement


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


def feature_matching( image, template, obj_kp, obj_desc, feature_detector: cv2.Feature2D, mask = None,
                      match_method = 'flann' ):
    ''' Method to perform feature detection
        
        @param image: the input image for detection
        @param obj_kp: the object keypoints
        @param obj_desc: the object feature descriptors 
        @param feature_detector (Feature2D class): feature detector of choice
        @param mask (Default = None):  the mask of the image to search through
        
        
        @returns (x_trans, y_trans, xform) which is the translation from the object keypoints
                to the image matches. For use in Kalman Filter, you will need to 
                add this translation to the original detected object (potentially).
                It also returns the affine transform fit (just in case).
                    
    
    '''
    # gray scale the image
    image_gray = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
    
    # begin feature detection
    img_kp, img_desc = feature_detector.detectAndCompute( image_gray, mask )
    
    # begin feature matching
    if match_method == 'flann': 
        # default params from opencv-pytutorials
        FLANN_INDEX_KDTREE = 0
        index_params = dict( algorithm = FLANN_INDEX_KDTREE, trees = 5 )
        search_params = dict( checks = 50 )   
        
        # create flann matcher
        flann = cv2.FlannBasedMatcher( index_params, search_params )
        
        matches = flann.knnMatch( obj_desc, img_desc, k = 2 )
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]  # filter out for good matches
    
        # grab keypoint matches 
        keypts_obj = []
        keypts_img = []
        for idx in range( len( good_matches ) ):
            keypt_obj, keypt_img = get_keypoint_coord_from_match( good_matches, obj_kp, img_kp, idx )
            keypts_obj.append( keypt_obj )
            keypts_img.append( keypt_img )
            
        # for
        keypts_obj = np.array( keypts_obj )
        keypts_img = np.array( keypts_img )
        
        # get affine transform 
        ransac_num_samples = 6
        if len( good_matches ) >= ransac_num_samples:
            xform, _, ransac_matches = ransac( good_matches, obj_kp, img_kp, num_samples = ransac_num_samples )  
            pt_temp_com = np.array( [template.shape[1], template.shape[0]] ).reshape( 1, 1, 2 ) / 2 
            pt_img_com = cv2.transform( pt_temp_com, xform ).squeeze()
            x_com = pt_img_com[0]
            y_com = pt_img_com[1]
            
            image_match = cv2.drawMatches( template, obj_kp, image, img_kp, ransac_matches, None )
            
        # if
        
        elif len( good_matches ) > 0:
            xform = get_affine_transform( keypts_obj, keypts_img )
            pt_temp_com = np.array( [template.shape[1], template.shape[0]] ).reshape( 1, 1, 2 ) / 2 
            pt_img_com = cv2.transform( pt_temp_com, xform ).squeeze()
            x_com = pt_img_com[0]
            y_com = pt_img_com[1]
            
            image_match = cv2.drawMatches( template, obj_kp, image, img_kp, good_matches, None )
            
        # elif
            
        else: 
            x_com, y_com = ( None, None )
            xform = None
            image_match = None
            
        # else
        
    # if
    
    else: 
        raise NotImplementedError( f"'{match_method}' not an implemented matching method" )
    
    # else
    
    return x_com, y_com, xform, image_match
    
# feature_matching


def get_affine_transform( src, dst ):
    ''' estimate affine transform via least squares 
    
        @param src: [N x 2] array
        @param dst: [N x 2] array
        
        @returns m [2 x 3] s.t. m @ src -> dst
    
    '''
    # assertions
    assert( src.shape[1:] == ( 2, ) )
    assert( dst.shape[1:] == ( 2, ) )
     
    # construct linalg lstsq
    A = np.zeros( ( 2 * src.shape[0], 6 ) )
    b = dst.reshape( -1 )
    
    src_h = np.hstack( ( src, np.ones( ( src.shape[0], 1 ) ) ) )  # homogeneous coords
    
    A[0::2,:3] = src_h
    A[1::2, 3:] = src_h
    
    # perform least squares
    t_vect, *_ = np.linalg.lstsq( A, b, rcond = None )
    
    # reshape to a [2 x 3 matrix]
    xform = t_vect.reshape( 2, 3 )
    
    return xform

# get_affine_transform


def get_keypoint_coord_from_match( matches, kp1, kp2, index ):
    """ Gets the keypoint coordinates that correspond to matches[index].
      For example, if we want to get the coordinates of the keypoints corresponding
      to the 10th matching pair, we would be passing

              get_keypoint_coord_from_match(matches, kp1, kp2, 10)

      Then it will return keypoint1, keypoint2, where
      keypoint1: [x, y] coordinate of the keypoint in img1 that corresponds to matches[10]
      keypoint2: [x, y] coordinate of the keypoint in img2 that corresponds to matches[10]
    """
    keypoint1 = [kp1[matches[index].queryIdx].pt[0], kp1[matches[index].queryIdx].pt[1]]
    keypoint2 = [kp2[matches[index].trainIdx].pt[0], kp2[matches[index].trainIdx].pt[1]]
    return keypoint1, keypoint2

# get_keypoint_coord_from_match


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


def ransac( matches, kp1, kp2, num_samples: int = 6, num_trials: int = 3000,
           inlier_thresh: int = 5 ):
    
    # Some parameters
    total_matches = len( matches )
    
    # To keep track of the best transformation
    xform = np.zeros( ( 3, 3 ) )
    most_inliers = 0

    # turn the keypts into a numpy array: rows are the x-y coordinates
    keypts1 = []
    keypts2 = []
    for idx in range( total_matches ):
        keypt1, keypt2 = get_keypoint_coord_from_match( matches, kp1, kp2, idx )
        
        keypts1.append( keypt1 )
        keypts2.append( keypt2 )
    # for
    
    keypts1 = np.array( keypts1 ).astype( np.float32 )
    keypts2 = np.array( keypts2 ).astype( np.float32 )
    inlier_matches = None
    # Loop through num_trials times
    for i in range( num_trials ):

        # Randomly choose num_samples indices from total number of matches
        choices = np.random.choice( total_matches, num_samples, replace = False )

        # Get the matching keypoint coordinates from those indices
        keypts1_choice = keypts1[choices,:]
        keypts2_choice = keypts2[choices,:]

        # get homography   
        xform_i = get_affine_transform( keypts1_choice, keypts2_choice )

        # count the number of inliers
        keypts1_xform_i = cv2.transform( np.expand_dims( keypts1, axis = 1 ), xform_i ).squeeze()
        dists_i = la.norm( keypts1_xform_i - keypts2, axis = 1 )
        num_inliers = np.count_nonzero( dists_i <= inlier_thresh )

        # If for this transformation we have found the most inliers update most_inliers and xform
        if num_inliers > most_inliers:
            most_inliers = num_inliers
            xform = np.copy( xform_i )
            inlier_matches = np.array( matches )[dists_i <= inlier_thresh].tolist()

        # if

    # for
    
    return xform, most_inliers, inlier_matches
    
# ransac


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
    
    data_file = data_dir + 'testooo.mov'
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
    
    data_file = data_dir + 'testooo.mov'
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


def test_hsv_measurement( vid_file ):
    # Set-up
    vid_stream = cv2.VideoCapture( vid_file )  # open video stream
    hsv_range = COLOR_HSVRANGE_TENNIS  # pick a hsv range
    
    while vid_stream.isOpened():
        ret, frame = vid_stream.read()
        
        if not ret:
            print( 'No frame left.' )
            break
        
        # if
        
        # hsv tracking
        x_com, y_com, _, color_frame = color_measurement( frame, hsv_range, method = 'kmeans' ) 
        
        # display results
        color_frame = cv2.circle( color_frame, ( int( x_com ), int( y_com ) ), 5, [0, 0, 255], -1 )
        color_frame = cv2.resize( color_frame, ( color_frame.shape[1] // 2, color_frame.shape[0] // 2 ) )
        frame = cv2.circle( frame, ( int( x_com ), int( y_com ) ), 5, [0, 0, 255], -1 )
        frame = cv2.resize( frame, ( frame.shape[1] // 2, frame.shape[0] // 2 ) )
        
        cv2.imshow( 'masked frame', color_frame )
        cv2.imshow( 'real frame', frame )
        if cv2.waitKey( 1 ) & 0xFF == ord( 'q' ):
            break
        
        # if 
    
    # while
    
    vid_stream.release()

# test_hsv_measurement





def test_feature_tracking( vid_file, template_file ):
    # Set-up
    vid_stream = cv2.VideoCapture( vid_file )  # open video stream
    template = cv2.imread( template_file, cv2.IMREAD_ANYCOLOR )
    template_gray = cv2.cvtColor( template, cv2.COLOR_BGR2GRAY )
    
    # video properties
    width = vid_stream.get( cv2.CAP_PROP_FRAME_WIDTH )
    height = vid_stream.get( cv2.CAP_PROP_FRAME_HEIGHT )
    
    # SIFT
    print( 'Creating SIFT Feature Detector...' )
    sift = cv2.SIFT_create()
    print( 'Computing SIFT Features on template image...' )
    obj_kp, obj_desc = sift.detectAndCompute( template_gray, None )
    print( 'Computed template SIFT features!', end = '\n\n' )
    
    # iterate through the video stream
    while vid_stream.isOpened():
        ret, frame = vid_stream.read()
        
        if not ret:
            print( 'No frame left.' )
            break
        
        # if
        
        # feature tracking
        x_com, y_com, xform, frame_match = feature_matching( frame, template, obj_kp, obj_desc, sift, mask = None )
        
        # display results
        if frame_match is not None:
            frame_match = cv2.resize( frame_match, ( frame_match.shape[1] // 2, frame_match.shape[0] // 2 ) )
            cv2.imshow( 'SIFT matches', frame_match )
            
        # if
        if x_com is not None and y_com is not None:
            frame = cv2.circle( frame, ( int( x_com ), int( y_com ) ), 5, [0, 0, 255], -1 )
            frame = cv2.resize( frame, ( frame.shape[1] // 2, frame.shape[0] // 2 ) )
        
            cv2.imshow( 'annotated video', frame )
            if cv2.waitKey( 1 ) & 0xFF == ord( 'q' ):
                break
            
            # if 
        # if
        
    # while
    
    vid_stream.release()

# test_feature_tracking


if __name__ == '__main__':
#     test_hsv_measurement( '../data/Ball_Drop_real.mov' )
    data_dir = '../data/real/'
    vid_file = data_dir + 'hold_square.mov'
    template_file = vid_file.replace( '.mov', '_template.png' )
    test_feature_tracking( vid_file, template_file )
    print( 'Program terminated.' )

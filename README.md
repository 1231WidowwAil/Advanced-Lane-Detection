# Advanced-Lane-Detection
import cv2
import numpy as np

images = ['test1.jpg','test2.jpg','test3.jpg','test4.jpg','test5.jpg']

def binary_converter(image,threshold):
    binary = np.zeros_like(image)
    binary[(image>= threshold[0])&(image<=threshold[1])]=255
    return binary

def rgb_converter(image):
    b,g,r= cv2.split(image)
    output_rgb= cv2.merge((r,g,b))
    return output_rgb

def hls_converter(image):
    output_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    return output_hls

def channel_seperate(image_hls):
    l = image_hls[:,:,1]
    return l

def white_mask(image):
    rgb_img = rgb_converter(image)
    img_hls_white_bin = np.zeros_like(rgb_img[:,:,0])
    img_hls_white_bin[((rgb_img[:,:,0] >= 204) & (rgb_img[:,:,0] <= 255))
                 & ((rgb_img[:,:,1] >= 204) & (rgb_img[:,:,1] <= 255))
                 & ((rgb_img[:,:,2] >= 204) & (rgb_img[:,:,2] <= 255))                
                ] = 255
    return  img_hls_white_bin

def yellow_mask(image):
    rgb = rgb_converter(image)
    hls_img= hls_converter(rgb)
    img_hls_yellow_bin = hls_img[:,:,0]
    img_hls_yellow_bin[((hls_img[:,:,0] >= 15) & (hls_img[:,:,0] <= 35))
                 & ((hls_img[:,:,1] >= 30) & (hls_img[:,:,1] <= 204))
                 & ((hls_img[:,:,2] >= 115) & (hls_img[:,:,2] <= 255))                
                ] =255
    return img_hls_yellow_bin

def sobel_operator(img,orient = True,dtype = cv2.CV_64F,ksize = 5):

    if (orient == True):

            sobel = cv2.Sobel(img,dtype,1,0,ksize)

    elif(orient == False):

            sobel = cv2.Sobel(img,dtype,0,1,ksize)

    return sobel

def uint8_converter(img):

    img_u8 = np.uint8(255*img/np.max(img))

    return img_u8

def classic_sobel(image):
    threshold = [30,255]
    gray = cv2.GaussianBlur(image,(5,5),0)
    sobel_applied = sobel_operator(gray)
    sobel_abs = np.absolute(sobel_applied)
    sobel_log = uint8_converter(sobel_abs)
    result = binary_converter(sobel_log,threshold)
    return result


def magnitude(image):
    threshold = [120,255]
    gray =cv2.GaussianBlur(image,(5,5),0)
    sobel_magnituded_x = sobel_operator(gray,orient =True)
    sobel_magnituded_y = sobel_operator(gray,orient= False)
    sobel_sg = np.sqrt((sobel_magnituded_x)**2 + (sobel_magnituded_y)**2 )
    sobel_log = uint8_converter(sobel_sg)
    result = binary_converter(sobel_log,threshold)
    return result

def laplacian(image):
    threshold = [100,255]
    gray = cv2.bilateralFilter(image,9,75,75)
    lapi = cv2.Laplacian(gray,cv2.CV_64F)
    lapi_abs = np.absolute(lapi)
    uu=uint8_converter(lapi_abs)
    result = binary_converter(lapi,threshold)
    return result


def apply_sobel(image):
    image = image
    image_l = channel_seperate(hls_converter(rgb_converter(image)))
    classicsobel = classic_sobel(image_l)
    image_magnitude = magnitude(image_l)
    image_laplacian = laplacian(image_l)
    yellow  = yellow_mask(image)
    white = white_mask(image)
    image_bin = np.zeros_like(image_l)
    image_bin[(image_magnitude == 255)|(image_laplacian == 255)|(classicsobel == 255)|(yellow==255)|(white==255)] = 255
    return image_bin




def perspective_warp(img,
                     dst_size=(1280,720),
                     src=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)]),
                     dst=np.float32([(0,0), (1, 0), (0,1), (1,1)])):
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
 
    dst = dst * np.float32(dst_size)
    M = cv2.getPerspectiveTransform(src, dst)
    
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped
left_a, left_b, left_c = [],[],[]
right_a, right_b, right_c = [],[],[]

def get_hist(img):
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
    return hist

def sliding_window(img, nwindows=9, margin=150, minpix = 1, draw_windows=True):
    global left_a, left_b, left_c,right_a, right_b, right_c 
    left_fit_= np.empty(3)
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img))*255

    histogram = get_hist(img)
    midpoint = int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    
    window_height = np.int(img.shape[0]/nwindows)
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
       
        if draw_windows == True:
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (100,255,255), 3) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (100,255,255), 3) 
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    left_a.append(left_fit[0])
    left_b.append(left_fit[1])
    left_c.append(left_fit[2])
    
    right_a.append(right_fit[0])
    right_b.append(right_fit[1])
    right_c.append(right_fit[2])
    
    left_fit_[0] = np.mean(left_a[-10:])
    left_fit_[1] = np.mean(left_b[-10:])
    left_fit_[2] = np.mean(left_c[-10:])
    
    right_fit_[0] = np.mean(right_a[-10:])
    right_fit_[1] = np.mean(right_b[-10:])
    right_fit_[2] = np.mean(right_c[-10:])
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit_[0]*ploty**2 + left_fit_[1]*ploty + left_fit_[2]
    right_fitx = right_fit_[0]*ploty**2 + right_fit_[1]*ploty + right_fit_[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]
    
    
    return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty

def get_curve(img, leftx, rightx,left_fitx,right_fitx):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    y_eval = np.max(ploty)
    ym_per_pix = 30.5/720 
    xm_per_pix = 3.7/720 

    
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    frame_center_pixels = img.shape[1]/2
    camera_position_pixels = ((left_fitx[-1]+right_fitx[-1])/2)
    center_offset_meters = (camera_position_pixels - frame_center_pixels)*xm_per_pix
    return (left_curverad,right_curverad, center_offset_meters)


for i in range(len(images)):
    image =cv2.imread('/home/pi/Desktop/Test/'+images[i])
    sobel_image = apply_sobel(image)
    warped = perspective_warp(sobel_image)
    out_img, curves, lanes, ploty = sliding_window(warped, draw_windows=False)
    curverad =get_curve(image, curves[0], curves[1],lanes[0],lanes[1])
    lane_curve = np.mean([curverad[0], curverad[1]])
    direction = None
    if curverad[2]>=0:
        direction = 'right'
    else:
        direction = 'left'
    print('lane curve:' ,lane_curve,'off center {} ,turn to  {}:'.format(curverad[2],direction))
        

import cv2
import numpy as np

image = cv2.imread('/home/pi/Desktop/roadi.jpg')

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







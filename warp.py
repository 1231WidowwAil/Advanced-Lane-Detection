import cv2
import numpy as np

img = cv2.imread('/home/pi/Desktop/screenshot.png')


line_dst_offset = 100
src = np.float32([[354, 282], \
              [421, 283], \
              [686, img.shape[0]], \
              [224, img.shape[0]]])

dst = np.float32([[src[3][0] + line_dst_offset, 0], \
              [src[2][0] - line_dst_offset, 0], \
              [src[2][0] - line_dst_offset, src[2][1]], \
              [src[3][0] + line_dst_offset, src[3][1]]])

M = cv2.getPerspectiveTransform(src,dst)
Minv = cv2.getPerspectiveTransform(dst,src)

def warp_perspective(image,M):
    
    result = cv2.warpPerspective(image,M,dsize=image.shape[0:2][::-1], flags=cv2.INTER_LINEAR)
    return result

result=warp_perspective(img,M)
cv2.imshow('frame',result)
cv2.waitKey(0)
cv2.destroyAllWindows()


import numpy as np
import cv2


class PerspectiveTransformer:
    '''Calculate the bird's-eye view perspective transform. This transform is used for all P4 images
    and videos because the assumption that the road is a flat plane is valid for this project'''
    def __init__(self):
        OFFSET = 260
        Y_HORIZON = 470
        Y_BOTTOM = 720

        # Define the source points
        src = np.float32([[ 300, Y_BOTTOM],      # bottom left
                          [ 580, Y_HORIZON],     # top left
                          [ 730, Y_HORIZON],     # top right
                          [1100, Y_BOTTOM]])     # bottom right

        # Define the destination points
        dst = np.float32([
            (src[0][0]  + OFFSET, Y_BOTTOM),
            (src[0][0]  + OFFSET, 0),
            (src[-1][0] - OFFSET    , 0),
            (src[-1][0] - OFFSET    , Y_BOTTOM)])

        # Compute the (inverse) perspective transform
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)


    def warp(self, image):
        return cv2.warpPerspective(image, self.M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)


    def inverse_warp(self, image):
        return cv2.warpPerspective(image, self.Minv, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

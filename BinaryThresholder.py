import cv2
import numpy as np


class BinaryThresholder():
    '''Create a binary threshold image for detecting lane lines'''
    def __init__(self, image):
        self.image = image

    def threshold_image(self):
        '''Create a binary threshold image and apply it to self.image'''
        # Convert to YUV color space and HLS color space
        yuv = cv2.cvtColor(self.image, cv2.COLOR_RGB2YUV)
        hls = cv2.cvtColor(self.image, cv2.COLOR_RGB2HLS)

        # Only use the U and V (YUV) and S (HLS) channels and convert to grayscale
        chs = np.stack((yuv[:, :, 1], yuv[:, :, 2], hls[:, :, 2]), axis=2)
        gray = np.mean(chs, 2)

        # Apply Sobel-x and Sobel-y operator
        s_x = abs_sobel(gray, orient='x', sobel_kernel=3)
        s_y = abs_sobel(gray, orient='y', sobel_kernel=3)

        # Apply direction and magnitude of the gradient
        grad_dir = dir_gradient(s_x, s_y)
        grad_mag = mag_gradient(s_x, s_y)

        # Extract yellow colours (for outer left lane lines only)
        ylw = extract_yellow(self.image)

        # Extract pixels that may be covered by shadow
        highlights = extract_highlights(self.image[:, :, 0])

        # Combine everything into one mask
        mask = np.zeros(self.image.shape[:-1], dtype=np.uint8)
        mask[((s_x >= 25) & (s_x <= 255) &
                            (s_y >= 25) & (s_y <= 255)) |
                           ((grad_mag >= 30) & (grad_mag <= 512) &
                            (grad_dir >= 0.2) & (grad_dir <= 1.)) |
                           (ylw == 255) |
                           (highlights == 255)] = 1

        # Ignore anything outside the region where lanes are expected to be (i.e. a trapezoidal shape)
        return region_of_interest(mask)


def abs_sobel(img_ch, orient='x', sobel_kernel=3):
    '''Absolute directional gradient for x or y'''
    if orient == 'x':
        axis = (1, 0)
    elif orient == 'y':
        axis = (0, 1)

    return np.absolute(cv2.Sobel(img_ch, -1, *axis, ksize=sobel_kernel))


def mag_gradient(sobel_x, sobel_y):
    '''Magnitude of the gradient'''
    return np.sqrt(sobel_x ** 2 + sobel_y ** 2).astype(np.uint16)


def dir_gradient(sobel_x, sobel_y):
    '''Direction of the gradient'''
    with np.errstate(divide='ignore', invalid='ignore'):
        abs_grad_dir = np.absolute(np.arctan(sobel_y / sobel_x))
        abs_grad_dir[np.isnan(abs_grad_dir)] = np.pi / 2

    return abs_grad_dir.astype(np.float32)


def extract_yellow(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return cv2.inRange(hsv, (20, 50, 150), (40, 255, 255))


def extract_highlights(image, p=99.9):
    '''Generate an image mask selecting highlights'''
    p = int(np.percentile(image, p) - 30)
    return cv2.inRange(image, p, 255)


def region_of_interest(image):
    '''
    Applies an image mask. Only keeps the region of the image defined by the polygon formed from `vertices`. The rest
    of the image is set to black
    '''
    MASK_X_PAD = 100
    MASK_Y_PAD = 85
    imshape = image.shape
    vertices = np.array([[(0, imshape[0]),                                              # bottom left
                          (imshape[1] / 2 - MASK_X_PAD, imshape[0] / 2 + MASK_Y_PAD),   # top left
                          (imshape[1] / 2 + MASK_X_PAD, imshape[0] / 2 + MASK_Y_PAD),   # top right
                          (imshape[1], imshape[0])]],                                   # bottom right
                        dtype=np.int32)

    mask = np.zeros_like(image)

    # Define a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # Fill pixels inside the polygon defined by "vertices" with the fill color (i.e. white)
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # Return the image only where mask pixels are nonzero
    return cv2.bitwise_and(image, mask)

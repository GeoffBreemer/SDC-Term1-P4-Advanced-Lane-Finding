'''Helper class responsible for detecting lane lines in a single frame/image. Requires a CameraCalibrator and
PerspectiveTransformer object to process each image/frame. Uses a BinaryThresholder to create a binary
threshold image for each image/frame'''
import numpy as np
import cv2
from scipy.signal import find_peaks_cwt
import matplotlib.pyplot as plt
from Line import Line, calc_curvature
from BinaryThresholder import BinaryThresholder

PEAK_MIN_WIDTH = 70                 # used by find_peaks_cwt to detect peaks
PEAK_MAX_WIDTH = 100                # as above
PEAK_SLIDING_WINDOW_NUM = 9         # number of sliding windows to use
PEAK_SLIDING_WINDOW_WIDTH = 50      # number of pixels a sliding window extends to the left and right of its centre

MIN_POINTS_REQUIRED = 3

PARALLEL_THRESH = (0.0003, 0.55)    # threshold for determining whether two lane lines are parallel
DIST_THRESH = (300, 460)            # threshold for appropriate distances between the left and right lane lines

XM_PER_PIXEL = 3.7 / 700            # meters per pixel in x dimension
YM_PER_PIXEL = 30.0 / 720           # meters per pixel in y dimension
IMAGE_MAX_Y = 719                   # bottom of the image (image/frame resolution is fixed at 1280/720)

class LaneFinder:
    '''Detect lanes in images/frames and draw the detected lane and additional information on the original
    image/frame'''
    def __init__(self, calibrator, perspective_transformer, n_frames=1):
        self.n_frames = n_frames
        self.calibrator = calibrator
        self.perspective_transformer = perspective_transformer

        self.left_line = None
        self.right_line = None


    def __are_lines_plausible(self, left, right):
        '''Determines if detected lane lines are plausible lines based on curvature and distance'''
        if len(left[0]) < MIN_POINTS_REQUIRED or len(right[0]) < MIN_POINTS_REQUIRED:
            return False
        else:
            new_left = Line(detected_y=left[0], detected_x=left[1])
            new_right = Line(detected_y=right[0], detected_x=right[1])

            is_parallel = new_left.are_lines_parallel(new_right, threshold=PARALLEL_THRESH)
            dist = new_left.distance_between_lines(new_right)
            is_plausible_dist = DIST_THRESH[0] < dist < DIST_THRESH[1]

            return is_parallel & is_plausible_dist


    def __validate_lines(self, left_points, right_points):
        '''Check detected lines against each other and against previous frames' lines to ensure they are valid lines'''
        left_detected = False
        right_detected = False

        if self.__are_lines_plausible(left_points, right_points):
            left_detected = True
            right_detected = True
        elif self.left_line is not None and self.right_line is not None:
            if self.__are_lines_plausible(left_points, (self.left_line.ally, self.left_line.allx)):
                left_detected = True
            if self.__are_lines_plausible(right_points, (self.right_line.ally, self.right_line.allx)):
                right_detected = True

        return left_detected, right_detected


    def __sliding_window(self, points, warped_thresh_image, img_height, window_height):
        '''Detect lane line centres by moving a sliding window from the bottom to the top of the birds-eye view of a
        binary threshold image'''
        for i in range(1, PEAK_SLIDING_WINDOW_NUM + 1):
            x_from = points[-1][0] - PEAK_SLIDING_WINDOW_WIDTH
            x_to = points[-1][0] + PEAK_SLIDING_WINDOW_WIDTH
            y_from = img_height - (i * window_height)
            y_to = img_height - ((i - 1) * window_height)

            # Find the centres in the current patch
            histogram = np.sum(warped_thresh_image[y_from:y_to, x_from:x_to], axis=0)
            peakind = find_peaks_cwt(histogram, widths=np.arange(PEAK_MIN_WIDTH, PEAK_MAX_WIDTH))

            if (len(peakind) > 0):
                # Add the centre to the list of points
                points.append((peakind[0] + x_from, y_to))

        return np.array(points)


    def __plot_lane(self, img_org, img_warp):
        '''Plot the detected lane, car/lane center indicator and textual information'''
        warp_zero = np.zeros_like(img_warp).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        yrange = np.linspace(0, 720)
        fitted_left_x = self.left_line.best_fit_poly(yrange)
        fitted_right_x = self.right_line.best_fit_poly(yrange)

        pts_left = np.array([np.transpose(np.vstack([fitted_left_x, yrange]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([fitted_right_x, yrange])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank image back to original image space using inverse perspective matrix (Minv)
        newwarp = self.perspective_transformer.inverse_warp(color_warp)

        # Combine the result with the original image
        result = cv2.addWeighted(img_org, 1, newwarp, 0.3, 0)

        # Draw the lane center and car center indicators
        lane_center_x = int(self.center_poly(IMAGE_MAX_Y))
        image_center_x = int(result.shape[1] / 2)
        offset_from_centre = (image_center_x - lane_center_x) * XM_PER_PIXEL               # in meters
        cv2.line(result, (lane_center_x, result.shape[0]), (lane_center_x, result.shape[0] - 20), (255, 255, 255), 6)
        cv2.line(result, (image_center_x, result.shape[0]), (image_center_x, result.shape[0] - 40), (255, 255, 255), 6)
        cv2.line(result, (lane_center_x, result.shape[0] - 2), (image_center_x, result.shape[0] - 2), (255, 255,
                                                                                                       255), 6)
        # Add information overlay rectangle
        font = cv2.FONT_HERSHEY_DUPLEX
        result_overlay = result.copy()
        cv2.rectangle(result_overlay, (10, 10), (610, 150), (255, 255, 255), -1)
        cv2.addWeighted(result_overlay, 0.5, result, 1 - 0.5, 0, result)

        # Add curvature and offset information
        left_right = 'left' if offset_from_centre < 0 else 'right'
        cv2.putText(result, "{:>14}: {:.2f}m to the {}".format("Off-center", abs(offset_from_centre), left_right),
                    (24, 50), font, 1, (255, 255, 255), 2)

        text = "{:.1f}m".format(self.curvature_left)
        cv2.putText(result, "{:>15}: {}".format("Left curvature", text), (19, 90), font, 1, (255, 255, 255), 2)

        text = "{:.1f}m".format(self.curvature_right)
        cv2.putText(result, "{:>15}: {}".format("Right curvature", text), (15, 130), font, 1, (255, 255, 255), 2)

        return result


    def __detect_lane(self, warped_thresh_image, orig_frame):
        """Detect lanes lines in the undisorted original image 'orig_frame' using the warped binary threshold image
        'warped_thres_image'"""
        left_points = []
        right_points = []
        left_detected = right_detected = False

        # 1. Create a histogram for the bottom half of the image and determine the peaks of both lane lines
        histogram = np.sum(warped_thresh_image[int(warped_thresh_image.shape[0] / 2):, :], axis=0)
        center_x = np.int(histogram.shape[0] / 2)
        left_peak = np.argmax(histogram[:center_x])
        right_peak = np.argmax(histogram[center_x:]) + center_x

        # Add the two detected peaks as base points to the array, they serve as sliding window starting points
        left_points.append((left_peak, warped_thresh_image.shape[0]))
        right_points.append((right_peak, warped_thresh_image.shape[0]))

        # 2. Use a sliding window to find all lane line centres in the image for both lane lines
        img_height = warped_thresh_image.shape[0]
        window_height = int(img_height / PEAK_SLIDING_WINDOW_NUM)               # height of each sliding window
        detected_left = self.__sliding_window(left_points, warped_thresh_image, img_height, window_height)
        detected_right = self.__sliding_window(right_points, warped_thresh_image, img_height, window_height)

        # 3. Determine whether proper lane lines were detected by comparing the left and right line curvature and
        # distance between the two lines
        if not left_detected or not right_detected:
            left_detected, right_detected = self.__validate_lines((detected_left[:, 0], detected_left[:, 1]),
                                                                  (detected_right[:, 0], detected_right[:,1]))

        # 4. Update line information
        if left_detected:
            if self.left_line is not None:
                self.left_line.update(y = detected_left[:,0], x = detected_left[:,1])
            else:
                self.left_line = Line(self.n_frames,
                                      detected_y = detected_left[:, 0],
                                      detected_x = detected_left[:, 1])

        if right_detected:
            if self.right_line is not None:
                self.right_line.update(y = detected_right[:,0], x = detected_right[:,1])
            else:
                self.right_line = Line(self.n_frames,
                                       detected_y = detected_right[:,0],
                                       detected_x = detected_right[:,1])

        # 5. Draw the lane and add additional information
        if self.left_line is not None and self.right_line is not None:
            self.curvature_left = calc_curvature(self.left_line.best_fit_poly)
            self.curvature_right = calc_curvature(self.right_line.best_fit_poly)
            self.center_poly = (self.left_line.best_fit_poly + self.right_line.best_fit_poly) / 2
            self.offset = (orig_frame.shape[1] / 2 - self.center_poly(IMAGE_MAX_Y)) * XM_PER_PIXEL

            orig_frame = self.__plot_lane(orig_frame, warped_thresh_image)

        return orig_frame


    def process_image(self, image, diag=False):
        # 1. Correct image distortion (BGR)
        image_undst = self.calibrator.undistort(image)

        # 2. Create thresholded binary image (B/W) and write it to disc (for the README.md)
        image_thres = BinaryThresholder(image_undst).threshold_image()

        color_thresh = np.dstack((image_thres, image_thres, image_thres)) * 255
        color_thresh = cv2.resize(color_thresh, (int(color_thresh.shape[1] / 2), int(color_thresh.shape[0] / 2)))
        cv2.imwrite('./output_images/test5_threshold.jpg', color_thresh)

        # 3. Apply perspective transform to create a bird's-eye view (B/W) and write it to disc (for the README.md)
        image_warp = self.perspective_transformer.warp(image_thres)

        color_warp = np.dstack((image_warp, image_warp, image_warp)) * 255
        color_warp = cv2.resize(color_warp, (int(color_warp.shape[1] / 2), int(color_warp.shape[0] / 2)))
        cv2.imwrite('./output_images/test5_warp.jpg', color_warp)

        # 4. Detect and plot the lane
        image_final = self.__detect_lane(image_warp, image_undst)

        if (diag==False):
            return image_final

        # 5. Optional: return the diagnostic image instead
        image_diag = plot_diagnostics(image, image_undst, image_thres, image_warp, image_final)

        return image_diag


    def process_video_frame(self, frame):
        '''Function that processes videos frame by frame. The steps are identical to those in function process_image,
        except that the latter includes additional code that writes images to disc so that they can be used in the
        accompanying README.md'''
        # 1. Correct image distortion (BGR)
        image_undst = self.calibrator.undistort(frame)

        # 2. Create thresholded binary image (B/W) and write it to disc (for the README.MD)
        image_thres = BinaryThresholder(image_undst).threshold_image()

        # 3. Apply perspective transform to create a bird's-eye view (B/W)
        image_warp = self.perspective_transformer.warp(image_thres)

        # 4. Detect and plot the lane for each frame
        image_final = self.__detect_lane(image_warp, image_undst)
        return image_final        # comment out this line to return the diagnostics view instead

        # 5. Optional: return the diagnostic image instead
        return plot_diagnostics(frame, image_undst, image_thres, image_warp, image_final)


# Helper functions only used during coding/debugging
def show_bgr(image):
    '''Helper function that shows a BGR image using Matplotlib.pyplot, only used during coding/debugging'''
    plt.figure(figsize=(6,4))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


def show_binary(image):
    '''Helper function that shows a binary threshold image using Matplotlib.pyplot, only used during coding/debugging'''
    plt.figure(figsize=(6,4))
    plt.imshow(image, cmap='gray')
    plt.show()


def plot_diagnostics(image, image_undst, image_thres, image_warp, image_final):
    '''Plot a number of images created by the pipeline into a single image'''
    diagScreen = np.zeros((1080, 1280, 3), dtype=np.uint8)

    # Main screen
    diagScreen[0:720, 0:1280] = image_final

    # Four screens along the bottom
    diagScreen[720:1080, 0:320] = cv2.resize(image, (320,360), interpolation=cv2.INTER_AREA)            # original frame
    diagScreen[720:1080, 320:640] = cv2.resize(image_undst, (320,360), interpolation=cv2.INTER_AREA)    # undistorted
    #  original frame

    color_thresh = np.dstack((image_thres, image_thres, image_thres)) * 255
    diagScreen[720:1080, 640:960] = cv2.resize(color_thresh, (320,360), interpolation=cv2.INTER_AREA)   # binary
    # threshold image

    color_warp = np.dstack((image_warp, image_warp, image_warp)) * 255
    diagScreen[720:1080, 960:1280] = cv2.resize(color_warp, (320,360), interpolation=cv2.INTER_AREA)    # birds-eye
    # view of the binary threshold image

    return diagScreen


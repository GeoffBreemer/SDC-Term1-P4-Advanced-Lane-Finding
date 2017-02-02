'''Driver script for Project 4 - Advanced Lane Finding'''
import cv2
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from CameraCalibrator import CameraCalibrator
from PerspectiveTransformer import PerspectiveTransformer
from LaneFinder import LaneFinder

# Camera calibration constants
CAMERA_CAL_IMAGES_PATH = "./camera_cal"
CAMERA_CAL_IMAGE_NAMES = "calibration*.jpg"
CAMERA_CAL_PICKLE_NAME = "calibration_data.p"

# Test image constants
TEST_IMAGES_PATH = './test_images'
TEST_FILE_NAME = "test5.jpg"
OUTPUT_PATH = './output_images'

def prepare_cal_test_image(calibrator):
    '''Save two copies of a checkerboard file and one of the test images to disc: an original copy and an
    undistorted one'''
    # 1. Checkerboard image
    image_org = cv2.imread(CAMERA_CAL_IMAGES_PATH + '/calibration2.jpg')
    image_undistorted = calibrator.undistort(image_org)

    # Resize to 25% of the original size before saving
    image_org = cv2.resize(image_org, (int(image_org.shape[1]/2), int(image_org.shape[0]/2)))
    cv2.imwrite(OUTPUT_PATH + "/calibration2_original.jpg", image_org)
    image_undistorted = cv2.resize(image_undistorted,
                                   (int(image_undistorted.shape[1]/2),
                                    int(image_undistorted.shape[0]/2)))
    cv2.imwrite(OUTPUT_PATH + "/calibration2_undistorted.jpg", image_undistorted)

    # 2. Test image
    image_org = cv2.imread(TEST_IMAGES_PATH + '/test5.jpg')
    image_undistorted = calibrator.undistort(image_org)

    # Resize to 25% of the original size before saving
    image_org = cv2.resize(image_org, (int(image_org.shape[1]/2), int(image_org.shape[0]/2)))
    cv2.imwrite(OUTPUT_PATH + "/test5_original.jpg", image_org)
    image_undistorted = cv2.resize(image_undistorted,
                                   (int(image_undistorted.shape[1]/2),
                                    int(image_undistorted.shape[0]/2)))
    cv2.imwrite(OUTPUT_PATH + "/test5_original_undistorted.jpg", image_undistorted)


def process_single_image(file_name):
    '''Detect lane lines in a single image and display the result'''
    lf = LaneFinder(calibrator, perspective_transformer)

    image = cv2.imread(TEST_IMAGES_PATH + '/' + file_name)
    image_det = lf.process_image(image, False)
    plt.figure(figsize=(6,4))
    plt.imshow(cv2.cvtColor(image_det, cv2.COLOR_BGR2RGB))
    plt.show()

    # Save a copy of the result reduced to 25% of the original size
    image_det = cv2.resize(image_det, (int(image_det.shape[1]/2), int(image_det.shape[0]/2)))
    cv2.imwrite(OUTPUT_PATH + "/test5_final.jpg", image_det)

    # Do it again, but this time create and save the diagnostic view
    image_det = lf.process_image(image, True)
    image_det = cv2.resize(image_det, (int(image_det.shape[1]/2), int(image_det.shape[0]/2)))
    cv2.imwrite(OUTPUT_PATH + "/test5_diag.jpg", image_det)


def process_test_images():
    '''Detect lane lines for the six test images, display the result and write the images along with the detected
    lane to disc'''
    images = []
    for i in range(0, 6):
        images.append(cv2.imread(TEST_IMAGES_PATH + '/test{}.jpg'.format(i+1)))

    # Plot the original image next to the iage showing the detected lane line
    fig, axis = plt.subplots(len(images), 2)
    for row in range(len(images)):
        image_org = images[row]
        axis[row, 0].imshow(cv2.cvtColor(image_org, cv2.COLOR_BGR2RGB))
        axis[row, 0].axis('off')

        # Detect the lane line
        lf = LaneFinder(calibrator, perspective_transformer)
        image_det = lf.process_image(image_org, False)
        cv2.imwrite(OUTPUT_PATH + '/test{}_detected.jpg'.format(row+1), image_det)

        axis[row, 1].imshow(cv2.cvtColor(image_det, cv2.COLOR_BGR2RGB))
        axis[row, 1].axis('off')

    plt.show()          # Manually save the image to disk


def process_video(video_name):
    '''Detect lane lines in an entire video and write the result to disc'''
    lf = LaneFinder(calibrator, perspective_transformer, n_frames=7)

    video_input = VideoFileClip(video_name + ".mp4")
    video_output = video_name + "_output.mp4"
    output = video_input.fl_image(lf.process_video_frame)
    output.write_videofile(video_output, audio=False)


if __name__ == "__main__":
    # Calibrate the camera - one-off
    calibrator = CameraCalibrator(CAMERA_CAL_IMAGES_PATH, CAMERA_CAL_IMAGE_NAMES, CAMERA_CAL_PICKLE_NAME)

    # Create the perspective transform - one-off (assumption: the road is a flat plane)
    perspective_transformer = PerspectiveTransformer()

    # Create an example of an undistorted checkboard image
    prepare_cal_test_image(calibrator)

    # Process a single image
    process_single_image(TEST_FILE_NAME)

    # Process the six test images
    process_test_images()

    # Process a video
    process_video("project_video")
    # process_video("challenge_video")
    # process_video("harder_challenge_video")
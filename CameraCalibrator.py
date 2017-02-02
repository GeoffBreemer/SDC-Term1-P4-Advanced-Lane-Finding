import numpy as np
import cv2
import glob, pickle, os


class CameraCalibrator:
    '''Calibrate the camera using checkerboard images provided with project P4'''
    def __init__(self, images_path, images_names, pickle_name):
        '''Initialise class variables'''
        self.__images_path = images_path
        self.__images_names = images_names
        self.__pickle_name = pickle_name
        self.__calibrated = False


    def __calibrate(self):
        '''Calibrate the camera, pickle and return calibration data'''
        print('Recalibrating the camera...')

        # Number of detectable corners for each checkerboard image
        board_dims = [(9, 5), # 1
                      (9, 6), # 10
                      (9, 6), # 11
                      (9, 6), # 12
                      (9, 6), # 13
                      (9, 6), # 14
                      (9, 6), # 15
                      (9, 6), # 16
                      (9, 6), # 17
                      (9, 6), # 18
                      (9, 6), # 19
                      (9, 6), # 2
                      (9, 6), # 20
                      (9, 6), # 3
                      (5, 6), # 4
                      (7, 6), # 5
                      (9, 6), # 6
                      (9, 6), # 7
                      (9, 6), # 8
                      (9, 6)  # 9
                      ] # note the order: image 1, 10, 11, 12, ..., 19, 2, 20, ..., etc.

        # Arrays holding object points and image points for each image
        objpoints = [] # 3D with z = 0
        imgpoints = [] # 2D corner points

        # Read calibration images
        images = glob.glob(self.__images_path + '/' + self.__images_names)

        for idx, fname in enumerate(sorted(images)):
            # The number of detectable checkerboard corners is different for each image
            (NX, NY) = board_dims[idx]

            # Create the chessboard object points grid or the current image (z = 0)
            objp = np.zeros((NX * NY, 3), np.float32)
            objp[:, :2] = np.mgrid[0:NX, 0:NY].T.reshape(-1, 2)

            # Read and convert the image to grayscale and find the corners
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (NX, NY), None)

            if ret == True:
                imgpoints.append(corners)
                objpoints.append(objp)
                img = cv2.drawChessboardCorners(img, (NX, NY), corners, ret)
            else:
                print('Error detecting checkerboard {}({})'.format(fname, idx))

        img_size = (img.shape[1], img.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

        # Pickle to save time for subsequent runs
        dist_pickle = {}
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"] = dist
        pickle.dump(dist_pickle, open(self.__images_path + '/' + self.__pickle_name, "wb"))

        self.mtx = mtx
        self.dist = dist
        self.__calibrated = True


    def __load_calibration(self):
        '''Load previously pickled camera calibration data'''
        print('Loading pickled camera calibration information...')

        with open(self.__images_path + '/' + self.__pickle_name, mode='rb') as f:
            cal_data = pickle.load(f)

        self.mtx = cal_data['mtx']
        self.dist = cal_data['dist']
        self.__calibrated = True


    def get_matrix_and_coefficients(self):
        '''Determine (either by calibrating the camera or loading previously pickled calibration data) and return
        the camera matrix and distortion coefficients'''
        if os.path.isfile(self.__images_path + '/' + self.__pickle_name):
            self.__load_calibration()
        else:
            self.__calibrate()

        return self.mtx, self.dist


    def undistort(self, image):
        '''Return an undistorted version of the image (calibrates the camera or loads pickled calibration data
        first if the camera was not calibrated)'''
        if  self.__calibrated == False:
            self.get_matrix_and_coefficients()

        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

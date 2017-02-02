import numpy as np

XM_PER_PIXEL = 3.7 / 700                # meters per pixel in x dimension
YM_PER_PIXEL = 30.0 / 720               # meters per pixel in y dimension

IMAGE_MAX_Y = 719

class Line():
    '''Keep track of a single lane line (i.e. the left or right lane line)'''
    def __init__(self, n_frames=1, detected_x=None, detected_y=None):

        self.n_frames = n_frames     # Number of previous frames used to smooth the current frame
        self.best_fit = None         # Polynomial coefficients averaged over the last n_frames frames
        self.current_fit = None      # Polynomial coefficients for the most recent frame
        self.current_fit_poly = None # Polynomial for the current frame's polynomial coefficients
        self.best_fit_poly = None    # Polynomial for the average coefficients over the last n_frames iterations
        self.allx = None             # x values for detected line pixels
        self.ally = None             # y values for detected line pixels

        # Update object variables
        self.update(detected_x, detected_y)


    def update(self, x, y):
        '''Update line properties based on detected x and y coordinates'''
        self.allx = x
        self.ally = y

        # Fit a polynomial and smooth it using previous frames
        self.current_fit = np.polyfit(self.allx, self.ally, 2)
        if self.best_fit is None:
            self.best_fit = self.current_fit
        else:
            self.best_fit = (self.best_fit * (self.n_frames - 1) + self.current_fit) / self.n_frames

        self.current_fit_poly = np.poly1d(self.current_fit)
        self.best_fit_poly = np.poly1d(self.best_fit)


    def are_lines_parallel(self, other_line, threshold=(0, 0)):
        '''Determine whether two lane lines are parallel by comparing fitted polynomial coefficients'''
        diff_coeff_first = np.abs(self.current_fit[0] - other_line.current_fit[0])
        diff_coeff_second = np.abs(self.current_fit[1] - other_line.current_fit[1])

        return diff_coeff_first < threshold[0] and diff_coeff_second < threshold[1]


    def distance_between_lines(self, other_line):
        '''Calculate the distance between the currently fitted polynomials of two lines'''
        return np.abs(self.current_fit_poly(IMAGE_MAX_Y) - other_line.current_fit_poly(IMAGE_MAX_Y))


    def distance_between_best_fit(self, other_line):
        '''Calculate the distance between the best fitted polynomials of two lines'''
        return np.abs(self.best_fit_poly(IMAGE_MAX_Y) - other_line.best_fit_poly(IMAGE_MAX_Y))


def calc_curvature(fit_cr):
    '''Calculate the line curvature (in meters)'''
    y = np.array(np.linspace(0, IMAGE_MAX_Y, num=10))
    x = np.array([fit_cr(x) for x in y])
    y_eval = np.max(y)

    fit_cr = np.polyfit(y * YM_PER_PIXEL, x * XM_PER_PIXEL, 2)
    curverad = ((1 + (2 * fit_cr[0] * y_eval / 2. + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])

    return curverad

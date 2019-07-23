import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import math
from threading import Thread


class Localizer:

    DICTIONARYID = 0
    MARKERLENGTH = 0.0345  # in meters
    XML_FILENAME = "calibration_data1.xml"

    def is_rotation_matrix(self, R):
        Rt = np.transpose(R)
        should_be_identity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - should_be_identity)
        return n < 1e-6

    # Calculates rotation matrix to euler angles x y z
    def rotation_matrix_to_euler_angles(self, R):
        assert (self.is_rotation_matrix(R))
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        return np.array([x, y, z])

    class WebCam:
        def __init__(self):
            self.camera = cv.VideoCapture(1)
            self.input_image = None

        def start(self):
            Thread(target=self._capture_image, args=()).start()

        def _capture_image(self):
            _, self.input_image = self.camera.read()

        def get_image(self):
            return self.input_image

    def __init__(self):
        # self.cam = cv.VideoCapture(1)
        self.cam = self.WebCam()
        self.cam.start()
        # read camera calibration data
        fs = cv.FileStorage(self.XML_FILENAME, cv.FILE_STORAGE_READ)
        if not fs.isOpened():
            print("Invalid camera file")
            exit(-1)
        self.camMatrix = fs.getNode("camera_matrix").mat()
        self.distCoeffs = fs.getNode("distortion_coefficients").mat()

    def pose_from_camera(self):
        # _, input_image = self.cam.read()
        input_image = self.cam.get_image()
        if input_image is not None:
            # detect markers from the input image
            dictionary = aruco.Dictionary_get(self.DICTIONARYID)
            parameters = aruco.DetectorParameters_create()
            marker_corners, marker_ids, _ = aruco.detectMarkers(input_image, dictionary, parameters=parameters)

            if marker_ids is not None:
                # find index of center marker
                index = 0
                index1 = 0
                index2 = 0
                for i in range(len(marker_ids)):
                    if marker_ids[i] == 0:
                        index = i
                    elif marker_ids[i] == 1:
                        index1 = i
                    elif marker_ids[i] == 2:
                        index2 = i

                # pose estimation
                if len(marker_ids) > 2:
                    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                        marker_corners, self.MARKERLENGTH, self.camMatrix, self.distCoeffs)

                    # check if tvecs is none
                    if tvecs is not None:
                        # translate tvecs from relation to camera to a marker
                        tvecs[index][0] -= tvecs[index1][0]

                        # get angle from rotational matrix
                        # convert rotational vector rvecs to rotational matrix
                        # convert euler in relation to a marker
                        rmat = np.empty([3, 3])
                        cv.Rodrigues(rvecs[index][0], rmat)  # cozmo vector to matrix
                        rmat_0 = np.empty([3, 3])
                        cv.Rodrigues(rvecs[index1][0], rmat_0)  # base marker vector to matrix
                        rmat_2 = np.empty([3, 3])
                        cv.Rodrigues(rvecs[index2][0], rmat_2)
                        euler_angle = self.rotation_matrix_to_euler_angles(rmat)  # cozmo relative to camera
                        euler_angle1 = self.rotation_matrix_to_euler_angles(rmat_0)  # base marker relative to camera
                        euler_angle = euler_angle - euler_angle1  # cozmo relative to base marker
                        euler_angle_cube = self.rotation_matrix_to_euler_angles(rmat_2) - euler_angle1  # cube relative

                        # display annotations (IDs and pose)
                        image_copy = input_image.copy()
                        cv.putText(image_copy, "Cozmo Pose", (10, 20), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0, 0))
                        msg = "X(m): " + str(tvecs[index][0][0])
                        cv.putText(image_copy, msg, (10, 45), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0, 0))
                        msg = "Y(m): " + str(tvecs[index][0][1])
                        cv.putText(image_copy, msg, (10, 70), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0, 0))
                        msg = "Angle(rad): " + str(euler_angle[2])
                        cv.putText(image_copy, msg, (10, 95), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0, 0))
                        aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids)
                        cv.imshow("HD Pro Webcam C920", image_copy)
                        cv.waitKey(100)

                        # return the x, y, z coordinates of cozmo in relation to base marker,
                        # x, y, z coordinates of cube relative to base marker, and euler angles
                        return tvecs[index][0], euler_angle, tvecs[index2][0], euler_angle_cube
                    else:
                        return None, None, None, None
                else:
                    return None, None, None, None
            else:
                return None, None, None, None
        else:
            print("No camera found!")
            exit(1)

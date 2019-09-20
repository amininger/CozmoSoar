import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import math
from threading import Thread


class Localizer:

    DICTIONARYID = 0
    MARKERLENGTH = 0.035  # in meters
    XML_FILENAME = "calibration_data.xml"

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
            _, self.input_image = self.camera.read()

        def start(self):
            Thread(target=self._capture_image, args=()).start()

        def _capture_image(self):
            while True:
                self.input_image = self.camera.read()[1]
                cv.imshow("Image", self.input_image)
                cv.waitKey(50)

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
        input_image = self.cam.get_image()
        if input_image is not None:
            # undistort image
            h, w = input_image.shape[:2]
            newcameramtx, roi = cv.getOptimalNewCameraMatrix(self.camMatrix, self.distCoeffs, (w, h), 1, (w, h))
            dst = cv.undistort(input_image, self.camMatrix, self.distCoeffs, None, newcameramtx)
            # crop the image
            x, y, w, h = roi
            input_image = dst[y:y + h, x:x + w]

            # detect markers from the input image
            dictionary = aruco.Dictionary_get(self.DICTIONARYID)
            parameters = aruco.DetectorParameters_create()
            marker_corners, marker_ids, _ = aruco.detectMarkers(input_image, dictionary, parameters=parameters)

            if marker_ids is not None:
                # find index of center marker
                index = 0
                index1 = 0
                for i in range(len(marker_ids)):
                    if marker_ids[i] == 0:
                        index = i
                    elif marker_ids[i] == 1:
                        index1 = i

                # pose estimation
                if len(marker_ids) > 1:
                    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                        marker_corners, self.MARKERLENGTH, self.camMatrix, self.distCoeffs)

                    # check if tvecs is none
                    if tvecs is not None:
                        # translate tvecs from relation to camera to a marker
                        tvecs[index][0] -= tvecs[index1][0]

                        # flip y axis
                        tvecs[index][0][1] = -tvecs[index][0][1]

                        # get angle from rotational matrix
                        # convert rotational vector rvecs to rotational matrix
                        # convert euler in relation to a marker
                        rmat = np.empty([3, 3])
                        cv.Rodrigues(rvecs[index][0], rmat)  # cozmo vector to matrix
                        rmat_0 = np.empty([3, 3])
                        cv.Rodrigues(rvecs[index1][0], rmat_0)  # base marker vector to matrix
                        euler_angle = self.rotation_matrix_to_euler_angles(rmat)  # cozmo relative to camera
                        euler_angle1 = self.rotation_matrix_to_euler_angles(rmat_0)  # base marker relative to camera
                        euler_angle = euler_angle - euler_angle1  # cozmo relative to base marker

                        # flip yaw
                        euler_angle = -euler_angle

                        # fix yaw to -pi to pi
                        if euler_angle[2] < -math.pi:
                            euler_angle[2] += math.pi * 2
                        elif euler_angle[2] > math.pi:
                            euler_angle[2] -= math.pi * 2

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
                        return tvecs[index][0], euler_angle
                    else:
                        return None, None, None, None
                else:
                    return None, None, None, None
            else:
                return None, None, None, None
        else:
            print("No camera found!")
            return None, None, None, None

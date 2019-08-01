import process_markers
import numpy as np
from math import cos, sin, pi, sqrt
from threading import Thread

rotation_matrix = lambda r: np.array([[cos(r), sin(r), 0], [-sin(r), cos(r), 0], [0, 0, 1]])
# translation_matrix = lambda x, y: np.array([[1, 0, 0], [0, 1, 0], [x, y, 1]])


# Reverse x axis
# transposes in functions may need fixing
class CameraLocalizer:

    def __init__(self):
        self.localizer = process_markers.Localizer()
        self.world_position = np.zeros([6])
        self.camera_cube_position = np.zeros([6])
        self.cozmo_origin_location = np.zeros([3])
        self.cozmo_origin_rotation = np.zeros([3])
        self.change_of_bases_r_to_s = np.zeros([3, 3])
        self.change_of_bases_s_to_r = np.zeros([3, 3])
        self.ready = False

    # start a thread for continuously processing markers
    def start(self):
        Thread(target=self._get_cam_pos, args=()).start()

    # get camera coordinates and angle
    def _get_cam_pos(self):
        while True:
            camera_cozmo_coord, camera_cozmo_ang, camera_cube_coord, camera_cube_ang = self.localizer.pose_from_camera()
            if camera_cozmo_coord is not None:
                self.world_position = np.array([camera_cozmo_coord[0], camera_cozmo_coord[1], camera_cozmo_coord[2],
                                                camera_cozmo_ang[0], camera_cozmo_ang[1], camera_cozmo_ang[2]])
                self.ready = True
                # print("world pos: ", self.world_position)
                self.camera_cube_position = np.array([camera_cube_coord[0], camera_cube_coord[1], camera_cube_coord[2],
                                                camera_cube_ang[0], camera_cube_ang[1], camera_cube_ang[2]])

    # pose: xyzrpy
    # pass in: (cozmo pose to cozmo)
    # updates: origin location and rotation
    def recalculate_transform(self, cozmo_pose):
        world_pose = self.world_position
        self.cozmo_origin_rotation = world_pose[5] - cozmo_pose[5]
        undo_cozmo_rotation = rotation_matrix(self.cozmo_origin_rotation)
        # cozmo_pos = np.array([cozmo_pose[0], cozmo_pose[1], 1])
        # origin_to_cozmo = np.matmul(cozmo_pos, undo_cozmo_rotation)
        # cozmo_world_pose - origin_to_cozmo
        # cozmo_rotation = rotation_matrix(self.cozmo_origin_rotation)
        # cozmo_translation = translation_matrix(world_pose[0] - origin_to_cozmo[0],
        #                                        world_pose[1] - origin_to_cozmo[1])
        self.change_of_bases_r_to_s = undo_cozmo_rotation
        self.change_of_bases_s_to_r = np.linalg.inv(self.change_of_bases_r_to_s)

        origin_to_cozmo = np.array([cozmo_pose[0], cozmo_pose[1], cozmo_pose[2]])
        origin_to_cozmo = np.matmul(origin_to_cozmo, self.change_of_bases_r_to_s)
        world_pos = np.array([world_pose[0], world_pose[1], world_pose[2]])
        self.cozmo_origin_location = world_pos - origin_to_cozmo

    # cozmo_pose: xyzrpy
    # pass in: (object position to cozmo, cozmo position to cozmo)
    # def get_world_pose(self, cozmo_pose, object_pose):
    #     print("Cozmo in camera coordinates", self.world_position)
    #     cozmo_to_object = np.array([object_pose[0] - cozmo_pose[0], object_pose[1] - cozmo_pose[1], 1])
    #     cozmo_to_object = np.matmul(cozmo_to_object, self.change_of_bases_r_to_s)
    #     world_to_cozmo = np.array([self.world_position[0], self.world_position[1], 1])
    #     world_to_obj = cozmo_to_object + world_to_cozmo
    #     obj_rot = object_pose[5] + self.cozmo_origin_rotation
    #     print("cube in camera position (actual): ", self.camera_cube_position)
    #     return [world_to_obj[0], world_to_obj[1], 0, 0, 0, obj_rot]

    def get_world_pose(self, object_pose):
        object_pos = np.array([object_pose[0], object_pose[1], object_pose[2]])
        object_pos = np.matmul(object_pos, self.change_of_bases_r_to_s)
        world_to_obj = object_pos + self.cozmo_origin_location
        obj_rot = object_pose[5] + self.cozmo_origin_rotation
        print("cube in camera position (actual): ", self.camera_cube_position)
        # fix yaw to -pi to pi
        if obj_rot < -pi:
            obj_rot += pi * 2
        elif obj_rot > pi:
            obj_rot -= pi * 2
        return [world_to_obj[0], world_to_obj[1], 0, 0, 0, obj_rot]

    # world_pose: xyzrpy
    # pass in: (object position in world, cozmo position to cozmo)
    def get_cozmo_pose(self, world_pose):
        origin_to_object = np.array([world_pose[0] - self.cozmo_origin_location[0],
                                     world_pose[1] - self.cozmo_origin_location[1],
                                     world_pose[2] - self.cozmo_origin_location[2]])
        origin_to_object = np.matmul(origin_to_object, self.change_of_bases_s_to_r)
        cozmo_rot = world_pose[5] - self.cozmo_origin_rotation
        # fix yaw to -pi to pi
        if cozmo_rot < -pi:
            cozmo_rot += pi * 2
        elif cozmo_rot > pi:
            cozmo_rot -= pi * 2
        return [origin_to_object[0], origin_to_object[1], 0, 0, 0, cozmo_rot]

    # Used cozmo pose to
    # def get_cozmo_pose(self, cozmo_pose, world_pose):
    #     cozmo_to_object = np.array([world_pose[0] - self.world_position[0],
    #                                 world_pose[1] - self.world_position[1],
    #                                 world_pose[2] - self.world_position[2]])
    #     cozmo_to_object = np.matmul(cozmo_to_object, self.change_of_bases_s_to_r)
    #     cozmo_pos = np.array([cozmo_pose[0], cozmo_pose[1], cozmo_pose[2]])
    #     origin_to_object = cozmo_pos + cozmo_to_object
    #     cozmo_rot = world_pose[5] - self.cozmo_origin_rotation
    #     return [origin_to_object[0], origin_to_object[1], 0, 0, 0, cozmo_rot]


# def test():
#     cam = CameraLocalizer()
#     cam.world_position = np.array([-0.1, -0.1, 0, 0, 0, -3 * pi / 4])
#     cozmo_position = [0, .05, 0.0, 0, 0, 0]
#     cam.recalculate_transform(cozmo_position)
#     tests = [
#         [sqrt(2)/10, 0.05, 0, 0, 0, 0],
#     ]
#     for t in tests:
#         print("Test: " + str(t))
#         print("world pose of object", cam.get_world_pose_2(t))
#
#
# test()

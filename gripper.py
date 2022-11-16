import trimesh
import numpy as np
from utils import utils


class Gripper:
    def __init__(self) -> None:
        self.load_gripper()
        self.construct_gripper_coordinate_frame()
        self.gripper_points = self.sample_points_on_gripper()

    def sample_points_on_gripper(self, num_points=250):
        points = trimesh.points.PointCloud(self.finger_mesh.sample(num_points))
        return points

    def load_gripper(self):
        self.finger_mesh = trimesh.load("gripper_model/finger.stl")

    def construct_gripper_coordinate_frame(self):
        self.current_coordinate_frame = np.eye(4)

    def get_coordinate_frame(self):
        return self.current_coordinate_frame

    def reset(self):
        self.load_gripper()
        self.sample_points_on_gripper()
        self.current_coordinate_frame = np.eye(4)

    def transform_coordinate_system(self, transformation_matrix):
        self.current_coordinate_frame = transformation_matrix.dot(self.current_coordinate_frame)

    def get_mesh_as_points(self):
        return self.gripper_points.vertices

    def transform_mesh(self, transformation_matrix=np.eye(4)):
        self.transform_coordinate_system(transformation_matrix)
        self.finger_mesh.apply_transform(transformation_matrix)
        self.gripper_points.apply_transform(transformation_matrix)

    def get_mesh(self):
        return self.finger_mesh

    def gripper_animation(self, transformation_idx, transformations, axis):
        self.reset()
        self.transform_mesh(transformations[transformation_idx])
        axis.set_verts(self.finger_mesh.triangles)
        return axis

    def place_gripper_at_pose(self, position, rotation):
        rot_mat_to_align_gripper_and_ref = utils.create_transformation_matrix_from_rot_and_trans(rotation, position)
        self.transform_mesh(rot_mat_to_align_gripper_and_ref)

    def find_collision_free_gripper_rotations(self, object_mesh, gripper_start_position, gripper_start_orientation, query_rotations_around_x_axis):

        self.place_gripper_at_pose(gripper_start_position, gripper_start_orientation[:3, :3])
        rot_matrix_finger_mesh = utils.create_rotation_matrix(
            query_rotations_around_x_axis[1], self.get_coordinate_frame()[:3, 0], gripper_start_position)

        admissable_angles = []
        for query_angle in query_rotations_around_x_axis:
            collision = self.point_based_collision_detection(object_mesh)
            if not collision:
                admissable_angles.append(query_angle)
            self.transform_mesh(rot_matrix_finger_mesh)

        self.reset()
        return np.asarray(admissable_angles)

    def point_based_collision_detection(self, object_mesh, finger_allowed_inside_object=False):
        points_on_gripper = self.sample_points_on_gripper().vertices
        points_inside_object = object_mesh.contains(points_on_gripper)
        if finger_allowed_inside_object:
            points_inside_object = ~points_inside_object
        if points_inside_object.sum() > 0:
            return True
        else:
            return False

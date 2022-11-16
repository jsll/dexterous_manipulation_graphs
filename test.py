import unittest
from DMG import DMG
from area_extractor import SuperVoxel, Centroid
from gripper import Gripper
import trimesh
import networkx as nx
import random
import numpy as np
from utils import utils, graph_utils
import math
import copy
import tempfile
import os


class TestAreaExtractor(unittest.TestCase):
    def setUp(self):
        self.box = trimesh.creation.box()
        self.centroid = Centroid()
        self.correct_graph = nx.from_edgelist([(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 7),
                                              (3, 8), (4, 6), (4, 7), (7, 9), (5, 6), (5, 10), (8, 9), (8, 10), (6, 11), (10, 11), (9, 11)])

    def create_graph_from_edge_list(self):
        graph = self.centroid.create_graph_from_edge_list(self.box.face_adjacency)
        self.assertEqual(graph.nodes(), self.correct_graph.nodes())
        self.assertEqual(graph.edges(), self.correct_graph.edges())
        self.assertEqual(len(graph.nodes()), 12)
        self.assertEqual(len(graph.edges()), 18)

    def test_extract_area_a(self):
        graph = self.centroid.extract_area_a(self.box)
        graph_expected = self.centroid.extract_as_triangles(self.box)
        self.assertTrue(nx.is_isomorphic(graph, graph_expected))

        graph_quad = self.centroid.extract_area_a(self.box, as_quad=True)
        graph_expected = self.centroid.extract_as_quadrilateral(self.box)
        self.assertTrue(nx.is_isomorphic(graph_quad, graph_expected))

    def test_extract_as_triangles(self):
        graph = self.centroid.extract_as_triangles(self.box)
        center_of_faces = self.box.triangles_center
        normals_to_faces = self.box.face_normals
        graph_face_indexes = list(nx.get_node_attributes(graph, "face_idx").keys())
        for node in graph.nodes():
            self.assertIn(graph.nodes[node]["face_idx"], graph_face_indexes)
            self.assertIn(graph.nodes[node]["face_center"], center_of_faces)
            self.assertIn(graph.nodes[node]["face_normal"], normals_to_faces)

    def test_extract_as_quadrilateral(self):
        graph = self.centroid.extract_as_quadrilateral(self.box)
        quad_faces = utils.extract_coplanar_faces_of_mesh(self.box)
        quadrilateral_centers = self.box.triangles_center[quad_faces].mean(axis=1)
        quadrilateral_normals = self.box.face_normals[quad_faces].mean(axis=1)

        graph_face_indexes = [0, 1, 4, 2, 5, 3]
        for i, node in enumerate(graph.nodes()):
            self.assertEqual(graph.nodes[node]["face_idx"], graph_face_indexes[i])
            np.testing.assert_allclose(graph.nodes[node]["face_center"], quadrilateral_centers[graph_face_indexes[i]])
            np.testing.assert_allclose(graph.nodes[node]["face_normal"], quadrilateral_normals[graph_face_indexes[i]])

    def test_extract_edge_list_for_quadrilaterals(self):
        quad_faces = utils.extract_coplanar_faces_of_mesh(self.box)
        triangle_faces_to_quad_faces = self.centroid.map_triangle_faces_to_quad_faces(quad_faces)
        edge_list = self.centroid.extract_edge_list_for_quadrilaterals(
            quad_faces, self.box.face_adjacency, triangle_faces_to_quad_faces)
        expected_edge_list = [[0, 1], [0, 4], [0, 2], [0, 5], [1, 2], [1, 4], [1, 3], [2, 5], [2, 3], [4, 5], [4, 3], [5, 3]]
        for edge in edge_list:
            self.assertIn(edge, expected_edge_list)

        graph_of_area_a = self.centroid.create_graph_from_edge_list(edge_list)
        self.assertEqual(graph_of_area_a.number_of_edges(), 12)
        for node in graph_of_area_a.nodes():
            self.assertEqual(graph_of_area_a.degree[node], 4)

    def test_map_triangle_faces_to_quad_faces(self):
        quad_faces = utils.extract_coplanar_faces_of_mesh(self.box)
        triangle_faces_to_quad_faces = self.centroid.map_triangle_faces_to_quad_faces(quad_faces)
        expected = {0: 0, 2: 0, 1: 1, 5: 1, 3: 2, 8: 2, 10: 3, 11: 3, 4: 4, 6: 4, 7: 5, 9: 5}
        self.assertDictEqual(expected, triangle_faces_to_quad_faces)


class TestGripper(unittest.TestCase):
    def setUp(self):
        self.gripper = Gripper()

    def test_sample_points(self):
        for num_points in range(100, 1000, 10):
            points = self.gripper.sample_points_on_gripper(num_points)
            self.assertEqual(points.shape[0], num_points)
            self.assertEqual(points.shape[1], 3)

        random.seed(10)
        np.random.seed(10)
        correct_points = np.array(
            [[-0.00685129, - 0.00983638, - 0.0279991],
             [-0.01858372, - 0.00026621, - 0.04485715],
             [-0.02595109, 0.0103938, - 0.03681923],
             [-0.01592064, - 0.00996163, - 0.01737488],
             [-0.00611593, 0.00483009, - 0.03045989],
             [-0.02817453, - 0.00152704, - 0.02093866],
             [-0.02615492, - 0.00411351, - 0.01453941],
             [-0.01177783, - 0.00964773, - 0.01422987],
             [-0.00606297, - 0.00702639, - 0.03109327],
             [-0.02126277, - 0.01042506, - 0.03826214]]
        )
        points = self.gripper.sample_points_on_gripper(10)
        points_diff = int(np.abs(points.vertices-correct_points).sum())
        self.assertAlmostEqual(points_diff, 0)

        random.seed()
        np.random.seed()

        # points_first_sampling = self.gripper.sample_points_on_gripper(10)
        # points_second_sampling = self.gripper.sample_points_on_gripper(10)
        # self.assertTrue(np.array_equal(points_first_sampling, points_second_sampling, equal_nan=True))

    def test_gripper_reset(self):

        random.seed(10)
        np.random.seed(10)
        # points_before_transformation = np.asarray(self.gripper.sample_points_on_gripper().vertices)
        gripper_coordinate_system = self.gripper.get_coordinate_frame()
        self.gripper.transform_coordinate_system(np.random.rand(4, 4))
        self.gripper.reset()
        gripper_coordinate_after_reset = self.gripper.get_coordinate_frame()
        # points_after_transformation = np.asarray(self.gripper.sample_points_on_gripper().vertices)
        self.assertTrue(np.array_equal(gripper_coordinate_system, gripper_coordinate_after_reset, equal_nan=True))
        # self.assertTrue(np.array_equal(points_before_transformation, points_after_transformation, equal_nan=True))
        random.seed()
        np.random.seed()

    def test_transform_coordinate_system(self):
        coordinate_frame = self.gripper.get_coordinate_frame()
        self.assertTrue(np.array_equal(coordinate_frame, np.eye(4), equal_nan=True))
        self.gripper.transform_coordinate_system(np.eye(4))
        self.assertTrue(np.array_equal(self.gripper.get_coordinate_frame(), np.eye(4), equal_nan=True))

        random_rotation_matrix = trimesh.transformations.random_rotation_matrix()
        self.gripper.transform_coordinate_system(random_rotation_matrix)
        self.assertTrue(np.array_equal(self.gripper.get_coordinate_frame(), random_rotation_matrix, equal_nan=True))

        self.gripper.transform_coordinate_system(random_rotation_matrix.T)
        np.testing.assert_allclose(self.gripper.get_coordinate_frame(), np.eye(4), equal_nan=True, atol=1e-15)

        self.gripper.reset()
        self.gripper.transform_coordinate_system(random_rotation_matrix)
        self.gripper.transform_coordinate_system(random_rotation_matrix)
        self.assertTrue(np.array_equal(self.gripper.get_coordinate_frame(),
                        random_rotation_matrix.dot(random_rotation_matrix), equal_nan=True))

        self.gripper.reset()
        random_translation_matrix = trimesh.transformations.translation_matrix(np.random.rand(3))
        self.gripper.transform_coordinate_system(random_translation_matrix)
        np.testing.assert_allclose(self.gripper.get_coordinate_frame(), random_translation_matrix, equal_nan=True)
        self.gripper.reset()

        M = trimesh.transformations.concatenate_matrices(random_translation_matrix, random_rotation_matrix)
        self.gripper.transform_coordinate_system(M)
        np.testing.assert_allclose(self.gripper.get_coordinate_frame(), np.eye(4).dot(M), equal_nan=True)

        self.gripper.transform_coordinate_system(np.linalg.inv(M))
        np.testing.assert_allclose(self.gripper.get_coordinate_frame(), np.eye(4), equal_nan=True, atol=1e-15)

    def test_transform_mesh(self):
        initial_vertices = self.gripper.get_mesh().vertices
        initial_gripper_points = self.gripper.gripper_points
        I = np.eye(4)
        self.gripper.transform_mesh(I)
        transformed_gripper_points = self.gripper.gripper_points
        transformed_vertices = self.gripper.get_mesh().vertices
        expected_vertices = I[:3, :3].dot(initial_vertices.T).T+I[:3, 3]
        np.testing.assert_allclose(transformed_vertices, expected_vertices,  equal_nan=True)
        self.assertEqual(initial_gripper_points, transformed_gripper_points)

        T = np.eye(4)
        for _ in range(5):
            random_translation = np.random.rand(3)
            random_rotation = trimesh.transformations.random_rotation_matrix()[:3, :3]
            T[:3, :3] = random_rotation
            T[:3, 3] = random_translation
            self.gripper.transform_mesh(T)
            transformed_vertices = self.gripper.get_mesh().vertices
            expected_vertices = T[:3, :3].dot(initial_vertices.T).T+T[:3, 3]
            transformed_gripper_points = self.gripper.gripper_points
            np.testing.assert_allclose(transformed_vertices, expected_vertices,  equal_nan=True)
            self.assertEqual(initial_gripper_points, transformed_gripper_points)
            self.gripper.reset()

    def test_place_finger_mesh_at_node(self):
        initial_vertices = self.gripper.get_mesh().vertices
        self.gripper.place_gripper_at_pose(np.zeros(3), np.eye(3))
        transformation_matrix = utils.create_transformation_matrix_from_rot_and_trans()
        transformed_vertices = self.gripper.get_mesh().vertices
        expected_vertices = transformation_matrix[:3, :3].dot(initial_vertices.T).T+transformation_matrix[:3, 3]
        np.testing.assert_allclose(transformed_vertices, expected_vertices,  equal_nan=True)
        np.testing.assert_allclose(initial_vertices, expected_vertices,  equal_nan=True)

        for _ in range(5):
            random_translation = np.random.rand(3)
            random_rotation = trimesh.transformations.random_rotation_matrix()[:3, :3]
            self.gripper.place_gripper_at_pose(random_translation, random_rotation)
            transformation_matrix = utils.create_transformation_matrix_from_rot_and_trans(random_rotation, random_translation)
            transformed_vertices = self.gripper.get_mesh().vertices
            expected_vertices = transformation_matrix[:3, :3].dot(initial_vertices.T).T+transformation_matrix[:3, 3]
            np.testing.assert_allclose(transformed_vertices, expected_vertices,  equal_nan=True)
            self.assertFalse(np.allclose(initial_vertices, expected_vertices,  equal_nan=True))
            self.gripper.reset()

    def test_point_based_collision_detection(self):
        test_object = utils.load_mesh("test_objects/box.obj")
        utils.scale_object(test_object, self.gripper.finger_mesh.scale)
        center_of_faces = test_object.triangles_center
        normals_to_faces = test_object.face_normals
        gripper_orientation_matrix = np.eye(3)
        for _ in range(3):
            random_face_index = np.random.randint(0, center_of_faces.shape[0])
            random_face_position = center_of_faces[random_face_index]
            random_face_normal = normals_to_faces[random_face_index]
            y_axis, z_axis = utils.vectors_spanning_plane_to_normal(random_face_normal)
            gripper_orientation_matrix[:, 0] = random_face_normal
            gripper_orientation_matrix[:, 1] = y_axis
            gripper_orientation_matrix[:, 2] = z_axis
            self.gripper.place_gripper_at_pose(random_face_position, gripper_orientation_matrix)
            collision = self.gripper.point_based_collision_detection(test_object)
            self.assertFalse(collision)
            self.gripper.reset()

            pi_rotation_around_z = trimesh.transformations.rotation_matrix(math.pi, [0, 0, 1])
            gripper_orientation_matrix_inside_box = gripper_orientation_matrix.dot(pi_rotation_around_z[:3, :3])
            self.gripper.place_gripper_at_pose(random_face_position, gripper_orientation_matrix_inside_box)
            collision = self.gripper.point_based_collision_detection(test_object)
            self.assertTrue(collision)
            self.gripper.reset()

        test_object = utils.load_mesh("test_objects/fig6_box.obj")
        utils.scale_object(test_object, self.gripper.finger_mesh.scale)
        center_of_face = test_object.triangles_center[23]
        normal_to_face = test_object.face_normals[23]
        gripper_orientation_matrix = np.eye(3)
        orientation_in_collision = np.array([[-0.,          0.16163121, - 0.98685123],
                                             [1.,         0.,  0.],
                                             [0., - 0.98685123, -  0.16163121]])
        orientation_not_in_collision = np.asarray([[-0., - 0.91436028,  0.40490157],
                                                   [1.,         0., - 0.],
                                                   [0.,         0.40490157, 0.91436028]])
        self.gripper.place_gripper_at_pose(center_of_face, orientation_in_collision)
        collision = self.gripper.point_based_collision_detection(test_object)
        self.assertTrue(collision)
        self.gripper.reset()
        self.gripper.place_gripper_at_pose(center_of_face, orientation_not_in_collision)
        collision = self.gripper.point_based_collision_detection(test_object)
        self.assertFalse(collision)

    def test_find_collision_free_gripper_rotation(self):
        test_object = utils.load_mesh("test_objects/fig6_box.obj")
        utils.scale_object(test_object, self.gripper.finger_mesh.scale)
        center_of_face = test_object.triangles_center[23]
        start_orientation_of_gripper = np.eye(4)
        start_orientation_of_gripper[:3, :3] = np.array([[-0.,          0.16163121, - 0.98685123],
                                                         [1.,         0.,  0.],
                                                         [0., - 0.98685123, -  0.16163121]])
        start_orientation_of_gripper[:3, 3] = center_of_face

        collision_free_orientations = self.gripper.find_collision_free_gripper_rotations(
            test_object, center_of_face, start_orientation_of_gripper, np.linspace(0, 2*math.pi, 2))
        expected = np.array([])
        np.testing.assert_array_equal(expected, collision_free_orientations)

        collision_free_orientations = self.gripper.find_collision_free_gripper_rotations(
            test_object, center_of_face, start_orientation_of_gripper, np.linspace(0, 2*math.pi, 3))
        expected = np.array([3.141592653589793])
        np.testing.assert_array_equal(collision_free_orientations, expected)

        collision_free_orientations = self.gripper.find_collision_free_gripper_rotations(
            test_object, center_of_face, start_orientation_of_gripper, np.linspace(0, 2*math.pi, 4))
        expected = np.array([2.0943951023931953, 4.1887902047863905])
        np.testing.assert_array_equal(collision_free_orientations, expected)

        collision_free_orientations = self.gripper.find_collision_free_gripper_rotations(
            test_object, center_of_face, start_orientation_of_gripper, np.linspace(0, 2*math.pi, 5))
        expected = np.array([1.5707963267948966, 3.141592653589793, 4.71238898038469])
        np.testing.assert_array_equal(collision_free_orientations, expected)


class TestUtils(unittest.TestCase):
    def test_cosine_distance(self):
        vector_a = np.array([1, 0, 0])
        self.assertEqual(utils.cosine_distance(vector_a, vector_a), 0)
        self.assertEqual(utils.cosine_distance(vector_a, -vector_a), 2)

        self.assertRaises(AssertionError, utils.cosine_distance, vector_a, 10*vector_a)

        vector_b = np.array([0, 1, 0])
        self.assertEqual(utils.cosine_distance(vector_a, vector_b), 1)

    def test_intersection_of_lists(self):
        list_a = [1, 2, 3]
        self.assertEqual(utils.intersection_of_lists(list_a, list_a), list_a)

        list_a = [1, 2, 3]
        list_b = [3, 4, 10]
        self.assertEqual(utils.intersection_of_lists(list_a, list_b), [3])

        list_b = [3, 2, 10]
        self.assertEqual(utils.intersection_of_lists(list_a, list_b), [2, 3])

    def test_union_intersection_of_lists(self):
        list_a = [1, 2, 3]
        self.assertEqual(utils.union_of_lists(list_a, list_a), list_a)

        list_a = [1, 2, 3]
        list_b = [3, 4, 10]
        self.assertEqual(utils.union_of_lists(list_a, list_b), [1, 2, 3, 4, 10])

        list_b = [3, 2, 10]
        self.assertEqual(utils.union_of_lists(list_a, list_b), [1, 2, 3, 10])

    def test_generate_random_normal_vec_to_x(self):

        for i in range(5):
            vec_x = np.random.rand(3)
            self.assertRaises(AssertionError, utils.generate_random_normal_vec_to_x, vec_x)

            vec_x = vec_x/np.linalg.norm(vec_x)
            normal_vec_to_x = utils.generate_random_normal_vec_to_x(vec_x)

            self.assertAlmostEqual(utils.cosine_distance(vec_x, normal_vec_to_x), 1)

    def test_scale_object_to_fit_finger(self):
        box = trimesh.creation.box((10, 10, 10))
        utils.scale_object(box, 1)
        self.assertEqual(box.extents.tolist(), [1, 1, 1])
        utils.scale_object(box, -1)
        self.assertEqual(box.extents.tolist(), [1, 1, 1])
        utils.scale_object(box, 10)
        self.assertEqual(box.extents.tolist(), [10, 10, 10])

    def test_rotate_axis_of_coordinate_system_to_align_with_vector(self):
        axis = 0
        random_rotation_matrix = trimesh.transformations.random_rotation_matrix()
        vector = np.array([1, 0, 0])
        aligned_coordinate_system = utils.rotate_axis_of_coordinate_system_to_align_with_vector(random_rotation_matrix, vector, axis=axis)
        np.testing.assert_allclose(aligned_coordinate_system[:3, axis], vector, equal_nan=True, atol=1e-14)
        self.assertAlmostEqual(np.linalg.det(aligned_coordinate_system), 1)

        for i in range(5):
            random_rotation_matrix = trimesh.transformations.random_rotation_matrix()
            random_vector = np.random.rand(3)
            random_axis = np.random.randint(3)
            random_vector /= np.linalg.norm(random_vector)
            aligned_coordinate_system = utils.rotate_axis_of_coordinate_system_to_align_with_vector(
                random_rotation_matrix, random_vector, axis=random_axis)
            np.testing.assert_allclose(aligned_coordinate_system[:3, random_axis], random_vector, equal_nan=True)
            self.assertAlmostEqual(np.linalg.det(aligned_coordinate_system), 1)

    def test_rotation_matrix_to_align_vec_a_to_b(self):
        vec_a = np.array([1, 0, 0])

        self.assertRaises(AssertionError, utils.rotation_matrix_to_align_vec_a_to_b, vec_a, 10*vec_a)

        np.testing.assert_allclose(utils.rotation_matrix_to_align_vec_a_to_b(vec_a, vec_a), np.eye(3), equal_nan=True)
        np.testing.assert_allclose(utils.rotation_matrix_to_align_vec_a_to_b(-vec_a, -vec_a), np.eye(3), equal_nan=True)

        for i in range(5):
            vec_a = np.random.rand(3)
            vec_b = np.random.rand(3)
            vec_a /= np.linalg.norm(vec_a)
            vec_b /= np.linalg.norm(vec_b)
            matrix_to_align_a_on_b = utils.rotation_matrix_to_align_vec_a_to_b(vec_a, vec_b)
            np.testing.assert_allclose(matrix_to_align_a_on_b .dot(vec_a), vec_b, equal_nan=True)

    def test_skew(self):
        a = np.random.rand(3)
        b = np.random.rand(3)
        skew_a = utils.skew(a)
        np.testing.assert_allclose(skew_a, -skew_a.T, equal_nan=True)
        np.testing.assert_allclose(np.cross(a, b), np.dot(skew_a, b), equal_nan=True)

    def test_rotate_coordinate_system_along_axis_with_angle(self):
        A = np.eye(4)
        rot_a_360_deg_around_x = utils.rotate_coordinate_system_along_axis_with_angle(A, 2*math.pi, [1, 0, 0])
        rot_a_360_deg_around_y = utils.rotate_coordinate_system_along_axis_with_angle(A, 2*math.pi, [0, 1, 0])
        rot_a_360_deg_around_z = utils.rotate_coordinate_system_along_axis_with_angle(A, 2*math.pi, [0, 0, 1])
        np.testing.assert_allclose(rot_a_360_deg_around_x, A, equal_nan=True, atol=1e-15)
        np.testing.assert_allclose(rot_a_360_deg_around_y, A, equal_nan=True, atol=1e-15)
        np.testing.assert_allclose(rot_a_360_deg_around_z, A, equal_nan=True, atol=1e-15)
        rot_a_360_deg_around_x = utils.rotate_coordinate_system_along_axis_with_angle(A, 2*math.pi, [-1, 0, 0])
        rot_a_360_deg_around_y = utils.rotate_coordinate_system_along_axis_with_angle(A, 2*math.pi, [0, -1, 0])
        rot_a_360_deg_around_z = utils.rotate_coordinate_system_along_axis_with_angle(A, 2*math.pi, [0, 0, -1])
        np.testing.assert_allclose(rot_a_360_deg_around_x, A, equal_nan=True, atol=1e-15)
        np.testing.assert_allclose(rot_a_360_deg_around_y, A, equal_nan=True, atol=1e-15)
        np.testing.assert_allclose(rot_a_360_deg_around_z, A, equal_nan=True, atol=1e-15)

        rot_a_57_deg_around_x = utils.rotate_coordinate_system_along_axis_with_angle(A, math.radians(57), [1, 0, 0])
        rot_57_deg = np.eye(4)
        rot_57_deg[:3, :3] = np.array([[1.0000000,  0.0000000,  0.0000000],
                                       [0.0000000,  0.5446391, -0.8386706],
                                       [0.0000000,  0.8386706,  0.5446391]])
        np.testing.assert_allclose(rot_a_57_deg_around_x, rot_57_deg, equal_nan=True, atol=1e-7)

        random_rotation_matrix = trimesh.transformations.random_rotation_matrix()
        rot_a_57_deg_around_x = utils.rotate_coordinate_system_along_axis_with_angle(
            random_rotation_matrix, math.radians(math.pi/2), [1, 0, 0])

        np.testing.assert_allclose(random_rotation_matrix[:3, 0], rot_a_57_deg_around_x[:3, 0], equal_nan=True, atol=1e-7)

    def test_map_2pi_to_zero_in_list(self):
        test_angles, _ = utils.create_angles_from_resolution(9)
        test_angles = utils.map_2pi_to_zero_in_list(test_angles)
        expected = np.array([0, 0.6981317, 1.3962634, 2.0943951, 2.7925268, 3.4906585,
                             4.1887902, 4.88692191, 5.58505361])
        np.testing.assert_allclose(expected, test_angles, equal_nan=True)
        test_angles, _ = utils.create_angles_from_resolution(9)
        # Remove zero from the array
        test_angles = np.delete(test_angles, 0)
        test_angles = utils.map_2pi_to_zero_in_list(test_angles)
        expected = np.array([0, 0.6981317, 1.3962634, 2.0943951, 2.7925268, 3.4906585,
                             4.1887902, 4.88692191, 5.58505361])
        np.testing.assert_allclose(expected, test_angles, equal_nan=True)

        test_angles, _ = utils.create_angles_from_resolution(9)
        # Remove 2pi from the array
        test_angles = np.delete(test_angles, -1)
        test_angles = utils.map_2pi_to_zero_in_list(test_angles)
        expected = np.array([0, 0.6981317, 1.3962634, 2.0943951, 2.7925268, 3.4906585,
                             4.1887902, 4.88692191, 5.58505361])
        np.testing.assert_allclose(expected, test_angles, equal_nan=True)

        test_angles, _ = utils.create_angles_from_resolution(9)
        # Remove 0 and 2pi from the array
        test_angles = np.delete(test_angles, [0, -1])
        test_angles = utils.map_2pi_to_zero_in_list(test_angles)
        expected = np.array([0.6981317, 1.3962634, 2.0943951, 2.7925268, 3.4906585,
                             4.1887902, 4.88692191, 5.58505361])
        np.testing.assert_allclose(expected, test_angles, equal_nan=True)

    def test_find_overlapping_angles_within_threshold(self):
        test_angles, _ = utils.create_angles_from_resolution(2)
        overlapping_angles = utils.find_overlapping_angles_within_threshold(test_angles, test_angles+1.0, 0.99)
        self.assertFalse(overlapping_angles.size > 0)
        overlapping_angles = utils.find_overlapping_angles_within_threshold(test_angles, test_angles+1.0, 1.000001)
        expected = np.array([0.,         3.14159265, 6.28318531])
        np.testing.assert_allclose(expected, overlapping_angles, equal_nan=True)

        overlapping_angles = utils.find_overlapping_angles_within_threshold(test_angles, test_angles[0], 10)
        expected = np.array([0.,         3.14159265, 6.28318531])
        np.testing.assert_allclose(expected, overlapping_angles, equal_nan=True)

        overlapping_angles = utils.find_overlapping_angles_within_threshold(test_angles, test_angles-1.0, 1.000001)
        np.testing.assert_allclose(expected, overlapping_angles, equal_nan=True)

        overlapping_angles = utils.find_overlapping_angles_within_threshold(test_angles, test_angles[:2]+1.0, 1.000001)
        expected = np.array([0.,         3.14159265])
        np.testing.assert_allclose(expected, overlapping_angles, equal_nan=True)

        overlapping_angles = utils.find_overlapping_angles_within_threshold(test_angles, np.array([]), 1.000001)
        expected = np.array([])
        np.testing.assert_allclose(expected, overlapping_angles, equal_nan=True)

    def test_find_angle_sign_for_aligning_frames_around_axis(self):
        # Positive rotation
        random_rotation_matrix = trimesh.transformations.random_rotation_matrix()
        pos_pi_rotation_around_x = trimesh.transformations.rotation_matrix(math.pi/2, [1, 0, 0])
        pi_rotated_random_matrix = random_rotation_matrix.dot(pos_pi_rotation_around_x)
        z_axis_random_rotation_matrix = random_rotation_matrix[:3, 3]
        z_axis_pi_rotated_random_rotation_matrix = pi_rotated_random_matrix[:3, 3]
        angle_between_z_vectors = np.arccos(z_axis_random_rotation_matrix.dot(z_axis_pi_rotated_random_rotation_matrix))
        angle_sign = utils.find_angle_sign_for_aligning_frames_around_axis(
            pi_rotated_random_matrix[:3, :3], random_rotation_matrix[:3, :3], angle_between_z_vectors)
        self.assertEqual(angle_sign, 1)

        random_rotation_matrix = trimesh.transformations.random_rotation_matrix()
        neg_pi_rotation_around_x = trimesh.transformations.rotation_matrix(-math.pi/2, [1, 0, 0])
        pi_rotated_random_matrix = random_rotation_matrix.dot(neg_pi_rotation_around_x)
        z_axis_random_rotation_matrix = random_rotation_matrix[:3, 3]
        z_axis_pi_rotated_random_rotation_matrix = pi_rotated_random_matrix[:3, 3]
        angle_between_z_vectors = np.arccos(z_axis_random_rotation_matrix.dot(z_axis_pi_rotated_random_rotation_matrix))
        angle_sign = utils.find_angle_sign_for_aligning_frames_around_axis(
            pi_rotated_random_matrix[:3, :3], random_rotation_matrix[:3, :3], angle_between_z_vectors)
        self.assertEqual(angle_sign, -1)

    def test_object_ray_intersections(self):
        # For now, I will not write a test for this function as it is based on trimesh
        # which is well tested.
        pass

    def test_find_angle_from_set_closest_to_query_angle(self):
        angle_set_1 = np.asarray(range(5))
        angle_set_2 = np.asarray(range(1))
        for i in range(5):
            closest_angle_1 = utils.find_angle_from_set_closest_to_query_angle(angle_set_1, i)
            closest_angle_2 = utils.find_angle_from_set_closest_to_query_angle(angle_set_2, i)
            self.assertEqual(closest_angle_1, i)
            self.assertEqual(closest_angle_2, 0)

        closest_angle = utils.find_angle_from_set_closest_to_query_angle(angle_set_1, 2*math.pi)
        self.assertEqual(closest_angle, 0)

    def test_find_disjoint_angular_sets(self):
        test_angles, resolution = utils.create_angles_from_resolution(9)
        set_of_disjoint_angles = utils.find_disjoint_angular_sets(test_angles, resolution)
        expected = [test_angles.tolist()]
        self.assertListEqual(set_of_disjoint_angles, expected)
        self.assertEqual(len(set_of_disjoint_angles), 1)

        test_angles, resolution = utils.create_angles_from_resolution(9)
        test_angles = test_angles[::2]
        set_of_disjoint_angles = utils.find_disjoint_angular_sets(test_angles, resolution)
        self.assertEqual(len(set_of_disjoint_angles), 5)
        expected = [[0.0], [1.3962634015954636],
                    [2.792526803190927], [4.1887902047863905], [5.585053606381854]]
        self.assertListEqual(set_of_disjoint_angles, expected)

        test_angles, resolution = utils.create_angles_from_resolution(10)
        test_angles = np.delete(test_angles, np.arange(0, test_angles.size, 3))
        set_of_disjoint_angles = utils.find_disjoint_angular_sets(test_angles, resolution)
        self.assertEqual(len(set_of_disjoint_angles), 4)
        expected = [[0.6283185307179586, 1.2566370614359172], [
            2.5132741228718345, 3.141592653589793], [4.39822971502571, 5.026548245743669], [6.283185307179586]]
        self.assertListEqual(set_of_disjoint_angles, expected)

    def test_join_sets_containing_zero_and_two_pi(self):
        test_angles, resolution = utils.create_angles_from_resolution(9)
        set_of_disjoint_angles = utils.find_disjoint_angular_sets(test_angles, resolution)
        self.assertRaises(AssertionError, utils.join_sets_containing_zero_and_two_pi, set_of_disjoint_angles)

        test_angles = test_angles[:: 2]
        set_of_disjoint_angles = utils.find_disjoint_angular_sets(test_angles, resolution)
        expected = copy.deepcopy(set_of_disjoint_angles)
        self.assertEqual(len(set_of_disjoint_angles), 5)
        utils.join_sets_containing_zero_and_two_pi(set_of_disjoint_angles)
        self.assertEqual(len(set_of_disjoint_angles), 5)
        self.assertListEqual(expected, set_of_disjoint_angles)

        set_of_disjoint_angles = [[0.0], [1.2566370614359172, 1.8849555921538759], [3.141592653589793,
                                                                                    3.7699111843077517], [5.026548245743669, 5.654866776461628, 6.283185307179586]]
        self.assertEqual(len(set_of_disjoint_angles), 4)
        utils.join_sets_containing_zero_and_two_pi(set_of_disjoint_angles)
        self.assertEqual(len(set_of_disjoint_angles), 3)
        expected = [[0.0, 5.026548245743669, 5.654866776461628, 6.283185307179586], [1.2566370614359172, 1.8849555921538759], [3.141592653589793,
                                                                                                                               3.7699111843077517]]
        self.assertListEqual(expected, set_of_disjoint_angles)

    def test_create_angles_from_resolution(self):

        self.assertRaises(AssertionError, utils.create_angles_from_resolution, -1)
        self.assertRaises(AssertionError, utils.create_angles_from_resolution, 2.5)
        for i in range(1, 10, 1):
            angle_discretization, resolution = utils.create_angles_from_resolution(i)
            self.assertEqual(len(angle_discretization), i+1)
            self.assertEqual(resolution, angle_discretization[1])
            self.assertIn(0, angle_discretization)
            self.assertIn(2*math.pi, angle_discretization)

    def test_compute_angle_around_x_axis_for_aligning_z_axis_of_two_frames(self):
        for i in range(10):
            random_rotation_matrix = trimesh.transformations.random_rotation_matrix()
            angle_to_rotate = utils.compute_angle_around_x_axis_for_aligning_z_axis_of_two_frames(
                random_rotation_matrix, random_rotation_matrix[: 3, 2], random_rotation_matrix)
            self.assertAlmostEqual(angle_to_rotate, 0, 6)

        Ry = trimesh.transformations.rotation_matrix(math.pi/2, [0, 1, 0])
        random_rotation_matrix_rot_90_deg_around_y = random_rotation_matrix.dot(Ry)

        self.assertRaises(AssertionError, utils.compute_angle_around_x_axis_for_aligning_z_axis_of_two_frames,
                          random_rotation_matrix, random_rotation_matrix[: 3, 2], random_rotation_matrix_rot_90_deg_around_y)

        Rz = trimesh.transformations.rotation_matrix(0.00001, [0, 0, 1])
        random_rotation_matrix_small_rot_around_z = random_rotation_matrix.dot(Rz)

        self.assertRaises(AssertionError, utils.compute_angle_around_x_axis_for_aligning_z_axis_of_two_frames,
                          random_rotation_matrix, random_rotation_matrix[: 3, 2], random_rotation_matrix_small_rot_around_z)

        for _ in range(10):
            random_rotation_matrix = trimesh.transformations.random_rotation_matrix()
            random_angle = np.random.uniform(0, 2*math.pi)
            Rx = trimesh.transformations.rotation_matrix(random_angle, [1, 0, 0])
            perturbed_rotation_matrix_around_x_axis = random_rotation_matrix.dot(Rx)
            rotation_angle_to_unperturb_the_rotation_matrix = utils.compute_angle_around_x_axis_for_aligning_z_axis_of_two_frames(
                random_rotation_matrix, random_rotation_matrix[: 3, 2], perturbed_rotation_matrix_around_x_axis)
            self.assertAlmostEqual(random_angle, rotation_angle_to_unperturb_the_rotation_matrix)

    def test_vectors_spanning_plane_to_normal(self):
        random_rotation_matrix = np.eye(3)
        for i in range(5):
            random_normal_vector = np.random.rand(3)
            random_normal_vector /= np.linalg.norm(random_normal_vector)
            y_axis, z_axis = utils.vectors_spanning_plane_to_normal(random_normal_vector)
            random_rotation_matrix[:, 0] = random_normal_vector
            random_rotation_matrix[:, 1] = y_axis
            random_rotation_matrix[:, 2] = z_axis

            self.assertAlmostEqual(np.linalg.det(random_rotation_matrix), 1)
            self.assertAlmostEqual(random_normal_vector.dot(y_axis), 0)
            self.assertAlmostEqual(random_normal_vector.dot(z_axis), 0)
            self.assertAlmostEqual(y_axis.dot(z_axis), 0)

    def test_euclidean_distance_between_points(self):
        point_a = np.asarray([0, 0, 0])
        distance = utils.euclidean_distance_between_points(point_a, point_a)
        self.assertEqual(distance, 0)
        point_b = np.asarray([1, 1, 1])
        distance = utils.euclidean_distance_between_points(point_a, point_b)
        self.assertEqual(distance, np.sqrt(3))

        distance = utils.euclidean_distance_between_points(-point_b, point_b)
        self.assertEqual(distance, np.sqrt(4+4+4))

    def test_query_angle_in_list_of_angles(self):
        angles = np.linspace(0, 2*math.pi, 10)
        threshold = angles[1]
        for angle in angles:
            self.assertTrue(utils.query_angle_in_list_of_angles(angles, angle, 0))
            self.assertFalse(utils.query_angle_in_list_of_angles(angles, angle+1e-12, 0))
            self.assertTrue(utils.query_angle_in_list_of_angles(angles, angle+1e-12, threshold/2))
            self.assertTrue(utils.query_angle_in_list_of_angles(angles, angle+threshold/2, threshold/2))

    def test_apply_transformation_to_list_of_poses(self):
        transformation = np.eye(4)
        rotations = []
        for _ in range(5):
            rotations.append(utils.create_rotation_matrix(math.pi, [0, 0, 1]))
        transformed_rotations = utils.apply_transformation_to_list_of_poses(np.asarray(rotations), transformation)
        for orig_rotations, transformed_rotation in zip(rotations, transformed_rotations):
            self.assertAlmostEqual(np.linalg.det(transformed_rotation[: 3, : 3]), 1)
            np.testing.assert_allclose(transformed_rotation, orig_rotations, equal_nan=True)

        for _ in range(5):
            random_transformation = trimesh.transformations.random_rotation_matrix()
            random_rotations = []
            for _ in range(5):
                random_rotations.append(trimesh.transformations.random_rotation_matrix())

            transformed_rotations = utils.apply_transformation_to_list_of_poses(np.asarray(random_rotations), random_transformation)
            for orig_rotations, transformed_rotation in zip(random_rotations, transformed_rotations):
                self.assertAlmostEqual(np.linalg.det(transformed_rotation[: 3, : 3]), 1)
                np.testing.assert_allclose(random_transformation.dot(orig_rotations), transformed_rotation)

    def test_extract_coplanar_faces_of_mesh(self):
        box = trimesh.creation.box()
        coplanar_surfaces = utils.extract_coplanar_faces_of_mesh(box)
        expected_coplanar_surfaces = np.asarray([[3, 8], [4, 6], [1, 5], [10, 11], [7, 9], [0, 2]])
        for coplanar_surface in coplanar_surfaces:
            self.assertTrue(coplanar_surface in expected_coplanar_surfaces)

    def test_inverse_transformation_matrix(self):
        T = np.eye(4)
        for _ in range(5):
            random_translation = np.random.rand(3)
            random_rotation = trimesh.transformations.random_rotation_matrix()[: 3, : 3]
            T[: 3, : 3] = random_rotation
            T[: 3, 3] = random_translation
            inv_T = utils.invert_transformation_matrix(T)
            expected_inv_T = np.linalg.inv(T)
            np.testing.assert_allclose(inv_T, expected_inv_T, equal_nan=True)

    def test_subtract_2pi_from_all_angles_in_array_over_2pi(self):
        test_array = np.asarray([2*math.pi])
        subtracted_2pi = utils.subtract_2pi_from_all_angles_in_array_over_2pi(test_array)
        expected = np.array([0])
        np.testing.assert_allclose(expected, subtracted_2pi)
        np.testing.assert_array_less(subtracted_2pi, 2*math.pi)

        test_array = np.linspace(0, 2*math.pi, endpoint=False)
        subtracted_2pi = utils.subtract_2pi_from_all_angles_in_array_over_2pi(test_array)
        expected = np.linspace(0, 2*math.pi, endpoint=False)
        np.testing.assert_allclose(subtracted_2pi, expected)
        np.testing.assert_array_less(subtracted_2pi, 2*math.pi)

        test_array = np.linspace(2*math.pi, 4*math.pi, endpoint=False)
        subtracted_2pi = utils.subtract_2pi_from_all_angles_in_array_over_2pi(test_array)
        expected = np.linspace(0, 2*math.pi, endpoint=False)
        np.testing.assert_allclose(subtracted_2pi, expected)
        np.testing.assert_array_less(subtracted_2pi, 2*math.pi)

    def test_map_angles_within_threshold_from_2pi_to_zero(self):
        test_array = np.asarray([2*math.pi])
        mapped_angles = utils.map_angles_within_threshold_from_2pi_to_zero(test_array, 0)
        expected = np.zeros(1)
        np.testing.assert_array_equal(mapped_angles, expected)

        self.assertRaises(AssertionError, utils.map_angles_within_threshold_from_2pi_to_zero, test_array, -1)

        test_array = np.asarray([2*math.pi-1, 2*math.pi, 2*math.pi+1])
        mapped_angles = utils.map_angles_within_threshold_from_2pi_to_zero(test_array, 1)
        expected = np.zeros(3)
        np.testing.assert_array_equal(mapped_angles, expected)

        test_array = np.asarray([1, 2, 3])
        mapped_angles = utils.map_angles_within_threshold_from_2pi_to_zero(test_array, 1)
        expected = np.asarray([1, 2, 3])
        np.testing.assert_array_equal(mapped_angles, expected)


class TestGraphUtils(unittest.TestCase):
    def setUp(self):
        self.random_graph_a = self.create_random_graph()

        centroid = Centroid()
        self.box = trimesh.creation.box()
        self.area_a_box = centroid.extract_area_a(self.box)
        self.adversarial_object = utils.load_mesh("test_objects/adversarial_object.obj")
        self.area_a_adv_obj = centroid.extract_area_a(self.adversarial_object)

    def create_random_graph(self):
        G = nx.complete_graph(20)
        angles, resolution = utils.create_angles_from_resolution(5)
        angles = angles.tolist()
        for node in G.nodes():
            graph_utils.populate_node_of_graph(G, node, "collision_free_angles", copy.deepcopy(angles))
        return G

    def test_remove_edges(self):
        G = nx.complete_graph(5)
        edges_in_graph = len(G.edges())
        for num_removed, e in enumerate(G.edges()):
            self.assertIn(e, G.edges())
            graph_utils.remove_edges(G, [e])
            self.assertEqual(len(G.edges()), edges_in_graph-(num_removed+1))
            self.assertNotIn(e, G.edges())

    def test_populate_node_of_graph(self):
        G = nx.Graph()
        for i in range(10):
            G.add_node(i)
            graph_utils.populate_node_of_graph(G, i, str(i), i*2)
            self.assertEqual(G.nodes[i][str(i)], i*2)

    def test_populate_edge_of_graph(self):
        G = nx.path_graph(10)
        for i in range(9):
            graph_utils.populate_edge_of_graph(G, i, i+1, str(i), i*2)
            self.assertEqual(G[i][i+1][str(i)], i*2)
            self.assertEqual(G[i+1][i][str(i)], i*2)

        G = nx.complete_graph(10)
        for i in range(10):
            for j in range(10):
                if i == j:
                    continue
                graph_utils.populate_edge_of_graph(G, i, j, str(i), i*2)
                self.assertEqual(G[i][j][str(i)], i*2)
                self.assertEqual(G[j][i][str(i)], i*2)

    def test_get_subgraph_of_graph(self):
        G = nx.Graph([(0, 1), (1, 2), (5, 6)])
        subgraphs = graph_utils.get_subgraphs_of_graph(G)
        self.assertEqual(len(subgraphs), 2)
        self.assertListEqual([0, 1, 2], list(subgraphs[0].nodes()))
        self.assertListEqual([5, 6], list(subgraphs[1].nodes()))
        self.assertListEqual([(0, 1), (1, 2)], list(subgraphs[0].edges()))
        self.assertListEqual([(5, 6)], list(subgraphs[1].edges()))

    def test_get_node_data(self):
        G = nx.Graph()
        for i in range(10):
            G.add_node(i)
            graph_utils.populate_node_of_graph(G, i, str(i), i*2)
            self.assertEqual(graph_utils.get_node_data(G, i, str(i)), i*2)

    def test_get_edge_data(self):
        G = nx.path_graph(10)
        for i in range(9):
            graph_utils.populate_edge_of_graph(G, i, i+1, str(i), i*2)
            self.assertEqual(graph_utils.get_edge_data(G, i, i+1, str(i)), i*2)
            self.assertEqual(graph_utils.get_edge_data(G, i+1, i, str(i)), i*2)

        G = nx.complete_graph(10)
        for i in range(10):
            for j in range(10):
                if i == j:
                    continue
                graph_utils.populate_edge_of_graph(G, i, j, str(i), i*2)
                self.assertEqual(graph_utils.get_edge_data(G, i, j, str(i)), i*2)
                self.assertEqual(graph_utils.get_edge_data(G, j, i, str(i)), i*2)

    def test_return_query_nodes_in_graph(self):
        G = nx.Graph()
        for i in range(10):
            G.add_node(i)
        query_nodes = [-3, -2, -1]
        nodes_found, _ = graph_utils.return_query_nodes_in_graph(query_nodes, G)
        self.assertEqual(len(nodes_found), 0)
        query_nodes = [1, 2, 3]
        nodes_found, _ = graph_utils.return_query_nodes_in_graph(query_nodes, G)
        self.assertEqual(len(nodes_found), 3)
        self.assertListEqual(query_nodes, nodes_found)
        query_nodes = [1, 2, 11]
        nodes_found, _ = graph_utils.return_query_nodes_in_graph(query_nodes, G)
        self.assertEqual(len(nodes_found), 2)
        self.assertListEqual([1, 2], nodes_found)

    def test_copy_node_a_to_b(self):
        G = nx.Graph()
        G.add_node(1)
        G.add_node(2)
        graph_utils.populate_node_of_graph(G, 1, "test_copy_node", 42)
        self.assertFalse(G.nodes[1] == G.nodes[2])
        graph_utils.copy_node_a_to_b(G, 1, 2)
        self.assertTrue(G.nodes[1] == G.nodes[2])

    def test_get_all_nodes_connected_to_query_node(self):
        G = nx.Graph([(0, 1), (1, 2), (5, 6)])
        nodes_connected_to_0 = list(graph_utils.get_all_nodes_connected_to_query_node(G, 0))
        self.assertListEqual([0, 1, 2], nodes_connected_to_0)
        nodes_connected_to_2 = list(graph_utils.get_all_nodes_connected_to_query_node(G, 0))
        self.assertListEqual([0, 1, 2], nodes_connected_to_2)

        nodes_connected_to_5 = list(graph_utils.get_all_nodes_connected_to_query_node(G, 5))
        self.assertListEqual([5, 6], nodes_connected_to_5)
        nodes_connected_to_6 = list(graph_utils.get_all_nodes_connected_to_query_node(G, 6))
        self.assertListEqual([5, 6], nodes_connected_to_6)

    def test_graph_exist_in_list_of_graphs(self):
        G = nx.Graph([(0, 1), (1, 2), (5, 6)])
        G2 = nx.Graph([(0, 1),  (5, 6)])
        self.assertTrue(graph_utils.graph_exist_in_list_of_graphs(G, [G]))
        self.assertTrue(graph_utils.graph_exist_in_list_of_graphs(G, [G, G2]))
        self.assertFalse(graph_utils.graph_exist_in_list_of_graphs(G, []))
        self.assertFalse(graph_utils.graph_exist_in_list_of_graphs(G, [G2]))

    def test_find_graphs_that_contain_both_start_and_goal_nodes(self):
        G = nx.Graph([(0, 1), (1, 2), (5, 6)])
        subgraphs = graph_utils.find_graphs_that_contain_both_start_and_goal_nodes(G, [0], [2])
        self.assertEqual(len(subgraphs), 1)
        self.assertListEqual([0, 1, 2], list(subgraphs[0].nodes()))
        self.assertListEqual([(0, 1), (1, 2)], list(subgraphs[0].edges()))

        subgraphs = graph_utils.find_graphs_that_contain_both_start_and_goal_nodes(G, [0], [2, 5])
        self.assertEqual(len(subgraphs), 1)
        self.assertListEqual([0, 1, 2], list(subgraphs[0].nodes()))
        self.assertListEqual([(0, 1), (1, 2)], list(subgraphs[0].edges()))

        subgraphs = graph_utils.find_graphs_that_contain_both_start_and_goal_nodes(G, [0], [5, 6])
        self.assertEqual(len(subgraphs), 0)

        subgraphs = graph_utils.find_graphs_that_contain_both_start_and_goal_nodes(G, [5, 6], [0])
        self.assertEqual(len(subgraphs), 0)

        subgraphs = graph_utils.find_graphs_that_contain_both_start_and_goal_nodes(G, [0, 1], [2])
        self.assertEqual(len(subgraphs), 1)
        self.assertListEqual([0, 1, 2], list(subgraphs[0].nodes()))
        self.assertListEqual([(0, 1), (1, 2)], list(subgraphs[0].edges()))

    def test_node_in_graph(self):
        G = nx.Graph()
        for i in range(10):
            G.add_node(i)
            self.assertTrue(graph_utils.node_in_graph(G, i))
            self.assertFalse(graph_utils.node_in_graph(G, i+1))

    def test_find_nodes_with_attribute_value(self):
        G = nx.Graph()
        for i in range(10):
            G.add_node(i)
            graph_utils.populate_node_of_graph(G, i, str(i), i*2)
            nodes = graph_utils.find_nodes_with_attribute_value(G, str(i), i*2)
            nodes_value_not_in_graph = graph_utils.find_nodes_with_attribute_value(G, str(i), -1+(i*-2))
            nodes_attribute_not_in_graph_2 = graph_utils.find_nodes_with_attribute_value(G, "test", -1+(i*-2))
            self.assertIn(i, nodes)
            self.assertEqual(len(nodes_value_not_in_graph), 0)
            self.assertEqual(len(nodes_attribute_not_in_graph_2), 0)

    def test_generate_new_node(self):
        G = nx.Graph()

        self.assertEqual(len(G.nodes()), 0)
        new_node_index = graph_utils.generate_new_node(G)
        self.assertEqual(new_node_index, 0)
        self.assertEqual(len(G.nodes()), 1)

        new_node_index = graph_utils.generate_new_node(G)
        self.assertEqual(new_node_index, 1)
        self.assertEqual(len(G.nodes()), 2)

        G.remove_node(0)
        new_node_index = graph_utils.generate_new_node(G)
        self.assertEqual(new_node_index, 2)
        self.assertEqual(len(G.nodes()), 2)

    def test_create_reference_frame_to_graph(self):
        G = nx.complete_graph(5)
        for j in range(10):
            random_positions = []
            for i in range(5):
                random_normal_vector = np.random.rand(3)
                random_normal_vector /= np.linalg.norm(random_normal_vector)
                G.nodes[i]["face_normal"] = random_normal_vector

                random_position = np.random.rand(3)
                G.nodes[i]["face_center"] = random_position
            reference_frame = graph_utils.create_reference_frame_to_graph(G)
            average_position = graph_utils.calculate_average_position_of_graph_nodes(G)
            self.assertAlmostEqual(np.linalg.det(reference_frame[: 3, : 3]), 1)
            np.testing.assert_allclose(reference_frame[: 3, 3], average_position, equal_nan=True)
            np.testing.assert_allclose(reference_frame[3, : 3], np.zeros(3), equal_nan=True)

    def test_create_reference_frame_to_node(self):
        G = nx.complete_graph(5)
        for _ in range(10):
            random_positions = []
            for i in range(5):
                random_normal_vector = np.random.rand(3)
                random_normal_vector /= np.linalg.norm(random_normal_vector)
                G.nodes[i]["face_normal"] = random_normal_vector

                random_position = np.random.rand(3)
                G.nodes[i]["face_center"] = random_position
                reference_frame = graph_utils.create_reference_frame_to_node(G, i)
                self.assertAlmostEqual(np.linalg.det(reference_frame[: 3, : 3]), 1)
                np.testing.assert_allclose(reference_frame[3, : 3], np.zeros(3), equal_nan=True)
                np.testing.assert_allclose(reference_frame[: 3, 3], random_position, equal_nan=True)

    def test_calculate_average_normal_vector_of_graph_nodes(self):
        G = nx.complete_graph(5)
        normal_vector = np.array([1, 0, 0])
        for i in range(5):
            G.nodes[i]["face_normal"] = normal_vector
        normal_vec_to_graph = graph_utils.calculate_average_normal_vector_of_graph_nodes(G)
        np.testing.assert_allclose(normal_vec_to_graph, normal_vector, equal_nan=True)

        G = nx.complete_graph(3)
        G.nodes[0]["face_normal"] = np.array([1, 0, 0])
        G.nodes[1]["face_normal"] = np.array([0, 1, 0])
        G.nodes[2]["face_normal"] = np.array([0, 0, 1])

        normal_vec_to_graph = graph_utils.calculate_average_normal_vector_of_graph_nodes(G)

        np.testing.assert_allclose(normal_vec_to_graph, np.array([0.57735027, 0.57735027, 0.57735027]), equal_nan=True)

    def test_calculate_average_position_of_graph_nodes(self):
        G = nx.complete_graph(5)
        for j in range(10):
            random_positions = []
            for i in range(5):
                random_pos = np.random.rand(3)
                G.nodes[i]["face_center"] = random_pos
                random_positions.append(random_pos)
            average_position = graph_utils.calculate_average_position_of_graph_nodes(G)
            np.testing.assert_allclose(average_position, np.asarray(random_positions).mean(0), equal_nan=True)

    def test_find_nodes_with_almost_antiparallel_normals_to_query_node(self):
        query_nodes = list(self.area_a_box.nodes())
        nodes_with_antiparallel_normals, _ = graph_utils.find_nodes_with_almost_antiparallel_normals_to_query_node(
            self.area_a_box, query_nodes, 4)
        expected = [3, 8]
        self.assertListEqual(nodes_with_antiparallel_normals, expected)
        nodes_with_antiparallel_normals, _ = graph_utils.find_nodes_with_almost_antiparallel_normals_to_query_node(
            self.area_a_box, query_nodes, 3)
        expected = [4, 6]
        self.assertListEqual(nodes_with_antiparallel_normals, expected)

        nodes_with_antiparallel_normals, _ = graph_utils.find_nodes_with_almost_antiparallel_normals_to_query_node(self.area_a_adv_obj, [
            80, 21, 83], 45)
        expected = [80, 21]
        self.assertListEqual(nodes_with_antiparallel_normals, expected)

    def test_nodes_that_intersect_another_nodes_normal_vector(self):
        nodes_that_intersect, _ = graph_utils.nodes_that_intersect_another_nodes_normal_vector(self.area_a_box, 1, self.box)
        expected = [7]
        self.assertListEqual(nodes_that_intersect, expected)
        nodes_that_intersect, _ = graph_utils.nodes_that_intersect_another_nodes_normal_vector(self.area_a_box, 4, self.box)
        expected = [3]
        self.assertListEqual(nodes_that_intersect, expected)

        nodes_that_intersect, _ = graph_utils.nodes_that_intersect_another_nodes_normal_vector(
            self.area_a_adv_obj, 45, self.adversarial_object)
        expected = [80, 21, 83]
        self.assertListEqual(nodes_that_intersect, expected)

        nodes_that_intersect, _ = graph_utils.nodes_that_intersect_another_nodes_normal_vector(
            self.area_a_adv_obj, 21, self.adversarial_object)
        # It is a bit odd that node 45 intersects with 21 while 21 intersects with 3. However, the reason is that the normal
        # to node 21 exactly intersects the edge between node 3 and 45. Thus, it seems that trimesh ray intersection function
        # chooses node 3 as the face it collides with
        expected = [3]
        self.assertListEqual(nodes_that_intersect, expected)

    def test_remove_2pi_angle_from_graph_if_0_angle_exist(self):

        for node in self.random_graph_a.nodes():
            collision_free_angles = graph_utils.get_node_data(self.random_graph_a, node, "collision_free_angles")
            self.assertIn(2*math.pi, collision_free_angles)
            self.assertIn(0, collision_free_angles)
        graph_utils.remove_2pi_angle_from_graph_if_0_angle_exist(self.random_graph_a)
        for node in self.random_graph_a.nodes():
            collision_free_angles = graph_utils.get_node_data(self.random_graph_a, node, "collision_free_angles")
            self.assertNotIn(2*math.pi, collision_free_angles)
            self.assertIn(0, collision_free_angles)

        graph_utils.remove_2pi_angle_from_graph_if_0_angle_exist(self.random_graph_a)
        for node in self.random_graph_a.nodes():
            collision_free_angles = graph_utils.get_node_data(self.random_graph_a, node, "collision_free_angles")
            self.assertNotIn(2*math.pi, collision_free_angles)
            self.assertIn(0, collision_free_angles)

    def test_save_graph_to_filename(self):
        self.assertRaises(AssertionError, graph_utils.save_graph_to_filename, None, "test.pickle")
        G = nx.complete_graph(5)
        temp = tempfile.NamedTemporaryFile(suffix='.gpickle')
        graph_utils.save_graph_to_filename(G, temp.name)
        self.assertTrue(os.path.isfile(temp.name))

    def test_load_graph_from_file(self):
        self.assertRaises(AssertionError, graph_utils.load_graph_from_file,  "test.pickle")
        G = nx.complete_graph(5)
        temp = tempfile.NamedTemporaryFile(suffix='.gpickle')
        graph_utils.save_graph_to_filename(G, temp.name)
        loaded_graph = graph_utils.load_graph_from_file(temp.name)
        self.assertTrue(nx.is_isomorphic(G, loaded_graph))


class TestDMG(unittest.TestCase):
    def setUp(self):
        self.gripper = Gripper()

        box = trimesh.creation.box()
        centroid = Centroid()
        self.area_a_box = centroid.extract_area_a(box)
        self.dmg_box = DMG(5, 0.3, box)
        self.random_graph_a = self.create_random_graph()

        adversarial_object = utils.load_mesh("test_objects/adversarial_object.obj")
        utils.scale_object(adversarial_object, self.gripper.finger_mesh.scale)
        self.area_a_adv_obj = centroid.extract_area_a(adversarial_object)
        self.dmg_adv_obj = DMG(5, 0.3, adversarial_object)

        adversarial_object_smaller = utils.load_mesh("test_objects/adversarial_object.obj")
        utils.scale_object(adversarial_object_smaller, 0.8*self.gripper.finger_mesh.scale)
        self.area_a_adv_obj_smaller = centroid.extract_area_a(adversarial_object_smaller)
        self.dmg_smaller_adv_obj = DMG(5, 0.3, adversarial_object_smaller)

        fig6_box = utils.load_mesh("test_objects/fig6_box.obj")
        utils.scale_object(fig6_box, self.gripper.finger_mesh.scale)
        self.area_a_fig6_box = centroid.extract_area_a(fig6_box)
        self.dmg_fig6_box = DMG(5, 0.3, fig6_box)

        self.random_dmg = DMG(5, 0.3, None)

        cylinder = trimesh.creation.cylinder(1.0, height=1)
        self.area_a_cylinder = centroid.extract_area_a(cylinder)

        self.dmg_cylinder = DMG(5, 0.3, cylinder)

        icosahedron = trimesh.creation.icosahedron()
        self.area_a_icosahedron = centroid.extract_area_a(icosahedron)

        self.dmg_icosahedron = DMG(5, 0.3, icosahedron)

    def create_random_graph(self):
        G = nx.complete_graph(20)
        angles, resolution = utils.create_angles_from_resolution(5)
        angles = angles.tolist()
        for node in G.nodes():
            graph_utils.populate_node_of_graph(G, node, "collision_free_angles", copy.deepcopy(angles))
        return G

    def test_set_translation_threshold(self):
        for i in range(100):
            self.dmg_box.set_translation_threshold(i)
            self.assertEqual(self.dmg_box.translation_threshold, i)

        self.assertRaises(AssertionError, self.dmg_box.set_translation_threshold, -1)

    def test_translation_refinement(self):
        self.dmg_box.translation_threshold = 2
        non_refined_graph = copy.deepcopy(self.area_a_box)
        self.dmg_box.translation_refinement(self.area_a_box)
        num_edges_removed = len(nx.difference(non_refined_graph, self.area_a_box).edges())
        self.assertEqual(num_edges_removed, 0)

        # The maximum cosine distance between box faces is 1. So if the translation threshold
        # is set to that, no edges will be removed.
        self.area_a_box = copy.deepcopy(non_refined_graph)
        self.dmg_box.translation_threshold = 1.0
        self.dmg_box.translation_refinement(self.area_a_box)
        edges_removed = nx.difference(non_refined_graph, self.area_a_box).edges()
        self.assertEqual(len(edges_removed), 0)

        edges_that_are_removed = [(0, 4), (5, 6), (6, 11), (4, 7), (5, 10), (0, 1), (2, 7), (9, 11), (8, 10), (8, 9), (1, 3), (2, 3)]
        self.dmg_box.translation_threshold = 0.3
        self.dmg_box.translation_refinement(self.area_a_box)
        edges_removed = list(nx.difference(non_refined_graph, self.area_a_box).edges())
        self.assertEqual(len(edges_removed), 12)
        self.assertEqual(len(edges_removed), len(edges_that_are_removed))
        for edge_that_are_remove in edges_that_are_removed:
            self.assertIn(edge_that_are_remove, edges_removed)

        self.dmg_cylinder.translation_refinement(self.area_a_cylinder)
        num_subgraphs = len(list(nx.connected_components(self.area_a_cylinder)))
        expected_subgraphs = 3
        self.assertEqual(num_subgraphs, expected_subgraphs)

    def test_rotation_refinement(self):
        G = nx.complete_graph(3)
        non_refined_graph = copy.deepcopy(G)
        angles = np.linspace(0, 2*math.pi, 6)
        G.nodes[0]["collision_free_angles"] = angles.tolist()
        G.nodes[1]["collision_free_angles"] = angles.tolist()
        G.nodes[2]["collision_free_angles"] = angles.tolist()
        self.dmg_box.rotation_refinement(G)
        num_edges_removed = len(nx.difference(non_refined_graph, G).edges())
        self.assertEqual(num_edges_removed, 0)

        G = copy.deepcopy(non_refined_graph)
        G.nodes[0]["collision_free_angles"] = angles[: 3]
        G.nodes[1]["collision_free_angles"] = angles[3:]
        G.nodes[2]["collision_free_angles"] = angles[3:]
        self.dmg_box.rotation_refinement(G)
        num_edges_removed = len(nx.difference(non_refined_graph, G).edges())
        self.assertEqual(num_edges_removed, 2)

        G = copy.deepcopy(non_refined_graph)
        G.nodes[0]["collision_free_angles"] = angles[: 2]
        G.nodes[1]["collision_free_angles"] = angles[2: 4]
        G.nodes[2]["collision_free_angles"] = angles[4:]
        self.dmg_box.rotation_refinement(G)
        num_edges_removed = len(nx.difference(non_refined_graph, G).edges())
        self.assertEqual(num_edges_removed, 3)

        G = nx.complete_graph(2)
        non_refined_graph = copy.deepcopy(G)
        angles = np.linspace(0, 2*math.pi, 5)
        angle_split = np.array_split(angles, 2)
        G.nodes[0]["collision_free_angles"] = angle_split[0]
        G.nodes[1]["collision_free_angles"] = angle_split[1]

        G.nodes[0]["reference_frame"] = np.eye(4)
        G.nodes[1]["reference_frame"] = trimesh.transformations.rotation_matrix(2*math.pi, [1, 0, 0])

        self.assertTrue(0 in G.neighbors(1))
        self.dmg_cylinder.rotation_refinement(G, True)
        num_edges_removed = len(nx.difference(non_refined_graph, G).edges())
        self.assertEqual(num_edges_removed, 1)
        self.assertFalse(0 in G.neighbors(1))

        G = copy.deepcopy(non_refined_graph)
        G.nodes[0]["collision_free_angles"] = angle_split[0]
        G.nodes[1]["collision_free_angles"] = angle_split[1]

        G.nodes[0]["reference_frame"] = np.eye(4)
        G.nodes[1]["reference_frame"] = trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0])

        self.assertTrue(0 in G.neighbors(1))
        self.dmg_cylinder.rotation_refinement(G, True)
        num_edges_removed = len(nx.difference(non_refined_graph, G).edges())
        self.assertEqual(num_edges_removed, 0)
        self.assertTrue(0 in G.neighbors(1))

        G = copy.deepcopy(non_refined_graph)
        G.nodes[0]["collision_free_angles"] = angle_split[0]
        G.nodes[1]["collision_free_angles"] = angle_split[1]

        G.nodes[0]["reference_frame"] = np.eye(4)
        G.nodes[1]["reference_frame"] = np.eye(4)

        self.assertTrue(0 in G.neighbors(1))
        self.dmg_cylinder.rotation_refinement(G, True)
        num_edges_removed = len(nx.difference(non_refined_graph, G).edges())
        self.assertEqual(num_edges_removed, 1)
        self.assertFalse(0 in G.neighbors(1))
        G = copy.deepcopy(non_refined_graph)
        G.nodes[0]["collision_free_angles"] = angle_split[0]
        G.nodes[1]["collision_free_angles"] = angle_split[1]

        G.nodes[0]["reference_frame"] = np.eye(4)
        G.nodes[1]["reference_frame"] = trimesh.transformations.rotation_matrix(math.pi/2, [0, 0, 1])

        self.assertTrue(0 in G.neighbors(1))
        self.dmg_cylinder.rotation_refinement(G, True)
        num_edges_removed = len(nx.difference(non_refined_graph, G).edges())
        self.assertEqual(num_edges_removed, 1)
        self.assertFalse(0 in G.neighbors(1))

        G = copy.deepcopy(non_refined_graph)
        G.nodes[0]["collision_free_angles"] = angle_split[0]
        G.nodes[1]["collision_free_angles"] = angle_split[1]

        G.nodes[0]["reference_frame"] = np.eye(4)
        G.nodes[1]["reference_frame"] = trimesh.transformations.rotation_matrix(
            math.pi, [1, 0, 0]).dot(trimesh.transformations.rotation_matrix(math.pi/2, [0, 0, 1]))

        self.assertTrue(0 in G.neighbors(1))
        self.dmg_cylinder.rotation_refinement(G, True)
        num_edges_removed = len(nx.difference(non_refined_graph, G).edges())
        self.assertEqual(num_edges_removed, 0)
        self.assertFalse(1 in G.neighbors(1))

    def test_split_nodes_with_rotation_collisions(self):
        self.dmg_adv_obj.translation_refinement(self.area_a_adv_obj)
        copy_of_area_a_adv_obj = copy.deepcopy(self.area_a_adv_obj)
        nodes_splitted = self.dmg_adv_obj.split_nodes_with_rotation_collisions(self.area_a_adv_obj, self.gripper)
        nodes_splitted_reference_per_node = self.dmg_adv_obj.split_nodes_with_rotation_collisions(
            copy_of_area_a_adv_obj, self.gripper, True)
        expected_nodes_removed = [63, 42, 74, 83, 21, 0, 32, 41]
        for expected_node_removed in expected_nodes_removed:
            self.assertFalse(self.area_a_adv_obj.has_node(expected_node_removed))
            self.assertFalse(copy_of_area_a_adv_obj .has_node(expected_node_removed))
        some_of_the_expected_nodes_kept = [80, 4, 12, 59]
        for expected_node_kept in some_of_the_expected_nodes_kept:
            self.assertTrue(self.area_a_adv_obj.has_node(expected_node_kept))
            self.assertTrue(copy_of_area_a_adv_obj .has_node(expected_node_kept))

        expected_nodes_split = [63, 42, 74, 83]
        for expected_node_split in expected_nodes_split:
            self.assertIn(expected_node_split, nodes_splitted)
            self.assertIn(expected_node_split, nodes_splitted_reference_per_node)

        self.gripper.reset()
        self.dmg_box.translation_refinement(self.area_a_box)
        nodes_splitted = self.dmg_box.split_nodes_with_rotation_collisions(self.area_a_box, self.gripper)
        expected_nodes_splitted = []
        self.assertListEqual(nodes_splitted, expected_nodes_splitted)
        self.dmg_cylinder.translation_refinement(self.area_a_cylinder)
        nodes_splitted = self.dmg_cylinder.split_nodes_with_rotation_collisions(self.area_a_cylinder, self.gripper, True)
        self.assertEqual(len(nodes_splitted), 0)

        self.gripper.reset()
        self.dmg_icosahedron.translation_refinement(self.area_a_icosahedron)
        self.assertRaises(AssertionError, self.dmg_icosahedron.split_nodes_with_rotation_collisions, self.area_a_icosahedron, self.gripper)

    def test_split_node(self):
        G = nx.complete_graph(1)
        test_angles = np.linspace(0, 2*math.pi, 10)
        self.dmg_box.split_node(G, 0, test_angles, np.eye(4))

        self.assertEqual(len(G.nodes()), 2)
        np.testing.assert_equal(G.nodes[1]["collision_free_angles"], test_angles)
        np.testing.assert_allclose(G.nodes[1]["reference_frame"], np.eye(4), equal_nan=True)

        G = nx.complete_graph(1)
        test_angles, angle_resolution = utils.create_angles_from_resolution(9)
        test_angles = test_angles[:: 2]
        set_of_disjoint_angles = utils.find_disjoint_angular_sets(test_angles, angle_resolution)
        random_reference_system = np.random.rand(4, 4)
        self.dmg_box.split_node(G, 0, test_angles, random_reference_system)
        self.assertEqual(len(G.nodes()), 6)
        for i in range(1, 6):
            np.testing.assert_equal(G.nodes[i]["collision_free_angles"], set_of_disjoint_angles[i-1])
            np.testing.assert_allclose(G.nodes[i]["reference_frame"], random_reference_system, equal_nan=True)

    def test_in_hand_planning(self):
        self.dmg_box.build_graph(self.area_a_box, self.gripper)
        goal_node_not_in_start_node = 7
        self.assertRaises(AssertionError, self.dmg_box.in_hand_planning, 1, goal_node_not_in_start_node, 0, 0)
        # path = self.dmg_box.in_hand_planning(0, 0, 0, 0)
        # print(path)

        goal_node_not_connected_to_start_node_on_secondary_finger = 111
        self.dmg_fig6_box.build_graph(self.area_a_fig6_box, self.gripper)

        self.assertRaises(AssertionError, self.dmg_fig6_box .in_hand_planning, 163,
                          goal_node_not_connected_to_start_node_on_secondary_finger, 0, 0)

    def test_construct_gripper_pose_for_node(self):
        dmg = DMG(5, 0.3, None)
        G = nx.complete_graph(2)
        random_position = np.random.rand(3)
        G.nodes[0]["face_center"] = random_position
        G.nodes[0]["reference_frame"] = np.eye(4)
        G.nodes[1]["face_center"] = random_position+2
        G.nodes[1]["secondary_finger_location"] = [random_position+5]
        G.nodes[1]["reference_frame"] = trimesh.transformations.rotation_matrix(math.pi/2, [1, 0, 0])
        G.nodes[1]["collision_free_angles"] = np.asarray([math.radians(135), math.radians(270), math.radians(315)])
        dmg.dmg = G

        correction_angle = utils.compute_angle_around_x_axis_for_aligning_z_axis_of_two_frames(
            G.nodes[0]["reference_frame"], G.nodes[0]["reference_frame"][:3, 2], G.nodes[1]["reference_frame"])
        next_node = 1
        previous_angle = math.pi/4
        primary_finger_pose, secondary_finger_pose, current_gripper_angle = dmg.construct_gripper_pose_for_node(
            next_node, previous_angle, correction_angle)
        expected_orientation_primary_finger = G.nodes[1]["reference_frame"][:3, :3].dot(
            trimesh.transformations.rotation_matrix(math.radians(315), [1, 0, 0])[:3, :3])
        expected_orientation_secondary_finger = expected_orientation_primary_finger.dot(
            trimesh.transformations.rotation_matrix(math.pi, [0, 0, 1])[:3, :3])
        self.assertEqual(current_gripper_angle, math.radians(315))
        np.testing.assert_allclose(primary_finger_pose[:3, 3], random_position+2, equal_nan=True)
        np.testing.assert_allclose(secondary_finger_pose[:3, 3], random_position+5, equal_nan=True)
        np.testing.assert_allclose(primary_finger_pose[:3, :3], expected_orientation_primary_finger, equal_nan=True)
        np.testing.assert_allclose(secondary_finger_pose[:3, :3], expected_orientation_secondary_finger, equal_nan=True)

        G.nodes[1]["collision_free_angles"] = np.asarray([math.radians(135), math.radians(270)])
        primary_finger_pose, secondary_finger_pose, current_gripper_angle = dmg.construct_gripper_pose_for_node(
            next_node, previous_angle, correction_angle)
        self.assertEqual(current_gripper_angle, math.radians(270))

        expected_orientation_primary_finger = G.nodes[1]["reference_frame"][:3, :3].dot(trimesh.transformations.rotation_matrix(math.radians(270), [
            1, 0, 0])[:3, :3])
        expected_orientation_secondary_finger = expected_orientation_primary_finger.dot(
            trimesh.transformations.rotation_matrix(math.pi, [0, 0, 1])[:3, :3])

        np.testing.assert_allclose(primary_finger_pose[:3, :3], expected_orientation_primary_finger, equal_nan=True)
        np.testing.assert_allclose(secondary_finger_pose[:3, :3], expected_orientation_secondary_finger, equal_nan=True)

        G = nx.complete_graph(5)
        for i in range(5):
            G.nodes[i]["face_center"] = random_position
            G.nodes[i]["reference_frame"] = trimesh.transformations.rotation_matrix(math.radians(90*i), [
                1, 0, 0])
            G.nodes[i]["secondary_finger_location"] = [random_position+5]
            G.nodes[i]["collision_free_angles"] = np.asarray([math.radians(135), math.radians(270), math.radians(315)])

        dmg.dmg = G
        previous_angle = math.pi/4
        expected_previous_angles = np.deg2rad([315, 270, 135, 315])
        expected_primary_finger_poses = []

        for expected_angle in [45, 90, 45, -45]:
            transformation_matrix = trimesh.transformations.rotation_matrix(math.radians(expected_angle), [
                1, 0, 0])
            expected_primary_finger_poses.append(transformation_matrix[:3, :3])

        for i in range(1, 5):
            correction_angle = utils.compute_angle_around_x_axis_for_aligning_z_axis_of_two_frames(
                G.nodes[i-1]["reference_frame"], G.nodes[i-1]["reference_frame"][:3, 2], G.nodes[i]["reference_frame"])
            primary_finger_pose, secondary_finger_pose, previous_angle = dmg.construct_gripper_pose_for_node(
                i, previous_angle, correction_angle)
            self.assertAlmostEqual(previous_angle, expected_previous_angles[i-1])
            np.testing.assert_allclose(primary_finger_pose[:3, :3], expected_primary_finger_poses[i-1], equal_nan=True, atol=1e-15)

    def test_construct_path_from_nodes(self):
        self.dmg_box.build_graph(self.area_a_box, self.gripper)
        dmg = self.dmg_box.get_dextrous_manipulation_graph()
        self.dmg_box.setup_secondary_finger_for_in_hand_planning(
            dmg, 0, 0, 0, 0)
        path_primary_finger, path_secondary_finger = self.dmg_box.construct_path_from_nodes([0, 0], 0, 0)

        primary_finger_position_node_0 = graph_utils.get_node_data(dmg, 0, "face_center")
        primary_finger_position_node_5 = graph_utils.get_node_data(dmg, 5, "face_center")
        secondary_finger_position_node_0 = graph_utils.get_node_data(dmg, 0, "secondary_finger_location")[0]
        primary_finger_reference_frame_node_0 = graph_utils.get_node_data(dmg, 0, "reference_frame")
        primary_finger_reference_frame_node_5 = graph_utils.get_node_data(dmg, 5, "reference_frame")
        secondary_finger_reference_frame_node_0 = utils.rotate_coordinate_system_along_axis_with_angle(
            primary_finger_reference_frame_node_0, math.pi, axis=[0, 0, 1])
        np.testing.assert_allclose(path_primary_finger[0], path_primary_finger[-1], equal_nan=True)
        np.testing.assert_allclose(path_secondary_finger[0], path_secondary_finger[-1], equal_nan=True)
        np.testing.assert_allclose(primary_finger_position_node_0, path_primary_finger[0, : 3, 3], equal_nan=True)
        np.testing.assert_allclose(primary_finger_position_node_0, path_primary_finger[-1, : 3, 3], equal_nan=True)
        np.testing.assert_allclose(secondary_finger_position_node_0, path_secondary_finger[0, : 3, 3], equal_nan=True)
        np.testing.assert_allclose(secondary_finger_position_node_0, path_secondary_finger[-1, : 3, 3], equal_nan=True)
        np.testing.assert_allclose(primary_finger_reference_frame_node_0[: 3, : 3], path_primary_finger[0, : 3, : 3], equal_nan=True)
        np.testing.assert_allclose(primary_finger_reference_frame_node_0[: 3, : 3], path_primary_finger[-1, : 3, : 3], equal_nan=True)
        np.testing.assert_allclose(secondary_finger_reference_frame_node_0[: 3, : 3], path_secondary_finger[0, : 3, : 3], equal_nan=True)
        np.testing.assert_allclose(secondary_finger_reference_frame_node_0[: 3, : 3], path_secondary_finger[-1, : 3, : 3], equal_nan=True)
        self.assertFalse(np.allclose(primary_finger_position_node_5, path_primary_finger[0, : 3, 3], equal_nan=True))
        self.assertFalse(np.allclose(primary_finger_reference_frame_node_5[: 3, : 3], path_primary_finger[0, : 3, : 3], equal_nan=True))

        self.dmg_box.setup_secondary_finger_for_in_hand_planning(
            dmg, 1, 5, 0, 0)
        path_primary_finger, path_secondary_finger = self.dmg_box.construct_path_from_nodes([1, 5], 0, 0)
        primary_finger_position_node_1 = graph_utils.get_node_data(dmg, 1, "face_center")
        primary_finger_reference_frame_node_1 = graph_utils.get_node_data(dmg, 1, "reference_frame")
        secondary_finger_position_node_1 = graph_utils.get_node_data(dmg, 1, "secondary_finger_location")[0]
        secondary_finger_position_node_5 = graph_utils.get_node_data(dmg, 5, "secondary_finger_location")[0]
        secondary_finger_reference_frame_node_1 = utils.rotate_coordinate_system_along_axis_with_angle(
            primary_finger_reference_frame_node_1, math.pi, axis=[0, 0, 1])
        secondary_finger_reference_frame_node_5 = utils.rotate_coordinate_system_along_axis_with_angle(
            primary_finger_reference_frame_node_5, math.pi, axis=[0, 0, 1])

        np.testing.assert_allclose(primary_finger_position_node_1, path_primary_finger[0, : 3, 3], equal_nan=True)
        np.testing.assert_allclose(primary_finger_position_node_5, path_primary_finger[-1, : 3, 3], equal_nan=True)
        np.testing.assert_allclose(secondary_finger_position_node_1, path_secondary_finger[0, : 3, 3], equal_nan=True)
        np.testing.assert_allclose(secondary_finger_position_node_5, path_secondary_finger[-1, : 3, 3], equal_nan=True)
        np.testing.assert_allclose(primary_finger_reference_frame_node_1[: 3, : 3], path_primary_finger[0, : 3, : 3], equal_nan=True)
        np.testing.assert_allclose(primary_finger_reference_frame_node_5[: 3, : 3], path_primary_finger[-1, : 3, : 3], equal_nan=True)
        np.testing.assert_allclose(secondary_finger_reference_frame_node_1[: 3, : 3], path_secondary_finger[0, : 3, : 3], equal_nan=True)
        np.testing.assert_allclose(secondary_finger_reference_frame_node_5[: 3, : 3], path_secondary_finger[-1, : 3, : 3], equal_nan=True)

        self.gripper.reset()
        self.dmg_fig6_box.build_graph(self.area_a_fig6_box, self.gripper)
        dmg_fig6_box = self.dmg_fig6_box.get_dextrous_manipulation_graph()
        start_angle = dmg_fig6_box.nodes[216].get("collision_free_angles")[0]
        goal_angle = dmg_fig6_box.nodes[226].get("collision_free_angles")[0]
        self.dmg_fig6_box.setup_secondary_finger_for_in_hand_planning(
            dmg_fig6_box, 216, 226, start_angle, goal_angle)

        nodes_from_start_to_goal, _ = self.dmg_fig6_box.find_path_and_distance_from_start_to_goal_node(dmg_fig6_box, 216, 226)

        path_primary_finger, path_secondary_finger = self.dmg_fig6_box.construct_path_from_nodes(
            nodes_from_start_to_goal, start_angle, goal_angle)

        primary_finger_reference_frame_node_i = graph_utils.get_node_data(dmg_fig6_box, nodes_from_start_to_goal[0], "reference_frame")
        primary_finger_reference_frame_node_i = utils.rotate_coordinate_system_along_axis_with_angle(
            primary_finger_reference_frame_node_i, start_angle, axis=[1, 0, 0])
        secondary_finger_reference_frame_node_i = utils.rotate_coordinate_system_along_axis_with_angle(
            primary_finger_reference_frame_node_i, math.pi, axis=[0, 0, 1])

        np.testing.assert_allclose(primary_finger_reference_frame_node_i[: 3, : 3], path_primary_finger[0, : 3, : 3], equal_nan=True)
        np.testing.assert_allclose(secondary_finger_reference_frame_node_i[: 3, : 3], path_secondary_finger[0, : 3, : 3], equal_nan=True)
        current_angle = start_angle
        for i, node_i in enumerate(nodes_from_start_to_goal):
            primary_finger_position_node_i = graph_utils.get_node_data(dmg_fig6_box, node_i, "face_center")
            secondary_finger_position_node_i = graph_utils.get_node_data(dmg_fig6_box, node_i, "secondary_finger_location")[0]
            np.testing.assert_allclose(primary_finger_position_node_i, path_primary_finger[i, : 3, 3], equal_nan=True)
            np.testing.assert_allclose(secondary_finger_position_node_i, path_secondary_finger[i, : 3, 3], equal_nan=True)

            if i == len(nodes_from_start_to_goal)-1:
                continue
            primary_finger_reference_frame_node_i = graph_utils.get_node_data(dmg_fig6_box, nodes_from_start_to_goal[i], "reference_frame")
            next_collision_free_angles = graph_utils.get_node_data(dmg_fig6_box, node_i, "collision_free_angles")
            if current_angle not in next_collision_free_angles:
                current_angle = utils.find_angle_from_set_closest_to_query_angle(
                    np.asarray(next_collision_free_angles), current_angle)

            primary_finger_reference_frame_node_i = utils.rotate_coordinate_system_along_axis_with_angle(
                primary_finger_reference_frame_node_i, current_angle, axis=[1, 0, 0])
            secondary_finger_reference_frame_node_i = utils.rotate_coordinate_system_along_axis_with_angle(
                primary_finger_reference_frame_node_i, math.pi, axis=[0, 0, 1])

            np.testing.assert_allclose(primary_finger_reference_frame_node_i[: 3, : 3], path_primary_finger[i, : 3, : 3], equal_nan=True)
            np.testing.assert_allclose(
                secondary_finger_reference_frame_node_i[: 3, : 3], path_secondary_finger[i, : 3, : 3], equal_nan=True)

        primary_finger_reference_frame_node_i = graph_utils.get_node_data(dmg_fig6_box, nodes_from_start_to_goal[-1], "reference_frame")
        primary_finger_reference_frame_node_i = utils.rotate_coordinate_system_along_axis_with_angle(
            primary_finger_reference_frame_node_i, goal_angle, axis=[1, 0, 0])
        secondary_finger_reference_frame_node_i = utils.rotate_coordinate_system_along_axis_with_angle(
            primary_finger_reference_frame_node_i, math.pi, axis=[0, 0, 1])

        np.testing.assert_allclose(primary_finger_reference_frame_node_i[: 3, : 3], path_primary_finger[-1, : 3, : 3], equal_nan=True)
        np.testing.assert_allclose(secondary_finger_reference_frame_node_i[: 3, : 3], path_secondary_finger[-1, : 3, : 3], equal_nan=True)

    def test_find_path_and_distance_from_start_to_goal_node(self):
        self.dmg_box.build_graph(self.area_a_box, self.gripper)
        dmg = self.dmg_box.get_dextrous_manipulation_graph()

        nodes_from_start_to_goal, distance = self.dmg_box.find_path_and_distance_from_start_to_goal_node(dmg, 1, 5)
        expected_nodes = [1, 5]
        point_a = dmg.nodes[1].get("face_center")
        point_b = dmg.nodes[5].get("face_center")

        distance_expected = utils.euclidean_distance_between_points(point_a, point_b)
        self.assertListEqual(nodes_from_start_to_goal, expected_nodes)
        self.assertEqual(distance, distance_expected)

        # Test for fig6 object

        self.dmg_fig6_box.build_graph(self.area_a_fig6_box, self.gripper)
        dmg_fig6_box = self.dmg_fig6_box.get_dextrous_manipulation_graph()

        nodes_from_start_to_goal, _ = self.dmg_fig6_box.find_path_and_distance_from_start_to_goal_node(dmg_fig6_box, 165, 53)
        expected_nodes = [165, 57, 166, 58, 162, 54, 163, 55, 164, 56, 160, 52, 161, 53]
        self.assertListEqual(nodes_from_start_to_goal, expected_nodes)

        nodes_from_start_to_goal, _ = self.dmg_fig6_box.find_path_and_distance_from_start_to_goal_node(dmg_fig6_box, 211, 207)
        expected_nodes = [211, 96, 210, 101, 207]
        self.assertListEqual(nodes_from_start_to_goal, expected_nodes)

        dmg_fig6_box = self.dmg_fig6_box.get_dextrous_manipulation_graph()
        start_angle = dmg_fig6_box.nodes[223].get("collision_free_angles")[0]
        goal_angle = dmg_fig6_box.nodes[220].get("collision_free_angles")[0]
        valid_graphs = self.dmg_fig6_box.setup_secondary_finger_for_in_hand_planning(dmg_fig6_box, 165, 53, start_angle, goal_angle)
        self.dmg_fig6_box.current_graph_for_secondary_finger = valid_graphs[0]

        nodes_from_start_to_goal, _ = self.dmg_fig6_box.find_path_and_distance_from_start_to_goal_node(
            dmg_fig6_box, 165, 53)
        expected_nodes = [165, 57, 166, 58, 162, 54, 163, 55, 164, 56, 160, 52, 161, 53]
        self.assertListEqual(nodes_from_start_to_goal, expected_nodes)

    def test_weight_function_for_dmg(self):
        self.dmg_box.build_graph(self.area_a_box, self.gripper)
        distance = self.dmg_box.weight_function_for_dmg(0, 0, 0)
        self.assertEqual(distance, 0)

        distance = self.dmg_box.weight_function_for_dmg(1, 5, 0)
        dmg_graph = self.dmg_box.get_dextrous_manipulation_graph()
        point_a = dmg_graph.nodes[1].get("face_center")
        point_b = dmg_graph.nodes[5].get("face_center")
        distance_expected = utils.euclidean_distance_between_points(point_a, point_b)
        self.assertEqual(distance, distance_expected)

        self.gripper.reset()
        self.dmg_smaller_adv_obj.build_graph(self.area_a_adv_obj_smaller, self.gripper)

        nodes_with_no_valid_secondary_finger_node = [22, 28]
        for node in nodes_with_no_valid_secondary_finger_node:
            distance = self.dmg_smaller_adv_obj.weight_function_for_dmg(67, node, 0)
            self.assertIsNone(distance)

    def test_setup_secondary_finger_for_in_hand_planning(self):
        self.dmg_box.build_graph(self.area_a_box, self.gripper)
        start_node = 1
        goal_node = 5
        start_angle = 0
        goal_angle = 0
        dmg = self.dmg_box.get_dextrous_manipulation_graph()
        valid_graphs_for_secondary_finger = self.dmg_box.setup_secondary_finger_for_in_hand_planning(
            dmg, start_node, goal_node, start_angle, goal_angle)
        connected_components = nx.connected_components(dmg)
        expected_valid_subgraphs = []
        for connected_component in connected_components:
            if 9 in connected_component:
                expected_valid_subgraphs.append(dmg.subgraph(connected_component))

        self.assertEqual(len(valid_graphs_for_secondary_finger), 1)
        for valid_graph in valid_graphs_for_secondary_finger:
            valid_graph_in_expected_graphs = False
            for expected_valid_subgraph in expected_valid_subgraphs:
                if nx.is_isomorphic(valid_graph, expected_valid_subgraph):
                    valid_graph_in_expected_graphs = True
                    break
            self.assertTrue(valid_graph_in_expected_graphs)

        self.gripper.reset()
        self.dmg_adv_obj.build_graph(self.area_a_adv_obj, self.gripper)
        dmg = self.dmg_adv_obj.get_dextrous_manipulation_graph()
        start_node = 3
        goal_node = 54
        start_angle = dmg.nodes[86].get("collision_free_angles")[0]
        goal_angle = dmg.nodes[85].get("collision_free_angles")[0]
        valid_graphs_for_secondary_finger = self.dmg_adv_obj.setup_secondary_finger_for_in_hand_planning(
            dmg, start_node, goal_node, start_angle, goal_angle)

        connected_components = nx.connected_components(dmg)
        expected_valid_subgraphs = []
        for connected_component in connected_components:
            if 86 in connected_component or 72 in connected_component:
                expected_valid_subgraphs.append(dmg.subgraph(connected_component))

        self.assertEqual(len(valid_graphs_for_secondary_finger), 2)
        for valid_graph in valid_graphs_for_secondary_finger:
            valid_graph_in_expected_graphs = False
            for expected_valid_subgraph in expected_valid_subgraphs:
                if nx.is_isomorphic(valid_graph, expected_valid_subgraph):
                    valid_graph_in_expected_graphs = True
                    break
            self.assertTrue(valid_graph_in_expected_graphs)
        self.gripper.reset()
        self.dmg_fig6_box.build_graph(self.area_a_fig6_box, self.gripper)
        dmg_fig6_box = self.dmg_fig6_box.get_dextrous_manipulation_graph()
        start_node = 165
        goal_node = 122
        # 223 is the node that the normal to 165 intersects. We use its collision free angle
        # as that node is has some collisions whereas node 165 has no collisions
        start_angle = dmg_fig6_box .nodes[223].get("collision_free_angles")[0]
        goal_angle = dmg_fig6_box .nodes[122].get("collision_free_angles")[0]
        valid_graphs = self.dmg_fig6_box.setup_secondary_finger_for_in_hand_planning(
            dmg_fig6_box, start_node, goal_node, start_angle, goal_angle)
        self.assertListEqual(valid_graphs, [])

    def test_find_valid_nodes_for_secondary_finger(self):
        self.dmg_box.build_graph(self.area_a_box, self.gripper)
        nodes = self.area_a_box.nodes()
        expected_secondary_finger_nodes = [10, 7, 10, 4, 3, 1, 7, 4, 3, 0, 1, 0]
        for i, node in enumerate(nodes):
            secondary_finger_nodes = self.dmg_box.find_valid_nodes_for_secondary_finger(self.area_a_box, node)
            self.assertIn(expected_secondary_finger_nodes[i], secondary_finger_nodes)

        self.gripper.reset()
        self.dmg_adv_obj.build_graph(self.area_a_adv_obj, self.gripper)
        nodes_to_test_on_adversarial_object = [45, 54]
        expected_secondary_finger_nodes = [[80, 86], [82, 85]]
        for i, node in enumerate(nodes_to_test_on_adversarial_object):
            secondary_finger_nodes = self.dmg_adv_obj.find_valid_nodes_for_secondary_finger(self.area_a_adv_obj, node)
            self.assertListEqual(expected_secondary_finger_nodes[i], secondary_finger_nodes)

    def test_nodes_with_overlapping_collision_free_angles(self):
        self.dmg_box.translation_refinement(self.area_a_box)
        self.dmg_box.split_nodes_with_rotation_collisions(self.area_a_box, self.gripper)
        self.dmg_box.rotation_refinement(self.area_a_box)

        query_nodes = list(self.area_a_box.nodes())
        for query_node in query_nodes:
            nodes_with_antiparallel_normals, _ = graph_utils.find_nodes_with_almost_antiparallel_normals_to_query_node(
                self.area_a_box, query_nodes, query_node)
            nodes_with_joint_collision_free_angles, _ = self.dmg_box.nodes_with_overlapping_collision_free_angles(
                self.area_a_box, nodes_with_antiparallel_normals, query_node)
            self.assertListEqual(nodes_with_antiparallel_normals, nodes_with_joint_collision_free_angles)

        G = nx.complete_graph(2)
        angles = np.linspace(0, 2*math.pi, 6, endpoint=False)
        G.nodes[0]["collision_free_angles"] = angles[: 3]
        G.nodes[1]["collision_free_angles"] = angles[3:]
        G.nodes[0]["reference_frame"] = np.eye(4)
        G.nodes[1]["reference_frame"] = np.eye(4)
        expected_nodes_with_joint_collision_free_angles = []
        nodes_with_joint_collision_free_angles, _ = self.random_dmg.nodes_with_overlapping_collision_free_angles(
            G, [0], 1)
        self.assertListEqual(expected_nodes_with_joint_collision_free_angles, nodes_with_joint_collision_free_angles)

        G = nx.complete_graph(3)
        G.nodes[0]["collision_free_angles"] = angles[: 3]
        G.nodes[1]["collision_free_angles"] = angles[3:]
        G.nodes[2]["collision_free_angles"] = angles[3:]
        G.nodes[0]["reference_frame"] = np.eye(4)
        G.nodes[1]["reference_frame"] = np.eye(4)
        pi_rotation_around_x = trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0])
        G.nodes[2]["reference_frame"] = pi_rotation_around_x
        nodes_with_joint_collision_free_angles, _ = self.random_dmg.nodes_with_overlapping_collision_free_angles(
            G, [1, 2], 0)
        expected_nodes_with_joint_collision_free_angles = [2]
        self.assertListEqual(expected_nodes_with_joint_collision_free_angles, nodes_with_joint_collision_free_angles)

    def test_get_dextrous_manipulation_graph(self):
        A = self.dmg_box.get_dextrous_manipulation_graph()
        self.assertIsNone(A)

    def test_save_dmg_to_file(self):
        pass

    def load_dmg_from_file(self):
        pass


if __name__ == '__main__':
    unittest.main()

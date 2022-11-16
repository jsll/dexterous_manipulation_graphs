import glob
import trimesh
import numpy as np
import math
from pathlib import Path


def get_files_in_folder(folder, specific_object="", suffix="*"):
    if specific_object != "":
        files = glob.glob(folder+specific_object+suffix)
    else:
        files = glob.glob(folder+"*"+suffix)
    return files


def load_mesh(mesh_file):
    mesh = trimesh.load(mesh_file)
    return mesh


def create_pointcloud_of_mesh(mesh, num_points):
    points_sampled, _ = trimesh.sample.sample_surface_even(mesh, num_points)
    return points_sampled


def cosine_distance(vec_a, vec_b):
    assert (np.round(np.linalg.norm(vec_a), 6) == np.round(np.linalg.norm(vec_b), 6) == 1), "Input vectors are not normalized"
    cosine_similarity = 1-vec_a.dot(vec_b)
    return cosine_similarity


def intersection_of_lists(list_a, list_b):
    union = list(set(list_a) & set(list_b))
    return union


def union_of_lists(lst1, lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list


def generate_random_normal_vec_to_x(vec_x):
    assert (np.round(np.linalg.norm(vec_x), 6) == 1)
    normal_vec = np.random.randn(3)
    normal_vec -= normal_vec.dot(vec_x) * vec_x
    normal_vec /= np.linalg.norm(normal_vec)
    return normal_vec


def scale_object(object_mesh, new_scale):
    bounding_box_of_object = object_mesh.bounding_box
    shortest_side = min(bounding_box_of_object.extents)
    mesh_scaling_factor = shortest_side/new_scale
    object_mesh.apply_scale(1.0/mesh_scaling_factor)


def create_translation_matrix(translation_vec):
    transformation_matrix = trimesh.transformations.translation_matrix(translation_vec)
    return transformation_matrix


def create_rotation_matrix(angle, axis, point=None):
    rot_matrix = trimesh.transformations.rotation_matrix(angle, axis, point)
    return rot_matrix


def points_to_mesh(points):
    if points.size == 0:
        return None
    if len(points.shape) < 2:
        points = np.expand_dims(points, 0)
    point_as_pointcloud = trimesh.points.PointCloud(points)
    return point_as_pointcloud


def create_transformation_matrix_from_rot_and_trans(rotation=np.eye(3), translation=np.zeros(3)):
    coordinate_system = np.eye(4)
    coordinate_system[:3, :3] = rotation
    coordinate_system[:3, 3] = translation
    return coordinate_system


def rotate_axis_of_coordinate_system_to_align_with_vector(
        coordinate_system, vector, axis=0):
    assert (axis == 0 or axis == 1 or axis == 2), "Axis needs to be 1, 0 or 2"
    assert (np.round(np.linalg.norm(vector), 6) == 1), "Input vectors are not normalized"
    coordinate_system_axis = coordinate_system[:3, axis]
    R = rotation_matrix_to_align_vec_a_to_b(coordinate_system_axis, vector)
    T = create_transformation_matrix_from_rot_and_trans(R)
    coordinate_system = T.dot(coordinate_system)
    return coordinate_system


def rotation_matrix_to_align_vec_a_to_b(vec_a, vec_b):
    assert (np.round(np.linalg.norm(vec_a), 6) == np.round(np.linalg.norm(vec_b), 6) == 1), "Input vectors are not normalized"
    v = np.cross(vec_a, vec_b)
    c = np.dot(vec_a, vec_b)

    skew_v = skew(v)
    R = np.eye(3)+skew_v+np.matmul(skew_v, skew_v)*(1/(1+c))
    return R


def skew(vector):
    """
    this function returns a numpy array with the skew symmetric cross product matrix for vector.
    the skew symmetric cross product matrix is defined such that
    np.cross(a, b) = np.dot(skew(a), b)

    :param vector: An array like vector to create the skew symmetric cross product matrix for
    :return: A numpy array of the skew symmetric cross product vector
    """

    return np.array([[0, -vector.item(2), vector.item(1)],
                     [vector.item(2), 0, -vector.item(0)],
                     [-vector.item(1), vector.item(0), 0]])


def rotate_coordinate_system_along_axis_with_angle(reference_frame, angle, axis):
    rot_matrix_around_axis = create_rotation_matrix(angle, axis)
    rotated_reference_frame = reference_frame.dot(rot_matrix_around_axis)
    return rotated_reference_frame


def map_2pi_to_zero_in_list(list_of_angles):
    index_of_2_pi_in_list = np.where(list_of_angles == 2*math.pi)[0]
    index_of_0_in_list = np.where(list_of_angles == 0)[0]

    # By construction, there can only be one 0
    if len(index_of_0_in_list) > 0 and len(index_of_2_pi_in_list) > 0:
        list_of_angles = np.delete(list_of_angles, index_of_2_pi_in_list)
    elif len(index_of_0_in_list) == 0 and len(index_of_2_pi_in_list) == 1:
        list_of_angles[index_of_2_pi_in_list] = 0
        list_of_angles.sort()
    return list_of_angles


def query_angle_within_threshold_of_angle_array(query_angle, angle_array,  threshold):
    angle_array_lower_bound = angle_array-threshold
    angle_array_upper_bound = angle_array+threshold
    return np.any((query_angle >= angle_array_lower_bound) & (query_angle <= angle_array_upper_bound))


def subtract_2pi_from_all_angles_in_array_over_2pi(query_array):
    query_array[(query_array) >= 2*math.pi] -= 2*math.pi
    return query_array


def map_angles_within_threshold_from_2pi_to_zero(angle_array, threshold):
    assert threshold >= 0, "Threshold needs to be nonnegative"
    angle_array[(angle_array+threshold) >= 2*math.pi] = 0
    return angle_array


def find_overlapping_angles_within_threshold(angle_list_a, angle_list_b, threshold_for_overlap):
    angle_list_b_lower_bound = angle_list_b-threshold_for_overlap
    angle_list_b_upper_bound = angle_list_b+threshold_for_overlap
    overlapping_angles = []
    for angle in angle_list_a:
        if np.any((angle >= angle_list_b_lower_bound) & (angle <= angle_list_b_upper_bound)):
            overlapping_angles.append(angle)
    overlapping_angles = np.asarray(overlapping_angles)
    return overlapping_angles


def find_angle_sign_for_aligning_frames_around_axis(frame_a, frame_b, angle, axis=[1, 0, 0]):
    rot_matrix = create_rotation_matrix(angle, axis)[:3, :3]
    rotated_frame_a = frame_a.dot(rot_matrix.T)

    if np.allclose(rotated_frame_a, frame_b):
        return 1
    else:
        return -1


def object_ray_intersections(mesh, ray_origin, ray_direction):
    triangles_hit, _, location_for_intersection = mesh.ray.intersects_id(ray_origin, ray_direction, return_locations=True)
    index_for_all_non_self_intersections = ~np.all(location_for_intersection == ray_origin, axis=1)
    location_for_intersection = location_for_intersection[index_for_all_non_self_intersections]
    triangles_hit = triangles_hit[index_for_all_non_self_intersections]
    return triangles_hit, location_for_intersection


def find_angle_from_set_closest_to_query_angle(angle_set, query_angle):
    assert query_angle >= 0, "Query angle needs to be non-negative"
    angle_differences = np.abs(angle_set-query_angle)
    angle_difference = math.pi - np.abs(angle_differences - math.pi)
    closest_angle = angle_set[angle_difference .argmin()]
    return closest_angle


def find_disjoint_angular_sets(admissable_angles, angle_resolution):
    k = 0
    set_of_disjoint_angle_sets = []
    set_of_consecutive_angles = []
    while k < len(admissable_angles)-1:
        phi_k = admissable_angles[k]
        set_of_consecutive_angles.append(phi_k)
        phi_k_next = admissable_angles[k+1]
        if (phi_k_next-phi_k) > (angle_resolution+1e-4):
            set_of_disjoint_angle_sets.append(set_of_consecutive_angles)
            set_of_consecutive_angles = []
        k += 1

    last_admissible_angle = admissable_angles[-1]
    set_of_consecutive_angles.append(last_admissible_angle)
    set_of_disjoint_angle_sets.append(set_of_consecutive_angles)
    if len(set_of_disjoint_angle_sets) > 1:
        join_sets_containing_zero_and_two_pi(set_of_disjoint_angle_sets)

    return set_of_disjoint_angle_sets


def join_sets_containing_zero_and_two_pi(set_of_disjoint_angle_sets):
    assert (isinstance(set_of_disjoint_angle_sets[0], list)), "The list set_of_disjoint_angle_sets needs to be a list of lists"
    assert (len(set_of_disjoint_angle_sets) > 1), "Need more than one set of disjoint angles to join"
    first_set = set_of_disjoint_angle_sets[0]
    last_set = set_of_disjoint_angle_sets[-1]
    if 0 in first_set and 2*math.pi in last_set:
        joined_set = first_set+last_set
        del set_of_disjoint_angle_sets[0]
        del set_of_disjoint_angle_sets[-1]
        set_of_disjoint_angle_sets.insert(0, joined_set)


def create_angles_from_resolution(resolution):
    assert (resolution > 0), "Angle resolution needs to be larger than zero"
    assert (isinstance(resolution, int)), "Angle resolution needs to be larger than zero"
    angle_discretization = np.linspace(0, 2*math.pi, resolution+1)
    angle_resolution = angle_discretization[1]
    return angle_discretization, angle_resolution


def compute_angle_around_x_axis_for_aligning_z_axis_of_two_frames(world_to_frame_a, z_axis_of_frame_a, world_to_frame_b):
    assert (np.allclose(world_to_frame_a[:3, 0], world_to_frame_b[:3, 0]) or np.allclose(world_to_frame_a[:3, 0], -world_to_frame_b[:3, 0])
            ), "This function assumes that the y-z axes of the two frames lie in the same plane"
    z_axis_of_frame_b = world_to_frame_b[:3, 2]
    cos_theta = z_axis_of_frame_a.dot(z_axis_of_frame_b)
    cos_theta = np.clip(cos_theta, -1, 1)
    angle_between_z_axises = np.arccos(cos_theta)
    if np.round(angle_between_z_axises, 4) == 0:
        return 0
    angle_sign_around_x_axis_for_aligning_z_axis = find_angle_sign_for_aligning_frames_around_axis(
        world_to_frame_b[:3, :3], world_to_frame_a[:3, :3], angle_between_z_axises, axis=[1, 0, 0])
    if angle_sign_around_x_axis_for_aligning_z_axis == -1:
        angle_between_z_axises = 2*math.pi-angle_between_z_axises
    return angle_between_z_axises


def vectors_spanning_plane_to_normal(normal_vector):
    assert (np.round(np.linalg.norm(normal_vector), 6) == 1), "Input vectors are not normalized"
    y_axis = generate_random_normal_vec_to_x(normal_vector)
    z_axis = np.cross(normal_vector, y_axis)
    assert np.round(np.linalg.norm(normal_vector), 3) == np.round(np.linalg.norm(y_axis), 3) == np.round(np.linalg.norm(z_axis), 3) == 1
    assert np.round(np.dot(normal_vector, y_axis), 5) == np.round(
        np.dot(normal_vector, z_axis), 5) == np.round(np.dot(y_axis, z_axis), 5) == 0

    return y_axis, z_axis


def euclidean_distance_between_points(point_a, point_b):
    distance_between_nodes = np.linalg.norm(point_a-point_b)
    return distance_between_nodes


def query_angle_in_list_of_angles(list_of_angles, query_angle, threshold_for_overlap):
    angle_list_b_lower_bound = list_of_angles-threshold_for_overlap
    angle_list_b_upper_bound = list_of_angles+threshold_for_overlap
    if np.any((query_angle >= angle_list_b_lower_bound) & (query_angle <= angle_list_b_upper_bound)):
        return True
    else:
        return False


def apply_transformation_to_list_of_poses(list_of_poses, transformation):
    transformed_poses = transformation@list_of_poses
    return transformed_poses


def extract_coplanar_faces_of_mesh(trimesh):
    indices_of_faces_that_are_almost_coplanar = trimesh.face_adjacency_angles < 1e-3
    coplanar_faces = trimesh.face_adjacency[indices_of_faces_that_are_almost_coplanar]
    return coplanar_faces


def invert_transformation_matrix(T):
    R = T[:3, :3]
    p = T[:3, 3]
    inv_T = np.eye(4)
    inv_T[:3, :3] = R.T
    inv_T[0, 3] = -R[:3, 0].dot(p)
    inv_T[1, 3] = -R[:3, 1].dot(p)
    inv_T[2, 3] = -R[:3, 2].dot(p)
    return inv_T


def get_filename_without_extension(filename):
    filename = Path(filename).stem
    return filename

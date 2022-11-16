import numpy as np
import networkx as nx
from utils import utils
from utils import graph_utils
import copy
import math
import visualize
import matplotlib.pyplot as plt


class DMG:
    def __init__(self, angle_discretization_steps, translation_threshold, mesh, finger_allowed_inside_object=False):
        self.mesh = mesh
        self.dmg = None
        self.current_graph_for_secondary_finger = None
        self.correction_angles_graph = nx.DiGraph()
        self.set_angle_discretization(angle_discretization_steps)
        self.set_finger_allowed_inside_object(finger_allowed_inside_object)
        self.set_translation_threshold(translation_threshold)

    def set_translation_threshold(self, translation_threshold):
        assert (translation_threshold >= 0), "Translation threshold needs to be nonnegative"
        self.translation_threshold = translation_threshold

    def get_translation_threshold(self):
        return self.translation_threshold

    def set_angle_discretization(self, number_of_angle_bins):
        self.angle_discretization, self.angle_resolution = utils.create_angles_from_resolution(number_of_angle_bins)
        self.angle_discretization_size = len(self.angle_discretization)

    def get_angle_resolution(self):
        return self.angle_resolution

    def get_angle_discretization(self):
        return self.angle_discretization

    def get_dextrous_manipulation_graph(self):
        return self.dmg

    def set_finger_allowed_inside_object(self, allow_finger_inside_object):
        self.finger_allowed_inside_object = allow_finger_inside_object

    def get_finger_allowed_inside_object(self):
        return self.finger_allowed_inside_object

    def set_mesh(self, mesh):
        self.mesh = mesh

    def get_mesh(self):
        return self.mesh

    def build_graph(self, graph_of_area_a, gripper, reference_frame_per_node=False):
        self.translation_refinement(graph_of_area_a)

        self.split_nodes_with_rotation_collisions(graph_of_area_a, gripper, reference_frame_per_node)
        self.rotation_refinement(graph_of_area_a, reference_frame_per_node)
        graph_utils.remove_2pi_angle_from_graph_if_0_angle_exist(graph_of_area_a)
        self.dmg = graph_of_area_a

    def translation_refinement(self, graph_of_area_a):

        edges_to_remove = []
        for node_x in graph_of_area_a.nodes():
            node_x_normal = graph_of_area_a.nodes[node_x]['face_normal']
            nodes_connected_to_x = graph_of_area_a.neighbors(node_x)
            for node_y in nodes_connected_to_x:
                node_y_normal = graph_of_area_a.nodes[node_y]['face_normal']
                cosine_distance = utils.cosine_distance(node_x_normal, node_y_normal)
                if cosine_distance > self.translation_threshold:
                    edges_to_remove.append((node_x, node_y))
            graph_utils.remove_edges(graph_of_area_a, edges_to_remove)

    def rotation_refinement(self, graph_of_area_a, reference_frame_per_node=False):

        for node_x in graph_of_area_a.nodes():
            edges_to_remove = []
            node_x_collision_free_grasp_angles = graph_utils.get_node_data(graph_of_area_a, node_x, "collision_free_angles")
            nodes_connected_to_x = graph_of_area_a.neighbors(node_x)
            if reference_frame_per_node:
                node_x_reference_frame = graph_utils.get_node_data(graph_of_area_a, node_x, "reference_frame")
            for node_y in nodes_connected_to_x:
                node_y_collision_free_grasp_angles = graph_utils.get_node_data(graph_of_area_a, node_y, "collision_free_angles")
                if reference_frame_per_node:
                    node_y_reference_frame = graph_utils.get_node_data(graph_of_area_a, node_y, "reference_frame")

                    x_axis_of_node_y_frame_rotated_on_node_x_frame = utils.rotate_axis_of_coordinate_system_to_align_with_vector(
                        node_y_reference_frame, node_x_reference_frame[:3, 0], axis=0)
                    correction_angle = utils.compute_angle_around_x_axis_for_aligning_z_axis_of_two_frames(
                        node_x_reference_frame, node_x_reference_frame[:3, 2], x_axis_of_node_y_frame_rotated_on_node_x_frame)

                    collision_free_angles_in_comparing_node_frame = node_y_collision_free_grasp_angles + correction_angle
                    angles_over_2_pi = collision_free_angles_in_comparing_node_frame > 2*math.pi
                    collision_free_angles_in_comparing_node_frame[angles_over_2_pi] -= 2*math.pi
                    angles_that_overlap = utils.find_overlapping_angles_within_threshold(
                        collision_free_angles_in_comparing_node_frame, node_x_collision_free_grasp_angles, self.angle_resolution/2.0)
                else:
                    correction_angle = 0
                    angles_that_overlap = utils.intersection_of_lists(
                        node_x_collision_free_grasp_angles, node_y_collision_free_grasp_angles)
                if len(angles_that_overlap) == 0:
                    edges_to_remove.append((node_x, node_y))
                else:
                    self.correction_angles_graph.add_edge(node_x, node_y)
                    graph_utils.populate_edge_of_graph(self.correction_angles_graph, node_x, node_y,
                                                       "correction_angle", correction_angle)

            graph_utils.remove_edges(graph_of_area_a, edges_to_remove)

    def split_nodes_with_rotation_collisions(self, graph_of_area_a, gripper, reference_frame_per_node=False):
        subgraphs = graph_utils.get_subgraphs_of_graph(graph_of_area_a)
        nodes_to_remove = []
        nodes_split = []
        for subgraph in subgraphs:
            nodes_to_potentially_split = list(subgraph.nodes())
            if not reference_frame_per_node:
                reference_frame = graph_utils.create_reference_frame_to_graph(subgraph)
            for node in nodes_to_potentially_split:
                if reference_frame_per_node:
                    reference_frame = graph_utils.create_reference_frame_to_node(subgraph, node)
                finger_position = subgraph.nodes[node]["face_center"]
                collision_free_angles_at_node = gripper.find_collision_free_gripper_rotations(
                    self.mesh, finger_position, reference_frame, self.angle_discretization)
                num_collision_free_angles = len(collision_free_angles_at_node)
                if num_collision_free_angles == 0:
                    nodes_to_remove.append(node)
                elif num_collision_free_angles == self.angle_discretization_size:
                    graph_utils.populate_node_of_graph(graph_of_area_a, node, "collision_free_angles", collision_free_angles_at_node)
                    graph_utils.populate_node_of_graph(graph_of_area_a, node, "reference_frame", reference_frame)
                else:
                    self.split_node(graph_of_area_a, node, collision_free_angles_at_node, reference_frame)
                    nodes_split.append(node)
                    nodes_to_remove.append(node)

        graph_of_area_a.remove_nodes_from(nodes_to_remove)
        return nodes_split

    def split_node(self, graph_of_area_a, node, admissible_angles, reference_frame):
        set_of_disjoint_angle_sets = utils.find_disjoint_angular_sets(admissible_angles, self.angle_resolution)

        for angle_set in set_of_disjoint_angle_sets:
            added_node_index = graph_utils.generate_new_node(graph_of_area_a)
            graph_utils.copy_node_a_to_b(graph_of_area_a, node, added_node_index)
            graph_utils.populate_node_of_graph(graph_of_area_a, added_node_index, "collision_free_angles", np.asarray(angle_set))
            graph_utils.populate_node_of_graph(graph_of_area_a, added_node_index, "reference_frame", reference_frame)

    def in_hand_planning(self, start_node, goal_node, start_angle, goal_angle):
        temp_graph = copy.deepcopy(self.dmg)
        C1 = graph_utils.get_all_nodes_connected_to_query_node(self.dmg, start_node)
        assert (goal_node in C1), "Goal node is not connected to start node"

        valid_graphs_for_secondary_finger = self.setup_secondary_finger_for_in_hand_planning(
            self.dmg, start_node, goal_node, start_angle, goal_angle)
        assert (len(valid_graphs_for_secondary_finger) >
                0), "No valid graphs for secondary finger found. This could mean that the start and goal node for the secondary finger are not connected"

        # Paths is a list of size-three path tuples where the first element is the path for the primary finger, the secondary element is the
        # path for the secondary finger and the third element is the distance of the path.
        # Each path tuple represents a different gripper path from the start to the goal node.
        paths = []
        for valid_graph_for_secondary_finger in valid_graphs_for_secondary_finger:
            self.current_graph_for_secondary_finger = valid_graph_for_secondary_finger
            try:
                nodes_from_start_to_goal, distance = self.find_path_and_distance_from_start_to_goal_node(
                    self.dmg, start_node, goal_node)
            except nx.exception.NetworkXNoPath:
                self.dmg = temp_graph
                continue
            path = self.construct_path_from_nodes(nodes_from_start_to_goal, start_angle, goal_angle)
            path += (distance,)
            paths.append(path)
            # secondary_finger_graph_to_path[valid_graph_for_secondary_finger] = path
            self.dmg = temp_graph
        return paths

    def find_path_and_distance_from_start_to_goal_node(self, graph, start_node, goal_node):
        distance, nodes_from_start_to_goal = nx.bidirectional_dijkstra(graph, start_node,
                                                                       target=goal_node, weight=self.weight_function_for_dmg)
        return nodes_from_start_to_goal, distance

    def weight_function_for_dmg(self, node_a, node_b, edge):
        valid_nodes_for_secondary_finger = self.find_valid_nodes_for_secondary_finger(
            self.dmg, node_b, secondary_finger_graph=self.current_graph_for_secondary_finger)
        if len(valid_nodes_for_secondary_finger) == 0:
            return None
        else:
            center_of_point_a = self.dmg.nodes[node_a].get("face_center")
            center_of_point_b = self.dmg.nodes[node_b].get("face_center")
            distance_between_nodes = utils.euclidean_distance_between_points(center_of_point_a, center_of_point_b)
            return distance_between_nodes

    def setup_secondary_finger_for_in_hand_planning(self, graph, start_node, goal_node, start_angle, goal_angle):
        secondary_finger_start_nodes = self.find_valid_nodes_for_secondary_finger(graph, start_node, start_angle)
        secondary_finger_goal_nodes = self.find_valid_nodes_for_secondary_finger(graph, goal_node, goal_angle)
        valid_graphs = graph_utils.find_graphs_that_contain_both_start_and_goal_nodes(
            graph, secondary_finger_start_nodes, secondary_finger_goal_nodes)
        return valid_graphs

    def find_valid_nodes_for_secondary_finger(self, graph, node_on_principal_finger, angle_on_principal_finger=None, secondary_finger_graph=None):
        nodes_intersecting_with_principal_finger, locations_for_intersections = graph_utils.nodes_that_intersect_another_nodes_normal_vector(
            graph, node_on_principal_finger, self.mesh)
        secondary_finger_nodes_antiparallel_to_principal_finger, node_indices_kept_antiparallel = graph_utils.find_nodes_with_almost_antiparallel_normals_to_query_node(
            graph, nodes_intersecting_with_principal_finger, node_on_principal_finger, self.get_translation_threshold())
        secondary_finger_valid_nodes, node_indices_kept_overlapping_angles = self.nodes_with_overlapping_collision_free_angles(
            graph, secondary_finger_nodes_antiparallel_to_principal_finger, node_on_principal_finger, angle_on_principal_finger)
        locations_for_intersections = locations_for_intersections[node_indices_kept_antiparallel,
                                                                  :][node_indices_kept_overlapping_angles, :]
        if secondary_finger_graph is not None:
            secondary_finger_valid_nodes, node_indices_kept_in_graph = graph_utils.return_query_nodes_in_graph(
                secondary_finger_valid_nodes, secondary_finger_graph)
            locations_for_intersections = locations_for_intersections[node_indices_kept_in_graph, :]
        assert (len(secondary_finger_valid_nodes) ==
                locations_for_intersections.shape[0]), "There must be as many intersections as there are valid nodes"
        graph_utils.populate_node_of_graph(graph, node_on_principal_finger, "valid_secondary_finger_nodes", locations_for_intersections)
        graph_utils.populate_node_of_graph(graph, node_on_principal_finger, "secondary_finger_location", locations_for_intersections)
        return secondary_finger_valid_nodes

    def nodes_with_overlapping_collision_free_angles(self, graph, list_of_nodes, node_to_compare_against, query_angle=None):
        angles_to_compare_against = list(graph.nodes[node_to_compare_against].get("collision_free_angles"))
        if (0 in angles_to_compare_against) and (not 2*math.pi in angles_to_compare_against):
            angles_to_compare_against.append(2*math.pi)
        elif (0 not in angles_to_compare_against) and (2*math.pi in angles_to_compare_against):
            angles_to_compare_against.append(0)

        reference_system_of_comparing_node = graph.nodes[node_to_compare_against].get("reference_frame")
        z_axis_of_comparing_node = reference_system_of_comparing_node[:3, 2]
        nodes_with_overlapping_angles = []
        indices_of_nodes_kept = []
        for index, query_node in enumerate(list_of_nodes):
            reference_frame_of_query_node = graph.nodes[query_node].get("reference_frame")
            correction_angle = utils.compute_angle_around_x_axis_for_aligning_z_axis_of_two_frames(
                reference_system_of_comparing_node, z_axis_of_comparing_node, reference_frame_of_query_node)
            collision_free_angles_query_node = graph.nodes[query_node].get("collision_free_angles")
            collision_free_angles_in_comparing_node_frame = collision_free_angles_query_node + correction_angle
            angles_over_2_pi = collision_free_angles_in_comparing_node_frame > 2*math.pi
            collision_free_angles_in_comparing_node_frame[angles_over_2_pi] -= 2*math.pi
            overlapping_angles = utils.find_overlapping_angles_within_threshold(
                angles_to_compare_against, collision_free_angles_in_comparing_node_frame, self.angle_resolution/2.0)
            if len(overlapping_angles) > 0:
                if query_angle is None:
                    nodes_with_overlapping_angles.append(query_node)
                    indices_of_nodes_kept.append(index)
                elif utils.query_angle_in_list_of_angles(overlapping_angles, query_angle, self.angle_resolution/2.0):
                    nodes_with_overlapping_angles.append(query_node)
                    indices_of_nodes_kept.append(index)
                elif utils.query_angle_in_list_of_angles(overlapping_angles, query_angle+correction_angle, self.angle_resolution/2.0):
                    nodes_with_overlapping_angles.append(query_node)
                    indices_of_nodes_kept.append(index)

        return nodes_with_overlapping_angles, indices_of_nodes_kept

    def construct_gripper_pose_for_node(self, node, previous_gripper_angle, correction_angle_between_two_nodes):
        position_of_gripper = graph_utils.get_node_data(self.dmg, node, "face_center")
        secondary_gripper_position = graph_utils.get_node_data(self.dmg, node, "secondary_finger_location")[0]

        reference_frame = graph_utils.get_node_data(self.dmg, node, "reference_frame")

        current_collision_free_angles_for_gripper = graph_utils.get_node_data(self.dmg, node, "collision_free_angles")

        current_collision_free_angles_for_gripper = current_collision_free_angles_for_gripper+correction_angle_between_two_nodes

        current_collision_free_angles_for_gripper = utils.subtract_2pi_from_all_angles_in_array_over_2pi(
            current_collision_free_angles_for_gripper)
        current_collision_free_angles_for_gripper = utils.map_angles_within_threshold_from_2pi_to_zero(
            current_collision_free_angles_for_gripper, self.angle_resolution/2)
        if utils.query_angle_within_threshold_of_angle_array(previous_gripper_angle, current_collision_free_angles_for_gripper, self.angle_resolution/2.0):
            current_gripper_angle = (previous_gripper_angle+(2*math.pi-correction_angle_between_two_nodes)) % (2*math.pi)
            current_primary_finger_orientation = utils.rotate_coordinate_system_along_axis_with_angle(
                reference_frame, current_gripper_angle, axis=[1, 0, 0])
        else:
            closest_collision_free_angle = utils.find_angle_from_set_closest_to_query_angle(
                current_collision_free_angles_for_gripper, previous_gripper_angle)
            current_gripper_angle = (closest_collision_free_angle+(2*math.pi-correction_angle_between_two_nodes)) % (2*math.pi)
            current_primary_finger_orientation = utils.rotate_coordinate_system_along_axis_with_angle(
                reference_frame, current_gripper_angle, axis=[1, 0, 0])

        pose_of_primary_gripper = utils.create_transformation_matrix_from_rot_and_trans(
            current_primary_finger_orientation[:3, :3], position_of_gripper)

        current_secondary_finger_orientation = utils.rotate_coordinate_system_along_axis_with_angle(
            current_primary_finger_orientation, math.pi, axis=[0, 0, 1])
        pose_of_secondary_gripper = utils.create_transformation_matrix_from_rot_and_trans(
            current_secondary_finger_orientation[:3, :3], secondary_gripper_position)

        return pose_of_primary_gripper, pose_of_secondary_gripper, current_gripper_angle

    def construct_path_from_nodes(self, nodes, start_angle, goal_angle):
        assert (start_angle in graph_utils.get_node_data(self.dmg, nodes[0],
                                                         "collision_free_angles")), "Start angle needs to be collision free"
        assert (goal_angle in graph_utils.get_node_data(self.dmg, nodes[-1],
                                                        "collision_free_angles")), "Start angle needs to be collision free"
        # The path consist of transformation matrices that the robot finger needs to track
        primary_finger_path = []
        secondary_finger_path = []

        start_pose_for_primary_gripper, start_pose_for_secondary_gripper, rotation_angle = self.construct_gripper_pose_for_node(
            nodes[0], start_angle, 0)
        primary_finger_path.append(start_pose_for_primary_gripper)
        secondary_finger_path.append(start_pose_for_secondary_gripper)

        for i in range(1, len(nodes)-1):
            overlapping_collision_free_angles_between_nodes = graph_utils.get_edge_data(
                self.correction_angles_graph, nodes[i-1], nodes[i], "correction_angle")

            intermediary_pose_for_primary_gripper, intermediary_pose_for_secondary_gripper, rotation_angle = self.construct_gripper_pose_for_node(
                nodes[i], rotation_angle, overlapping_collision_free_angles_between_nodes)
            primary_finger_path.append(intermediary_pose_for_primary_gripper)
            secondary_finger_path.append(intermediary_pose_for_secondary_gripper)

        goal_pose_for_primary_gripper, goal_pose_for_secondary_gripper, rotation_angle = self.construct_gripper_pose_for_node(
            nodes[-1], goal_angle, 0)
        primary_finger_path.append(goal_pose_for_primary_gripper)
        secondary_finger_path.append(goal_pose_for_secondary_gripper)

        return np.asarray(primary_finger_path), np.asarray(secondary_finger_path)

    def graphically_select_start_and_goal_position_and_orientation_for_planning(self):
        self.graphically_select_start_and_goal_positions()
        self.graphically_select_start_and_goal_angles()
        return self.start_node, self.goal_node, self.start_angle, self.goal_angle

    def graphically_select_start_and_goal_positions(self):
        self.start_node = None
        self.goal_node = None
        self.visualize(interactive=True)

    def graphically_select_start_and_goal_angles(self):
        self.start_angle = None
        self.goal_angle = None
        self.visualize_admissible_angles_for_node(self.start_node, interactive=True)
        collision_free_angles_for_start_node = graph_utils.get_node_data(self.dmg, self.start_node, "collision_free_angles")
        self.start_angle = collision_free_angles_for_start_node[self.start_angle_index]
        self.visualize_admissible_angles_for_node(self.goal_node, interactive=True)
        collision_free_angles_for_goal_node = graph_utils.get_node_data(self.dmg, self.goal_node, "collision_free_angles")
        self.goal_angle = collision_free_angles_for_goal_node[self.goal_angle_index]

    def visualize(self, interactive=False, show=True):
        subgraphs = [self.dmg.subgraph(c) for c in nx.connected_components(self.dmg)]
        num_colors = len(subgraphs)+1
        cmap = visualize.get_cmap(num_colors)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        if interactive:
            self.plot_to_subgraph_dic = {}
            self.plot_to_edges = {}
        for i, subgraph in enumerate(subgraphs):
            color = cmap(i)
            plot, edges = visualize.visualize_3D_graph(subgraph, vertex_color=color, edge_color=color, ax=ax, interactive=interactive)
            if interactive:
                self.plot_to_subgraph_dic[plot] = subgraph
                self.plot_to_edges[plot] = edges
        visualize.visualize_mesh_matplot(self.mesh, ax)
        if interactive:
            fig.canvas.mpl_connect('pick_event', self.pick_graph_nodes)

        if show:
            plt.axis('off')
            plt.show()

    def visualize_admissible_angles_for_node(self, nodes, interactive):
        plt.close()
        subgraphs = [self.dmg.subgraph(nx.node_connected_component(self.dmg, c)) for c in nodes]

        num_colors = len(subgraphs)+1
        cmap = visualize.get_cmap(num_colors)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        if interactive:
            self.plot_to_node = {}
        for i, subgraph in enumerate(subgraphs):
            plot = visualize.visualize_node_and_admissible_angles(
                subgraph, nodes[i], color=cmap(i), plot=False, ax=ax, interactive=interactive)
            if interactive:
                self.plot_to_node[plot] = nodes[i]
        visualize.visualize_mesh_matplot(self.mesh, ax)
        if interactive:
            fig.canvas.mpl_connect('pick_event', self.pick_graph_node_angle)

        plt.axis('off')
        plt.show()

    def pick_graph_nodes(self, event):
        try:
            subgraph = self.plot_to_subgraph_dic[event.artist]
        except KeyError:
            raise ("Could not find subgraph that correspond to selected node")

        index = event.artist._z_markers_idx[event.ind]
        list_of_nodes = np.asarray(subgraph.nodes())
        node_indices = list_of_nodes[index]
        if len(node_indices) == 1 or self.node_positions_match(node_indices, subgraph):
            if self.start_node is None:
                self.start_node = node_indices
                for plot in self.plot_to_subgraph_dic.keys():
                    if event.artist != plot:
                        visualize.hide_plot(plot, self.plot_to_edges[plot])
                plt.draw()
            elif self.start_node is not None and node_indices not in self.start_node:
                self.goal_node = node_indices
                plt.close("all")

    def node_positions_match(self, node_indices, graph):
        first_node = node_indices[0]
        position = graph_utils.get_node_data(graph, first_node, "face_center")
        for node in node_indices[1:]:
            query_position = graph_utils.get_node_data(graph, node, "face_center")
            if np.any(position != query_position):
                return False
        return True

    def pick_graph_node_angle(self, event):
        try:
            node = self.plot_to_node[event.artist]
        except KeyError:
            raise ("Could not find subgraph that correspond to selected node")

        if len(event.ind) == 1:
            angle_index = int(event.ind[0])
            if self.start_angle is None:
                self.start_angle_index = angle_index
                self.start_node = node
            elif self.goal_angle is None:
                self.goal_angle_index = angle_index
                self.goal_node = node
            plt.close("all")

    def save_dmg_to_file(self, folder, filename):
        graph_utils.save_graph_to_filename(self.dmg, folder+filename+"_dmg.gpickle")
        graph_utils.save_graph_to_filename(self.correction_angles_graph, folder+filename+"_correction_angles.gpickle")

    def load_dmg_from_file(self, folder, filename):
        self.dmg = graph_utils.load_graph_from_file(folder+filename+"_dmg.gpickle")
        self.correction_angles_graph = graph_utils.load_graph_from_file(folder+filename+"_correction_angles.gpickle")

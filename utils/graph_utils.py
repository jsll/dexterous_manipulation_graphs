import networkx as nx
import numpy as np
from utils import utils
import math
import pickle


def remove_edges(graph, edges):
    graph.remove_edges_from(edges)


def populate_node_of_graph(graph, node, attribute, data):
    graph.nodes[node][attribute] = data


def populate_edge_of_graph(graph, node_x, node_y, attribute, data):
    graph[node_x][node_y][attribute] = data


def get_subgraphs_of_graph(graph):
    subgraphs = [graph.subgraph(c) for c in nx.connected_components(graph)]
    return subgraphs


def get_node_data(graph, node, attribute):
    return graph.nodes[node][attribute]


def get_edge_data(graph, node_x, node_y, attribute):
    return graph[node_x][node_y][attribute]


def return_query_nodes_in_graph(query_nodes_to_check, graph):
    nodes_kept = []
    nodes_indices_kept = []
    for index, node in enumerate(query_nodes_to_check):
        if graph.has_node(node):
            nodes_kept.append(node)
            nodes_indices_kept.append(index)
    return nodes_kept, nodes_indices_kept


def copy_node_a_to_b(graph, node_a, node_b):
    edges_from_node = graph.edges(node_a)
    for _, to_node in edges_from_node:
        graph.add_edge(node_b, to_node)
    attributes = graph.nodes[node_a]
    graph.nodes[node_b].update(attributes)


def get_all_nodes_connected_to_query_node(graph, query_node):
    nodes_connected_to_query_node = nx.node_connected_component(graph, query_node)
    return nodes_connected_to_query_node


def graph_exist_in_list_of_graphs(query_graph, list_of_graphs):
    for graph in list_of_graphs:
        if nx.is_isomorphic(graph, query_graph):
            return True
    return False


def find_graphs_that_contain_both_start_and_goal_nodes(graph, start_nodes, goal_nodes):
    valid_graphs = []
    for start_node in start_nodes:
        C = get_all_nodes_connected_to_query_node(graph, start_node)
        for goal_node in goal_nodes:
            if goal_node in C:
                graph_containing_start_and_goal_node = graph.subgraph(C)
                if not graph_exist_in_list_of_graphs(graph_containing_start_and_goal_node, valid_graphs):
                    valid_graphs.append(graph_containing_start_and_goal_node)
                break
    return valid_graphs


def node_in_graph(graph, node):
    return node in graph


def find_nodes_with_attribute_value(graph, attribute, value):
    nodes = []
    for node in graph.nodes():
        if graph.nodes[node].get(attribute) == value:
            nodes.append(node)
    return nodes


def generate_new_node(graph):
    graph_nodes = list(graph.nodes())
    if len(graph_nodes) == 0:
        new_node_index = 0
    else:
        graph_nodes.sort()
        new_node_index = graph_nodes[-1]+1
    graph.add_node(new_node_index)
    return new_node_index


def create_reference_frame_to_graph(graph):
    reference_frame = np.eye(4)

    position = calculate_average_position_of_graph_nodes(graph)
    reference_frame[:3, 3] = position

    x_axis = calculate_average_normal_vector_of_graph_nodes(graph)
    assert np.any(np.isnan(x_axis)) == False, "The average x-axis is nan meaning that the average of all the vectors cancel each other out"
    y_axis, z_axis = utils.vectors_spanning_plane_to_normal(x_axis)

    reference_frame[:3, 0] = x_axis
    reference_frame[:3, 1] = y_axis
    reference_frame[:3, 2] = z_axis
    return reference_frame


def create_reference_frame_to_node(graph, node):
    reference_frame = np.eye(4)

    reference_frame[:3, 3] = graph.nodes[node]["face_center"]
    x_axis = graph.nodes[node]["face_normal"]

    y_axis, z_axis = utils.vectors_spanning_plane_to_normal(x_axis)

    reference_frame[:3, 0] = x_axis
    reference_frame[:3, 1] = y_axis
    reference_frame[:3, 2] = z_axis

    return reference_frame


def calculate_average_normal_vector_of_graph_nodes(graph):
    average_normal_vector = np.zeros(3)
    for node in graph.nodes():
        average_normal_vector += graph.nodes[node]['face_normal']
    average_normal_vector /= len(graph.nodes())
    norm_of_average_vector = np.linalg.norm(average_normal_vector)
    assert norm_of_average_vector != 0, "The average vector is the zero vector meaning that all the normal vectors used to calculate it \
        cancels each other out. This often happens when using a sphere. To fix this issue, set the reference_frame_per_node flag"
    average_normal_vector /= np.linalg.norm(average_normal_vector)
    return average_normal_vector


def calculate_average_position_of_graph_nodes(graph):
    average_position = np.zeros(3)
    for node in graph.nodes():
        average_position += graph.nodes[node]['face_center']
    average_position /= len(graph.nodes())
    return average_position


def remove_2pi_angle_from_graph_if_0_angle_exist(graph):
    # Only exists for 0 and 2*pi angles so only need to check those
    for node in graph.nodes():
        collision_free_angles = get_node_data(graph, node, "collision_free_angles")
        if (0 and 2*math.pi) in collision_free_angles:
            collision_free_angles.sort()
            collision_free_angles = collision_free_angles[:-1]
            populate_node_of_graph(graph, node, "collision_free_angles", collision_free_angles)


def find_nodes_with_almost_antiparallel_normals_to_query_node(graph, list_of_nodes, node_to_compare_against, threshold=0.3):
    normal_to_compare_against = graph.nodes[node_to_compare_against].get("face_normal")
    nodes_with_antiparallel_normals = []
    indices_of_nodes_kept = []
    for i, query_node in enumerate(list_of_nodes):
        query_normal = graph.nodes[query_node].get("face_normal")
        # Check if the dot product is almost close to -1 meaning that the vectors essentially point in opposite directions
        if normal_to_compare_against.dot(query_normal) < (threshold-1):
            nodes_with_antiparallel_normals.append(query_node)
            indices_of_nodes_kept.append(i)
    return nodes_with_antiparallel_normals, indices_of_nodes_kept


def nodes_that_intersect_another_nodes_normal_vector(graph, query_node, mesh):
    negative_normal_to_query_node = -1*graph.nodes[query_node]['face_normal'].reshape((1, 3))
    position_of_query_node = graph.nodes[query_node]['face_center'].reshape((1, 3))

    mesh_triangles_that_intersect, locations_for_intersections = utils.object_ray_intersections(
        mesh, position_of_query_node, negative_normal_to_query_node)
    nodes_that_intersect = []
    for intersection_triangle in mesh_triangles_that_intersect:
        nodes_that_intersect += find_nodes_with_attribute_value(graph, "face_idx", intersection_triangle)
    return nodes_that_intersect, locations_for_intersections


def save_graph_to_filename(graph, filename):
    assert ".gpickle" in filename, "The filename has to be a gpickle filename"
    with open(filename, "wb") as input_file:
        pickle.dump(graph, input_file)


def load_graph_from_file(filename):
    assert ".gpickle" in filename, "The filename has to be a gpickle filename"
    with open(filename, "rb") as input_file:
        graph = pickle.load(input_file)
    return graph

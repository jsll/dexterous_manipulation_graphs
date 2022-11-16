import trimesh
import numpy as np
import matplotlib.pyplot as plt
from utils import utils
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import networkx as nx
import matplotlib.animation as animation
from gripper import Gripper


def visualize_mesh(mesh):
    mesh.show()


def visualize_points(points, ax=None, color=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],  s=100, ec="w", color=color)


def visualize_edges(edges, ax=None, color=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    line_collection = Line3DCollection(edges, color=color)
    ax.add_collection(line_collection)

    return line_collection


def visualize_subgraphs_in_different_colors(graph, ax=None, plot_reference_frame=False):
    subgraphs = [graph.subgraph(c) for c in nx.connected_components(graph)]
    num_colors = len(subgraphs)+1
    cmap = get_cmap(num_colors)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    for i, subgraph in enumerate(subgraphs):
        color = cmap(i)
        visualize_3D_graph(subgraph, vertex_color=color, edge_color=color, ax=ax, plot_reference_frame=plot_reference_frame)


def visualize_coordinate_system(coordinate_transformation, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    start_pos = coordinate_transformation[:3, 3]
    x_axis = coordinate_transformation[:3, 0]
    y_axis = coordinate_transformation[:3, 1]
    z_axis = coordinate_transformation[:3, 2]
    ax.quiver(start_pos[0], start_pos[1], start_pos[2], x_axis[0], x_axis[1], x_axis[2], length=0.3, normalize=True, color="r")
    ax.quiver(start_pos[0], start_pos[1], start_pos[2], y_axis[0], y_axis[1], y_axis[2], length=0.3, normalize=True, color="g")
    ax.quiver(start_pos[0], start_pos[1], start_pos[2], z_axis[0], z_axis[1], z_axis[2], length=0.3, normalize=True, color="b")


def visualize_3D_graph(graph, plot_normals=False, vertex_color=None, edge_color=None, ax=None, interactive=False, plot_reference_frame=False):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    xyz_points = []
    for data in graph.nodes.data():
        xyz_point = data[1]['face_center']
        xyz_points.append(xyz_point)
        if plot_normals:
            xyz_normal = data[1]['face_normal']
            ax.quiver(xyz_point[0], xyz_point[1], xyz_point[2], xyz_normal[0], xyz_normal[1], xyz_normal[2], length=0.3, normalize=True)
        if plot_reference_frame:
            visualize_coordinate_system(data[1]['reference_frame'], ax)
    xyz_points = np.asarray(xyz_points)
    edge_xyz = np.array([(graph.nodes[u]['face_center'], graph.nodes[v]['face_center']) for u, v in graph.edges()])

    path_collection = ax.scatter(xyz_points[:, 0], xyz_points[:, 1], xyz_points[:, 2],  s=100,
                                 color=vertex_color, picker=interactive)
    edge_collection = visualize_edges(edge_xyz, color=edge_color, ax=ax)
    return path_collection, edge_collection


def visualize_node_and_admissible_angles(graph, node, color=None, plot=True, ax=None, interactive=False):
    # z-axis point in the direction of the finger and x-axis is normal to the surface for grasping
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    angles = graph.nodes[node].get("collision_free_angles")
    reference_frame = graph.nodes[node].get("reference_frame")
    node_center = graph.nodes[node].get("face_center")
    node_center = np.tile(node_center, (len(angles), 1))
    angle_directions = np.zeros((len(angles), 3))
    node_centers = []
    for data in graph.nodes.data():
        node_centers.append(data[1]['face_center'])
    node_centers = np.asarray(node_centers)
    ax.scatter(node_centers[:, 0], node_centers[:, 1], node_centers[:, 2], s=100, ec="w", color="black")
    for i in range(len(angles)):
        rotated_reference_system = utils.rotate_coordinate_system_along_axis_with_angle(reference_frame, angles[i], axis=[1, 0, 0])
        z_axis = rotated_reference_system[:3, 2]
        angle_directions[i, :] = z_axis
    if color is not None:
        quiver = ax.quiver(node_center[:, 0], node_center[:, 1], node_center[:, 2], angle_directions[:, 0],
                           angle_directions[:, 1], angle_directions[:, 2], length=0.03, normalize=True, picker=interactive, color=color)
    else:
        quiver = ax.quiver(node_center[:, 0], node_center[:, 1], node_center[:, 2], angle_directions[:, 0],
                           angle_directions[:, 1], angle_directions[:, 2], length=0.03, normalize=True, picker=interactive, color="black")

    edge_xyz = np.array([(graph.nodes[u]['face_center'], graph.nodes[v]['face_center']) for u, v in graph.edges()])
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="black")
    if plot:
        plt.show()
    return quiver


def visualize_mesh_matplot(mesh, ax, facecolor=None, edgecolor=None, alpha=0.3):
    mesh = ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], triangles=mesh.faces,
                           Z=mesh.vertices[:, 2], alpha=alpha, color="black", edgecolors=edgecolor)
    mesh.set_fc(facecolor)
    return mesh


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def visualize_list_of_meshes(list_of_meshes, list_of_coordinate_axes=[]):
    scene = trimesh.scene.scene.Scene(list_of_meshes)
    if list_of_coordinate_axes is not None:
        for coordinate_axes in list_of_coordinate_axes:
            axis = trimesh.creation.axis(origin_size=0.001, transform=coordinate_axes)
            scene.add_geometry(axis)
    scene.show()


def hide_plot(graph_plot, edge_plot):
    graph_plot.set_picker(None)
    graph_plot.set_alpha(0)
    edge_plot.set_alpha(0)


def visualize_mesh_with_face_indices(mesh, graph=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    visualize_mesh_matplot(mesh, ax)
    face_centers = mesh.triangles_center
    num_face_centers = face_centers.shape[0]
    #visualize_points(face_centers, ax)
    if graph is not None:
        for i in graph.nodes():
            x, y, z = graph.nodes[i].get("face_center")
            ax.text(x, y, z, str(i))
    else:
        for i in range(num_face_centers):
            x, y, z = face_centers[i]
            ax.text(x, y, z, str(i))
    plt.show()


def visualize_mesh_with_list_of_coordinate_systems(mesh, list_of_coordinate_systems):
    scene = trimesh.scene.scene.Scene(mesh)
    for coordinate_system in list_of_coordinate_systems:
        axis = trimesh.creation.axis(origin_size=0.001, transform=coordinate_system)
        scene.add_geometry(axis)
    scene.show()


def visualize_path(primary_finger_path, secondary_finger_path,  mesh):
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    visualize_mesh_matplot(mesh, ax)
    primary_gripper = Gripper()
    secondary_gripper = Gripper()
    visualize_path_points_and_edges(primary_finger_path, ax)
    visualize_path_points_and_edges(secondary_finger_path, ax, color="red")
    primary_gripper.transform_mesh(primary_finger_path[0])
    secondary_gripper.transform_mesh(secondary_finger_path[0])
    ax_gripper_primary_finger = visualize_mesh_matplot(
        primary_gripper .get_mesh(), ax,  facecolor="Black")
    ax_gripper_secondary_finger = visualize_mesh_matplot(
        secondary_gripper .get_mesh(), ax, facecolor="red")
    plt.axis('off')
    ani_primary_finger = animation.FuncAnimation(fig, primary_gripper.gripper_animation, len(
        primary_finger_path), fargs=(primary_finger_path, ax_gripper_primary_finger), interval=500)
    ani_secondary_finger = animation.FuncAnimation(fig, secondary_gripper.gripper_animation, len(
        secondary_finger_path), fargs=(secondary_finger_path, ax_gripper_secondary_finger), interval=500)
    plt.show()


def visualize_path_points_and_edges(path, ax, color="black"):
    points = path[:, :3, 3]
    visualize_points(points, ax, color=color)
    edges = []
    for i in range(points.shape[0]-1):
        edges.append([points[i], points[i+1]])
    edges = np.asarray(edges)
    visualize_edges(edges, ax, color=color)

import argparse
from DMG import DMG
from utils import utils
from area_extractor import SuperVoxel, Centroid
import visualize
from gripper import Gripper


def setup_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-mp', '--mesh_path', type=str, default="test_objects/",
                        help='path to mesh object')
    parser.add_argument('-ar', '--angle_resolution', type=int, default=5,
                        help='The discretization step for the angles.')
    parser.add_argument('-cd', '--translation_threshold', type=float, default=0.3,
                        help='The threshold distance between two normals before they are on different sides of the object. Delta in paper')
    parser.add_argument('-a', '--area_extractor', choices=['supervoxel', 'centroids'], type=str, default="centroids",
                        help="The method used to extract the area a in the paper.")
    parser.add_argument('-v', '--visualize', action='store_true',
                        help='Set if we want to enable some intermediate visualization.')
    parser.add_argument('-s', '--save', action='store_true',
                        help='Set if we want to save the DMG of the object.')
    parser.add_argument('-l', '--load', action='store_true',
                        help='If set, we load the DMG of the object if it exists.')
    parser.add_argument('-o', '--specific_object', type=str, default="",
                        help='If specified, only this object is used.')
    arguments = parser.parse_args()
    if arguments.area_extractor == "supervoxel":
        parser.add_argument('-ps', '--num_point_to_sample_on_mesh', type=int, default=2048,
                            help='Number of points we sample on the mesh.')
        arguments = parser.parse_args()
    else:
        arguments.num_point_to_sample_on_mesh = None

    return arguments


def get_area_extractor_method(area_extractor, num_point_to_sample_on_mesh):
    if area_extractor == "supervoxel":
        extractor = SuperVoxel(num_point_to_sample_on_mesh)
    elif area_extractor == 'centroids':
        extractor = Centroid()
    else:
        raise NotImplementedError("Extractor method "+area_extractor+" is not implemented.")

    return extractor


if __name__ == '__main__':
    args = setup_args()
    mesh_files = utils.get_files_in_folder(args.mesh_path, args.specific_object, ".obj")
    extractor = get_area_extractor_method(args.area_extractor, args.num_point_to_sample_on_mesh)
    gripper = Gripper()
    dmgs = []
    for mesh_file in mesh_files:
        mesh = utils.load_mesh(mesh_file)
        utils.scale_object(mesh, gripper.finger_mesh.scale)
        graph_of_area_a = extractor.extract_area_a(mesh)
        dmg = DMG(args.angle_resolution, args.translation_threshold, mesh)

        separate_reference_frame = False
        if "sphere.obj" in mesh_file or "cylinder.obj" in mesh_file:
            separate_reference_frame = True

        if args.load:
            try:
                mesh_name = utils.get_filename_without_extension(mesh_file)
                filename = mesh_name + "_" + str(args.angle_resolution) + "_"+str(args.translation_threshold)
                dmg.load_dmg_from_file("saved_graphs/", filename)
            except ValueError as e:
                dmg.build_graph(graph_of_area_a, gripper, separate_reference_frame)
        else:
            dmg.build_graph(graph_of_area_a, gripper, separate_reference_frame)
        if args.save:
            mesh_name = utils.get_filename_without_extension(mesh_file)
            filename = mesh_name + "_" + str(args.angle_resolution) + "_"+str(args.translation_threshold)
            dmg.save_dmg_to_file("saved_graphs/", filename)
        if args.visualize:
            dmg.visualize()

        start_node, goal_node, start_angle, goal_angle = dmg.graphically_select_start_and_goal_position_and_orientation_for_planning()
        paths = dmg.in_hand_planning(start_node, goal_node, start_angle, goal_angle)
        if len(paths) == 0:
            print("No path found from start to goal node")
        else:
            for path in paths:
                visualize.visualize_path(path[0], path[1], mesh)
        dmgs.append(dmg)
        gripper.reset()

from utils import utils
import numpy as np
import networkx as nx


class SuperVoxel:
    def __init__(self, num_point_to_sample_on_mesh):
        self.num_point_to_sample_on_mesh = num_point_to_sample_on_mesh

    def extract_area_a(self, mesh):
        point_cloud_of_mesh = utils.create_pointcloud_of_mesh(mesh, self.num_point_to_sample_on_mesh)
        supervoxels = self.extract_supervoxels(point_cloud_of_mesh)
        return None

    def extract_supervoxels(self, point_cloud):
        return None


class Centroid:
    def __init__(self) -> None:
        pass

    def extract_area_a(self, mesh, as_quad=False):
        if as_quad:
            area_a = self.extract_as_quadrilateral(mesh)
        else:
            area_a = self.extract_as_triangles(mesh)
        return area_a

    def extract_as_triangles(self, mesh):
        center_of_faces = mesh.triangles_center
        normals_to_faces = mesh.face_normals
        graph_of_area_a = self.create_graph_from_edge_list(mesh.face_adjacency)
        for face_idx, (pa, na) in enumerate(zip(center_of_faces, normals_to_faces)):
            self.populate_node_of_graph(graph_of_area_a, face_idx, "face_center", pa)
            self.populate_node_of_graph(graph_of_area_a, face_idx, "face_normal", na)
            self.populate_node_of_graph(graph_of_area_a, face_idx, "face_idx", face_idx)
        return graph_of_area_a

    def extract_as_quadrilateral(self, mesh):
        coplanar_faces = utils.extract_coplanar_faces_of_mesh(mesh)
        triangle_face_to_quad_face = self.map_triangle_faces_to_quad_faces(coplanar_faces)
        quadrilateral_centers = mesh.triangles_center[coplanar_faces].mean(axis=1)
        quadrilateral_normals = mesh.face_normals[coplanar_faces].mean(axis=1)
        edge_list = self.extract_edge_list_for_quadrilaterals(coplanar_faces, mesh.face_adjacency, triangle_face_to_quad_face)
        graph_of_area_a = self.create_graph_from_edge_list(edge_list)
        for quadrilateral_index, (qc, qn) in enumerate(zip(quadrilateral_centers, quadrilateral_normals)):
            self.populate_node_of_graph(graph_of_area_a, quadrilateral_index, "face_center", qc)
            self.populate_node_of_graph(graph_of_area_a, quadrilateral_index, "face_normal", qn)
            self.populate_node_of_graph(graph_of_area_a, quadrilateral_index, "face_idx", quadrilateral_index)
        return graph_of_area_a

    def extract_edge_list_for_quadrilaterals(self, quadrilateral_faces, triangle_face_adjacency, triangle_face_to_quad_face):
        edge_list = []
        for index, quadrilateral_face in enumerate(quadrilateral_faces):
            triangle_face_1_in_quad = quadrilateral_face[0]
            triangle_face_2_in_quad = quadrilateral_face[1]
            adjacency_faces_to_triangle_1 = triangle_face_adjacency[triangle_face_adjacency[:, 0] == triangle_face_1_in_quad, 1
                                                                    ]
            adjacency_faces_to_triangle_2 = triangle_face_adjacency[triangle_face_adjacency[:, 0] == triangle_face_2_in_quad, 1
                                                                    ]
            adjacency_faces_to_triangle_1 = adjacency_faces_to_triangle_1[adjacency_faces_to_triangle_1 != triangle_face_2_in_quad]
            adjacency_faces_to_triangle_2 = adjacency_faces_to_triangle_2[adjacency_faces_to_triangle_2 != triangle_face_1_in_quad]
            for triangle_face_index in adjacency_faces_to_triangle_1:
                quad_face_index = triangle_face_to_quad_face[triangle_face_index]
                edge_list.append([index, quad_face_index])
            for triangle_face_index in adjacency_faces_to_triangle_2:
                quad_face_index = triangle_face_to_quad_face[triangle_face_index]
                edge_list.append([index, quad_face_index])

        return edge_list

    def map_triangle_faces_to_quad_faces(self, quadrilateral_faces):
        triangle_face_to_quad_face_dict = {}
        for quad_index, quad in enumerate(quadrilateral_faces):
            triangle_1 = quad[0]
            triangle_2 = quad[1]
            triangle_face_to_quad_face_dict[triangle_1] = quad_index
            triangle_face_to_quad_face_dict[triangle_2] = quad_index
        return triangle_face_to_quad_face_dict

    def create_graph_from_edge_list(self, edge_list):
        graph = nx.from_edgelist(edge_list)
        return graph

    def populate_node_of_graph(self, graph, node, attribute, data):
        graph.nodes[node][attribute] = data

import open3d as o3d
import openalea.plantgl as plantgl
import openalea.plantgl.math as mt
import openalea.plantgl.scenegraph as sg
import openalea.plantgl.algo as alg
import numpy as np

def convert_scene_to_mesh(scene, file_path=None):

    disc = alg.Discretizer()

    vertices = []
    vertex_colors = []
    mesh_faces = []

    for scene_obj in scene:
        if scene_obj.apply(disc):
            rez = disc.result
            if isinstance(rez, plantgl.scenegraph._pglsg.PointSet):
                continue

            pts = rez.pointList
            faces = rez.indexList

            c = scene_obj.appearance.ambient
            color = np.array([c.red, c.green, c.blue]) / 255.0
            base_idx = len(vertices)

            for pt in pts:
                vertices.append(pt)
                vertex_colors.append(color)

            for face in faces:
                face = (np.array(face) + base_idx).astype(np.int32)
                if len(face) == 3:
                    mesh_faces.append(face)
                elif len(face) == 4:
                    mesh_faces.append(face[[0,1,2]])
                    mesh_faces.append(face[[2,3,0]])
                else:
                    raise Exception('Face with {} points?'.format(len(face)))


    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    mesh.triangles = o3d.utility.Vector3iVector(mesh_faces)

    if file_path is not None:
        o3d.io.write_triangle_mesh(file_path, mesh)

    return mesh

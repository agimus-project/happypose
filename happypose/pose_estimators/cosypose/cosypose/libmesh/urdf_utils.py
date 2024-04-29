import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import numpy as np
import trimesh
from tqdm import tqdm

from happypose.toolbox.datasets.object_dataset import RigidObjectDataset


def convert_rigid_body_dataset_to_urdfs(
    rb_ds: RigidObjectDataset,
    urdf_dir: Path,
    texture_size=(1024, 1024),
    override=True,
    label2objname_file_name: str = "objname2label.json",
):
    """
    Converts a RigidObjectDataset into a directory of urdf files with structure:

    urdf_dir/<label2objname_file_name>.json
    urdf_dir/obj_000001/obj_000001.mtl
                        obj_000001.obj
                        obj_000001_texture.png
                        obj_000001.urdf
    urdf_dir/obj_000002/obj_000002.mtl
                        obj_000002.obj
                        obj_000002_texture.png
                        obj_000002.urdf

    <label2objname_file_name>.json: stores a map between object file names (e.g. obj_000002) and
    object labels used in happypose (e.g. the detector may output "ycbv-obj_000002")
    """
    if override and urdf_dir.exists():
        shutil.rmtree(urdf_dir, ignore_errors=True)
    urdf_dir.mkdir(exist_ok=True, parents=True)

    objname2label = {}
    for obj in tqdm(rb_ds.list_objects):
        objname = obj.mesh_path.with_suffix("").name  # e.g. "obj_000002"
        objname2label[objname] = obj.label  # e.g. obj_000002 -> ycbv-obj_000002
        # Create object folder
        obj_urdf_dir = urdf_dir / objname
        obj_urdf_dir.mkdir(exist_ok=True)  # urdf_dir/obj_000002/ created
        # Convert mesh from ply to obj
        obj_path = (obj_urdf_dir / objname).with_suffix(".obj")
        if obj.mesh_path.suffix == ".ply":
            ply_to_obj(obj.mesh_path, obj_path, texture_size)
        else:
            ValueError(f"{obj.mesh_path.suffix} file type not supported")
        # Create a .urdf file associated to the .obj file
        urdf_path = obj_path.with_suffix(".urdf")
        obj_to_urdf(obj_path, urdf_path)

    with open(urdf_dir / label2objname_file_name, "w") as fp:
        json.dump(objname2label, fp)


def ply_to_obj(ply_path: Path, obj_path: Path, texture_size=None):
    assert obj_path.suffix == ".obj"
    mesh = trimesh.load(ply_path)
    obj_label = obj_path.with_suffix("").name

    # adapt materials according to previous example meshes
    if mesh.visual.defined:
        mesh.visual.material.ambient = np.array([51, 51, 51, 255], dtype=np.uint8)
        mesh.visual.material.diffuse = np.array([255, 255, 255, 255], dtype=np.uint8)
        mesh.visual.material.specular = np.array([255, 255, 255, 255], dtype=np.uint8)
        mesh.visual.material.name = obj_label + "_texture"

    # print(mesh.visual.uv)
    kwargs_export = {"mtl_name": f"{obj_label}.mtl"}
    _ = mesh.export(obj_path, **kwargs_export)


def obj_to_urdf(obj_path, urdf_path):
    obj_path = Path(obj_path)
    urdf_path = Path(urdf_path)
    assert urdf_path.parent == obj_path.parent

    geometry = ET.Element("geometry")
    mesh = ET.SubElement(geometry, "mesh")
    mesh.set("filename", obj_path.name)
    mesh.set("scale", "1.0 1.0 1.0")

    material = ET.Element("material")
    material.set("name", "mat_part0")
    color = ET.SubElement(material, "color")
    color.set("rgba", "1.0 1.0 1.0 1.0")

    inertial = ET.Element("inertial")
    origin = ET.SubElement(inertial, "origin")
    origin.set("rpy", "0 0 0")
    origin.set("xyz", "0.0 0.0 0.0")

    mass = ET.SubElement(inertial, "mass")
    mass.set("value", "0.1")

    inertia = ET.SubElement(inertial, "inertia")
    inertia.set("ixx", "1")
    inertia.set("ixy", "0")
    inertia.set("ixz", "0")
    inertia.set("iyy", "1")
    inertia.set("iyz", "0")
    inertia.set("izz", "1")

    robot = ET.Element("robot")
    robot.set("name", obj_path.with_suffix("").name)

    link = ET.SubElement(robot, "link")
    link.set("name", "base_link")

    visual = ET.SubElement(link, "visual")
    visual.append(geometry)
    visual.append(material)

    collision = ET.SubElement(link, "collision")
    collision.append(geometry)

    link.append(inertial)

    xmlstr = minidom.parseString(ET.tostring(robot)).toprettyxml(indent="   ")
    Path(urdf_path).write_text(xmlstr)  # Write xml file
    return

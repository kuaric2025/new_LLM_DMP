"""Build URDF for objects.
Scripts from https://github.com/harvard-microrobotics/object2urdf
"""

import os
from object_builder import ObjectBuilder

# Build entire libraries of URDFs
# This is only suitable for objects built with single obj/stl file
# Models such as robots or articulated objects will not work properly
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
object_folder = root_dir + "/ycb"

max_convex_hull = 5  # for coacd, default -1
decimate_face_count = 32  # for decimation, default -1 (no decimation)

builder = ObjectBuilder(
    object_folder,
    "_prototype.urdf",
    "_prototype.xml",
    "ycb_mass.json",
)
builder.build_library(
    # Fit object mesh with OBB
    # Fit yaw only for OBB, not roll/pitch
    # This assumes the object is modeled upright
    fit_obb=True,
    fit_yaw_only=True,
    # Build URDF and XML
    force_overwrite=True,
    center="mass",  # object center is at mass center
    # Convex Decomposition
    decompose_concave=True,  # build *_coacd.obj collision mesh
    force_decompose=True,  # overwrite decomposed files
    max_convex_hull=max_convex_hull,
    decimate_face_count=decimate_face_count,
)

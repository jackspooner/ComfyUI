"""Build the optional CPU mesh encoder extension from local sources; no downloads."""
import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

root = Path(__file__).resolve().parents[1]
eigen = Path(os.environ.get("EIGEN_INCLUDE_DIR", "/usr/include/eigen3"))
if not (eigen / "Eigen/Dense").is_file():
    raise SystemExit("Set EIGEN_INCLUDE_DIR to an Eigen 3 header directory containing Eigen/Dense.")
os.chdir(root)
source = root / "comfy/ldm/trellis2/voxelize"
setup(
    name="comfy-trellis2-voxel",
    packages=[],
    ext_modules=[CppExtension(
        "comfy.ldm.trellis2._mesh_voxel",
        [str(source / "bindings.cpp"), str(source / "flexible_dual_grid.cpp")],
        include_dirs=[str(eigen.resolve())],
        extra_compile_args=["-O2", "-g0", "-std=c++20"],
    )],
    cmdclass={"build_ext": BuildExtension},
    script_args=["build_ext", "--inplace"],
)

"""Setup shim: lands the activation .pth directly in site-packages.

Wheels install every root-level file into site-packages (the directory the
``site`` module actually scans for ``.pth`` files), so we attach the .pth via
``cmdclass`` post-build relocation instead of ``data_files`` (whose wheel
``.data/data`` target is the environment prefix root, not site-packages).
"""

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class build_py(_build_py):
    """Copy the activation .pth next to the package in the wheel."""

    def run(self):
        super().run()
        import shutil
        dest = os.path.join(self.build_lib, "tilelang_sail_cuda_wrapper.pth")
        shutil.copyfile("tilelang_sail_cuda_wrapper.pth", dest)


import os  # noqa: E402

setup(cmdclass={"build_py": build_py})

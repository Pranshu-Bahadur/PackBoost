from pathlib import Path
import sys

from setuptools import setup

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from setup_native import build_native_extension

ext_modules, cmdclass = build_native_extension()

setup(ext_modules=ext_modules, cmdclass=cmdclass)

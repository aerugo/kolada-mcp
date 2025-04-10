"""
setup.py for packaging Kolada MCP with py2app
"""

# setup.py

import sys
import zlib

from py2app.build_app import py2app as Py2AppCommand


# 1) Monkey-patch: define a function that safely copies zlib
def safe_copy_zlib(self, arcdir):
    """
    Original py2app code: self.copy_file(zlib.__file__, os.path.dirname(arcdir))

    We'll check if zlib.__file__ exists first.
    """
    zlib_path = getattr(zlib, "__file__", None)
    if zlib_path:
        self.copy_file(zlib_path, os.path.dirname(arcdir))
    else:
        # If there's no zlib.__file__, just skip
        print("[py2app monkey-patch] zlib.__file__ not found; skipping copy.")


# 2) Patch 'build_executable' or wherever the code references zlib.__file__
# We'll store the original method for reference if needed
original_build_executable = Py2AppCommand.build_executable


def patched_build_executable(self, script, pkgexts, extensions, extra_scripts):
    # We monkey-patch inside the method or near the line it tries to copy zlib.
    # We'll do a quick hack: search the code for zlib reference or override the line
    # that does the copy_file call.
    #
    # Easiest approach is to replicate or wrap the original method logic,
    # but skip or patch the line referencing zlib.__file__.

    # we can either re-implement or do a pre-processing
    # For a simple approach, we can just override
    # "self.copy_file(zlib.__file__, ...)"
    # with our safe_copy_zlib call. That means editing the code in memory.
    # This is more complex if it’s inline, so we might just do:

    # 1) run original method, but if it fails with zlib error, handle it
    try:
        return original_build_executable(
            self, script, pkgexts, extensions, extra_scripts
        )
    except AttributeError as e:
        if "zlib.__file__" in str(e):
            # Perform our safe copy
            # For instance, we can guess arcdir from self.get_destination_path(...)
            # But we might not know the exact arcdir without the original code path.
            # So another approach is to skip or do partial re-run.
            print("[py2app monkey-patch] Caught zlib.__file__ error, skipping copy.")
            # Possibly do: safe_copy_zlib(self, some_path)
            # and continue or just ignore.
            return
        else:
            raise


# 3) Override the method on the class
Py2AppCommand.build_executable = patched_build_executable

# 4) Now import setuptools and do the normal setup
from setuptools import setup

sys.setrecursionlimit(10000)


APP = ["src/server.py"]
DATA_FILES: list[tuple[str, list[str]]] = []
OPTIONS = {
    "packages": [
        "httpx",
        "mcp",
        "numpy",
        "polars",
        "sentence_transformers",
        "statistics",
    ],
    "excludes": ["packaging", "wheel", "pip", "distutils", "setuptools"],
}

setup(
    app=APP,
    name="KoladaMCP",
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)

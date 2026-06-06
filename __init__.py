"""Top-level package for applesharp."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """Apyekitosh"""
__email__ = "apy.q.ns@gmail.com"
__version__ = "0.0.1"

from .src.applesharp.nodes import NODE_CLASS_MAPPINGS
from .src.applesharp.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

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

# Register a route to serve PLY files from the output directory.
# ComfyUI's built-in /view endpoint is image-only; PLY files need their own route.
try:
    import os
    from aiohttp import web
    from server import PromptServer
    import folder_paths

    @PromptServer.instance.routes.get("/applesharp/ply")
    async def serve_ply(request):
        import io
        from plyfile import PlyData

        filename  = request.query.get("filename", "")
        subfolder = request.query.get("subfolder", "")

        output_dir = folder_paths.get_output_directory()
        raw_path   = os.path.join(output_dir, subfolder, filename) if subfolder else os.path.join(output_dir, filename)
        full_path  = os.path.realpath(raw_path)
        safe_root  = os.path.realpath(output_dir)

        if not full_path.startswith(safe_root + os.sep) and full_path != safe_root:
            return web.Response(status=403, text="Access denied")

        if not os.path.isfile(full_path):
            return web.Response(status=404, text="PLY file not found")

        # gsplat.js only supports float properties. SHARP's PLY includes extra metadata
        # elements (image_size, color_space, version…) that use uint/uchar types and
        # cause a parse error. Strip everything except the vertex element before serving.
        plydata = PlyData.read(full_path)
        vertex_only = PlyData([e for e in plydata.elements if e.name == "vertex"],
                               text=False)
        buf = io.BytesIO()
        vertex_only.write(buf)
        return web.Response(body=buf.getvalue(),
                            content_type="application/octet-stream")

    print("[SHARP] Registered /applesharp/ply route")
except Exception as _e:
    print(f"[SHARP] Warning: could not register PLY route: {_e}")

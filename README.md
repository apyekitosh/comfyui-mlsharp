# ComfyUI-MLSharp

ComfyUI nodes for [Apple's ML-SHARP](https://github.com/apple/ml-sharp) — single-image 3D Gaussian splatting reconstruction.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/apyekitosh/comfyui-mlsharp.git
cd comfyui-mlsharp
pip install -r requirements.txt
```

Then restart ComfyUI.

## Nodes

### SHARP Gaussian Splat

Converts a single RGB image into a 3D Gaussian Splat (`.ply` file) using Apple's SHARP model.

| Input | Description |
|---|---|
| `image` | Input RGB image |
| `focal_length_mm` | 35mm-equivalent focal length (default 30) |
| `filename_prefix` | Prefix for the output PLY file |
| `model_mode` | `persistent` saves the model to `models/sharp/` for reuse. `temporary` downloads to a temp file each session. |

The model is downloaded automatically on first use from Apple's CDN.

### SHARP Splat Preview

Makes a `.ply` Gaussian Splat file available for preview in the ComfyUI frontend. Connect it to the output of **SHARP Gaussian Splat**.

| Input | Description |
|---|---|
| `ply_path` | Path to a `.ply` file (from SHARP Gaussian Splat) |

### SHARP Shot Render

Renders a `.ply` Gaussian Splat to a 2D image using a CPU rasterizer (no CUDA required). Includes an interactive 3D viewport — orbit/pan/zoom in the viewer, then re-run the node to render the image at the current camera angle.

| Input | Description |
|---|---|
| `ply_path` | Path to a `.ply` file (from SHARP Gaussian Splat) |
| `output_width` / `output_height` | Render resolution (default 1024x1024) |
| `gaussian_scale` | Global scale multiplier for Gaussian sizes |
| `max_gaussians` | Max splats to render (0 = unlimited). Lower values = faster preview |
| `background` | Background color (black, mid-gray, white) |

The node outputs an `IMAGE` tensor that can be chained to `PreviewImage`, `SaveImage`, or any other image node.

> **Tip:** Orbit in the viewport to find the angle you want, click **Commit View** (or just let go of the mouse — the camera auto-commits on drag-end), then re-run the node. The image output will match what you see.

---

This project includes code from Apple's ML-SHARP project.

Copyright (C) 2025 Apple Inc.

The original license and acknowledgements can be found in:
- [ml-sharp/LICENSE](ml-sharp/LICENSE)
- [ml-sharp/ACKNOWLEDGEMENTS](ml-sharp/ACKNOWLEDGEMENTS)

# ComfyUI-MLSharp

ComfyUI nodes for [Apple's ML-SHARP](https://github.com/apple/ml-sharp) — single-image 3D Gaussian splatting reconstruction.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<your-username>/comfyui-mlsharp.git
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

---

This project includes code from Apple's ML-SHARP project.

Copyright (C) 2025 Apple Inc.

The original license and acknowledgements can be found in:
- [ml-sharp/LICENSE](ml-sharp/LICENSE)
- [ml-sharp/ACKNOWLEDGEMENTS](ml-sharp/ACKNOWLEDGEMENTS)

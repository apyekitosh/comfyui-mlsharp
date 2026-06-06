import sys
from pathlib import Path

_SHARP_SRC = str(Path(__file__).resolve().parents[2] / "ml-sharp" / "src")
if _SHARP_SRC not in sys.path:
    sys.path.insert(0, _SHARP_SRC)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
MODEL_FILENAME = "sharp_2572gikvuh.pt"


class SharpGaussianNode:
    _predictor = None
    _device = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "focal_length_mm": ("FLOAT", {
                    "default": 30.0,
                    "min": 1.0,
                    "max": 200.0,
                    "step": 0.5,
                    "tooltip": "35mm-equivalent focal length. Use 30 for a typical wide shot if unknown.",
                }),
                "filename_prefix": ("STRING", {"default": "sharp"}),
            }
        }

    RETURN_TYPES = ("STRING", "EXTRINSICS", "INTRINSICS")
    RETURN_NAMES = ("ply_path", "extrinsics", "intrinsics")
    FUNCTION = "predict"
    CATEGORY = "3D"
    OUTPUT_NODE = True

    def predict(self, image, focal_length_mm, filename_prefix):
        import torch
        import torch.nn.functional as F
        import numpy as np
        import folder_paths
        from sharp.models import PredictorParams, create_predictor
        from sharp.utils.gaussians import save_ply, unproject_gaussians
        from sharp.utils.io import convert_focallength

        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch, "mps") and torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        model_dir = Path(folder_paths.models_dir) / "sharp"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / MODEL_FILENAME

        if not model_path.exists():
            print(f"[SHARP] Downloading model to {model_path} ...")
            import urllib.request
            urllib.request.urlretrieve(DEFAULT_MODEL_URL, model_path)
            print("[SHARP] Download complete.")

        if SharpGaussianNode._predictor is None or SharpGaussianNode._device != device:
            print(f"[SHARP] Loading model on {device} ...")
            state_dict = torch.load(model_path, weights_only=True, map_location=device)
            predictor = create_predictor(PredictorParams())
            predictor.load_state_dict(state_dict)
            predictor.eval().to(device)
            SharpGaussianNode._predictor = predictor
            SharpGaussianNode._device = device

        img_np = (image[0].cpu().numpy() * 255).astype("uint8")
        height, width = img_np.shape[:2]
        f_px = convert_focallength(width, height, focal_length_mm)

        print(f"[SHARP] Running inference on {width}x{height} image (f={f_px:.1f}px) ...")
        # Inlined from sharp.cli.predict.predict_image — that module also imports
        # sharp.cli.render, which hard-imports the gsplat package (CUDA build-only,
        # needed solely for trajectory rendering, not for single-image inference).
        torch_device = torch.device(SharpGaussianNode._device)
        internal_shape = (1536, 1536)
        with torch.no_grad():
            image_pt = torch.from_numpy(img_np.copy()).float().to(torch_device).permute(2, 0, 1) / 255.0
            _, img_h, img_w = image_pt.shape
            disparity_factor = torch.tensor([f_px / img_w]).float().to(torch_device)

            image_resized_pt = F.interpolate(
                image_pt[None],
                size=(internal_shape[1], internal_shape[0]),
                mode="bilinear",
                align_corners=True,
            )

            gaussians_ndc = SharpGaussianNode._predictor(image_resized_pt, disparity_factor)

            intrinsics_pt = torch.tensor(
                [
                    [f_px, 0, img_w / 2, 0],
                    [0, f_px, img_h / 2, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            ).float().to(torch_device)
            intrinsics_resized = intrinsics_pt.clone()
            intrinsics_resized[0] *= internal_shape[0] / img_w
            intrinsics_resized[1] *= internal_shape[1] / img_h

            gaussians = unproject_gaussians(
                gaussians_ndc, torch.eye(4).to(torch_device), intrinsics_resized, internal_shape
            )

        out_dir = Path(folder_paths.get_output_directory()) / "gaussians"
        out_dir.mkdir(exist_ok=True)

        # avoid overwriting existing files
        ply_path = out_dir / f"{filename_prefix}.ply"
        counter = 1
        while ply_path.exists():
            ply_path = out_dir / f"{filename_prefix}_{counter:04d}.ply"
            counter += 1

        save_ply(gaussians, f_px, (height, width), ply_path)
        print(f"[SHARP] Saved gaussian splat to {ply_path}")

        # SHARP predicts in a camera-at-origin frame (no pose estimation), so
        # extrinsics is always identity — matches the "dummy extrinsics" save_ply embeds in the PLY.
        extrinsics = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        intrinsics = [
            [f_px, 0.0, width / 2.0],
            [0.0, f_px, height / 2.0],
            [0.0, 0.0, 1.0],
        ]

        return (str(ply_path), extrinsics, intrinsics)


NODE_CLASS_MAPPINGS = {
    "SharpGaussian": SharpGaussianNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SharpGaussian": "SHARP Gaussian Splat",
}

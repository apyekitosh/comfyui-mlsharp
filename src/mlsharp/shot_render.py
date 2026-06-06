import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

try:
    import folder_paths
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
except Exception:
    COMFYUI_OUTPUT_FOLDER = None

try:
    from numba import njit

    @njit(cache=False, fastmath=True)
    def _accumulate_splats_numba(
        image: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        colors: np.ndarray,
        opacities: np.ndarray,
        sigma2: np.ndarray,
        radius: np.ndarray,
        width: int,
        height: int,
    ) -> None:
        n = u.shape[0]
        for idx in range(n):
            r = int(math.ceil(float(radius[idx])))
            ui = float(u[idx])
            vi = float(v[idx])
            x0 = max(0, int(math.floor(ui - float(r))))
            x1 = min(width, int(math.ceil(ui + float(r) + 1.0)))
            y0 = max(0, int(math.floor(vi - float(r))))
            y1 = min(height, int(math.ceil(vi + float(r) + 1.0)))
            if x0 >= x1 or y0 >= y1:
                continue

            c00 = float(sigma2[idx, 0, 0])
            c01 = float(sigma2[idx, 0, 1])
            c10 = float(sigma2[idx, 1, 0])
            c11 = float(sigma2[idx, 1, 1])
            det2 = c00 * c11 - c01 * c10
            if det2 <= 1e-12:
                continue
            idet = 1.0 / det2
            inv00 = c11 * idet
            inv01 = -c01 * idet
            inv10 = -c10 * idet
            inv11 = c00 * idet
            inv_cross = inv01 + inv10

            opac = float(opacities[idx])
            cr = float(colors[idx, 0])
            cg = float(colors[idx, 1])
            cb = float(colors[idx, 2])

            max_al = 0.0
            for yy in range(y0, y1):
                dys = float(yy) - vi
                for xx in range(x0, x1):
                    dxs = float(xx) - ui
                    q = inv00 * dxs * dxs + inv_cross * dxs * dys + inv11 * dys * dys
                    al = opac * math.exp(-0.5 * q)
                    if al > 1.0:
                        al = 1.0
                    elif al < 0.0:
                        al = 0.0
                    if al > max_al:
                        max_al = al
            if max_al < 1e-4:
                continue

            for yy in range(y0, y1):
                dys = float(yy) - vi
                for xx in range(x0, x1):
                    dxs = float(xx) - ui
                    q = inv00 * dxs * dxs + inv_cross * dxs * dys + inv11 * dys * dys
                    al = opac * math.exp(-0.5 * q)
                    if al > 1.0:
                        al = 1.0
                    elif al < 0.0:
                        al = 0.0
                    om = 1.0 - al
                    image[yy, xx, 0] = image[yy, xx, 0] * om + cr * al
                    image[yy, xx, 1] = image[yy, xx, 1] * om + cg * al
                    image[yy, xx, 2] = image[yy, xx, 2] * om + cb * al

    _HAS_NUMBA_SPLATS = True
except ImportError:
    _accumulate_splats_numba = None
    _HAS_NUMBA_SPLATS = False


# ── PLY loading ──────────────────────────────────────────────────────────────

def _decode_sh_to_rgb(sh0: np.ndarray) -> np.ndarray:
    coeff = math.sqrt(1.0 / (4.0 * math.pi))
    return np.clip(sh0 * coeff + 0.5, 0.0, 1.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _quat_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quat_xyzw.astype(np.float64), axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    q = (quat_xyzw.astype(np.float64) / norms).astype(np.float32)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    mats = np.empty((q.shape[0], 3, 3), dtype=np.float32)
    mats[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    mats[:, 0, 1] = 2.0 * (xy - wz)
    mats[:, 0, 2] = 2.0 * (xz + wy)
    mats[:, 1, 0] = 2.0 * (xy + wz)
    mats[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    mats[:, 1, 2] = 2.0 * (yz - wx)
    mats[:, 2, 0] = 2.0 * (xz - wy)
    mats[:, 2, 1] = 2.0 * (yz + wx)
    mats[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return mats


def _load_gaussian_ply(ply_path: str) -> dict[str, np.ndarray]:
    path = Path(ply_path)
    plydata = PlyData.read(path)
    vertices = next(e for e in plydata.elements if e.name == "vertex")

    xyz = np.stack([
        np.asarray(vertices["x"], dtype=np.float32),
        np.asarray(vertices["y"], dtype=np.float32),
        np.asarray(vertices["z"], dtype=np.float32),
    ], axis=1)

    sh0 = np.stack([
        np.asarray(vertices["f_dc_0"], dtype=np.float32),
        np.asarray(vertices["f_dc_1"], dtype=np.float32),
        np.asarray(vertices["f_dc_2"], dtype=np.float32),
    ], axis=1)
    colors = _decode_sh_to_rgb(sh0).astype(np.float32)

    opacities = _sigmoid(np.asarray(vertices["opacity"], dtype=np.float32)).astype(np.float32)

    scales = np.exp(np.stack([
        np.asarray(vertices["scale_0"], dtype=np.float32),
        np.asarray(vertices["scale_1"], dtype=np.float32),
        np.asarray(vertices["scale_2"], dtype=np.float32),
    ], axis=1)).astype(np.float32)

    quats = np.stack([
        np.asarray(vertices["rot_1"], dtype=np.float32),
        np.asarray(vertices["rot_2"], dtype=np.float32),
        np.asarray(vertices["rot_3"], dtype=np.float32),
        np.asarray(vertices["rot_0"], dtype=np.float32),
    ], axis=1)

    rot_mats = _quat_to_matrix(quats)
    diag = np.zeros((xyz.shape[0], 3, 3), dtype=np.float32)
    diag[:, 0, 0] = scales[:, 0] ** 2
    diag[:, 1, 1] = scales[:, 1] ** 2
    diag[:, 2, 2] = scales[:, 2] ** 2
    covariances = np.einsum("nij,njk,nlk->nil", rot_mats, diag, rot_mats).astype(np.float32)

    return {"xyz": xyz, "colors": colors, "opacities": opacities, "covariances": covariances}


# ── Camera math ──────────────────────────────────────────────────────────────

def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return (v / norm).astype(np.float32)


def _euler_deg_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    ax, ay, az = math.radians(rx), math.radians(ry), math.radians(rz)
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    rxm = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    rym = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rzm = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (rzm @ rym @ rxm).astype(np.float32)


def _orbit_basis_from_yp(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    return _euler_deg_to_matrix(pitch_deg, yaw_deg, 0.0)


def _apply_roll_to_basis(basis: np.ndarray, roll_deg: float) -> np.ndarray:
    roll_rad = math.radians(roll_deg)
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    ref_right = (basis @ np.array([1.0, 0.0, 0.0], dtype=np.float32)).astype(np.float32)
    ref_down = (basis @ np.array([0.0, 1.0, 0.0], dtype=np.float32)).astype(np.float32)
    forward = (basis @ np.array([0.0, 0.0, 1.0], dtype=np.float32)).astype(np.float32)
    right = (ref_right * cr + ref_down * sr).astype(np.float32)
    down = (ref_down * cr - ref_right * sr).astype(np.float32)
    return np.array([
        [right[0], down[0], forward[0]],
        [right[1], down[1], forward[1]],
        [right[2], down[2], forward[2]],
    ], dtype=np.float32)


def _camera_axes(yaw_deg: float, pitch_deg: float, roll_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rot = _apply_roll_to_basis(_orbit_basis_from_yp(yaw_deg, pitch_deg), roll_deg)
    right = rot @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
    down = rot @ np.array([0.0, 1.0, 0.0], dtype=np.float32)
    forward = rot @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return right.astype(np.float32), down.astype(np.float32), forward.astype(np.float32)


def _view_rotation_from_camera_state(state: dict[str, float]) -> np.ndarray:
    right, down, forward = _camera_axes(
        float(state["yaw_deg"]), float(state["pitch_deg"]), float(state["roll_deg"])
    )
    return np.stack([right, down, forward], axis=0).astype(np.float32)


def _state_to_camera(state: dict[str, float]) -> dict:
    pivot = np.array([state["pivot_x"], state["pivot_y"], state["pivot_z"]], dtype=np.float32)
    forward = _camera_axes(state["yaw_deg"], state["pitch_deg"], 0.0)[2]
    position = (pivot - forward * float(state["distance"])).astype(np.float32)
    return {"position": position, "pivot": pivot}


def _parse_interactive_state(state_text: str | None) -> dict[str, float]:
    default = {
        "pivot_x": 0.0, "pivot_y": 0.0, "pivot_z": 0.0,
        "yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0, "distance": 0.0,
        "vp_fx": 0.0, "vp_fy": 0.0, "vp_width": 0.0, "vp_height": 0.0,
    }
    if not state_text:
        return default
    try:
        data = json.loads(state_text)
        if not isinstance(data, dict):
            return default
        for k in default:
            if k in data:
                default[k] = float(data[k])
        return default
    except Exception:
        return default


def _build_framed_state(xyz: np.ndarray) -> dict[str, float]:
    if xyz.size == 0:
        return _parse_interactive_state(None)
    pivot = ((xyz.min(axis=0) + xyz.max(axis=0)) * 0.5).astype(np.float32)
    bbox_size = xyz.max(axis=0) - xyz.min(axis=0)
    radius = max(float(np.linalg.norm(bbox_size) * 0.5), 0.5)
    position = pivot + np.array([0.0, 0.0, radius * 3.0], dtype=np.float32)
    forward = _normalize(pivot - position)
    yaw = math.degrees(math.atan2(float(forward[0]), float(forward[2])))
    pitch = math.degrees(math.asin(float(np.clip(forward[1], -1.0, 1.0))))
    distance = float(np.linalg.norm(pivot - position))
    return {
        "pivot_x": float(pivot[0]), "pivot_y": float(pivot[1]), "pivot_z": float(pivot[2]),
        "yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": 0.0, "distance": max(distance, 0.1),
    }


# ── CPU rasterizer ───────────────────────────────────────────────────────────

def _render_gaussians(
    ply_path: str,
    camera_state: dict[str, float],
    width: int,
    height: int,
    gaussian_scale: float,
    max_gaussians: int,
    background: str,
    ply_data: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    data = ply_data if ply_data is not None else _load_gaussian_ply(ply_path)
    xyz = data["xyz"]
    colors = data["colors"]
    opacities = data["opacities"]
    covariances = data["covariances"]

    cam = _state_to_camera(camera_state)
    cam_pos = cam["position"]
    r_wc = _view_rotation_from_camera_state(camera_state)

    vp_fx = float(camera_state.get("vp_fx", 0))
    vp_fy = float(camera_state.get("vp_fy", 0))
    vp_width = float(camera_state.get("vp_width", 0))
    vp_height = float(camera_state.get("vp_height", 0))

    if vp_fx > 0 and vp_width > 0:
        fx = vp_fx * (width / vp_width)
        fy = vp_fy * (height / vp_height)
    else:
        f = min(width, height) * 0.9
        fx = fy = f
    cx = width * 0.5
    cy = height * 0.5

    rel = xyz - cam_pos[None, :]
    cam_xyz = rel @ r_wc.T
    z = cam_xyz[:, 2]
    visible = z > 1e-3
    if not np.any(visible):
        bg_val = _background_fill_value(background)
        return np.full((height, width, 3), bg_val, dtype=np.float32)

    cam_xyz = cam_xyz[visible]
    colors = colors[visible]
    opacities = opacities[visible]
    covariances = covariances[visible]
    r_cov = np.einsum("ij,njk,lk->nil", r_wc, covariances, r_wc).astype(np.float32)

    x = cam_xyz[:, 0]
    y = cam_xyz[:, 1]
    z = cam_xyz[:, 2]
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy

    j = np.zeros((cam_xyz.shape[0], 2, 3), dtype=np.float32)
    inv_z = 1.0 / z
    inv_z2 = inv_z * inv_z
    j[:, 0, 0] = fx * inv_z
    j[:, 0, 2] = -fx * x * inv_z2
    j[:, 1, 1] = fy * inv_z
    j[:, 1, 2] = -fy * y * inv_z2

    sigma2 = np.einsum("nij,njk,nlk->nil", j, r_cov, j).astype(np.float32)
    sigma2 *= max(float(gaussian_scale), 1e-4) ** 2

    trace = sigma2[:, 0, 0] + sigma2[:, 1, 1]
    det = sigma2[:, 0, 0] * sigma2[:, 1, 1] - sigma2[:, 0, 1] * sigma2[:, 1, 0]
    det = np.clip(det, 1e-10, None)
    disc = np.sqrt(np.clip(trace * trace - 4.0 * det, 0.0, None))
    sigma_major = np.sqrt(np.clip((trace + disc) * 0.5, 1e-10, None))
    radius = np.clip(3.0 * sigma_major, 1.0, 96.0)

    in_frame = (
        (u + radius >= 0) & (u - radius < width)
        & (v + radius >= 0) & (v - radius < height)
        & (opacities > 1e-4)
    )
    if not np.any(in_frame):
        bg_val = _background_fill_value(background)
        return np.full((height, width, 3), bg_val, dtype=np.float32)

    u = u[in_frame]; v = v[in_frame]; z = z[in_frame]
    colors = colors[in_frame]; opacities = opacities[in_frame]
    sigma2 = sigma2[in_frame]; radius = radius[in_frame]

    importance = opacities * radius * radius / np.maximum(z, 1e-4)
    if max_gaussians > 0 and importance.shape[0] > max_gaussians:
        keep = np.argpartition(importance, -max_gaussians)[-max_gaussians:]
        u = u[keep]; v = v[keep]; z = z[keep]
        colors = colors[keep]; opacities = opacities[keep]
        sigma2 = sigma2[keep]; radius = radius[keep]

    order = np.argsort(z)[::-1]
    u = u[order]; v = v[order]
    colors = colors[order]; opacities = opacities[order]
    sigma2 = sigma2[order]; radius = radius[order]

    bg_val = _background_fill_value(background)
    image = np.full((height, width, 3), bg_val, dtype=np.float32)

    if _HAS_NUMBA_SPLATS and _accumulate_splats_numba is not None:
        _accumulate_splats_numba(
            image,
            np.ascontiguousarray(u),
            np.ascontiguousarray(v),
            np.ascontiguousarray(colors),
            np.ascontiguousarray(opacities),
            np.ascontiguousarray(sigma2),
            np.ascontiguousarray(radius),
            width,
            height,
        )
    else:
        for idx in range(u.shape[0]):
            r = int(math.ceil(float(radius[idx])))
            x0 = max(0, int(math.floor(float(u[idx]) - r)))
            x1 = min(width, int(math.ceil(float(u[idx]) + r + 1)))
            y0 = max(0, int(math.floor(float(v[idx]) - r)))
            y1 = min(height, int(math.ceil(float(v[idx]) + r + 1)))
            if x0 >= x1 or y0 >= y1:
                continue

            cov = sigma2[idx]
            det2 = float(cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0])
            if det2 <= 1e-12:
                continue
            inv = np.array([[cov[1, 1], -cov[0, 1]], [-cov[1, 0], cov[0, 0]]], dtype=np.float32) / det2
            xs = np.arange(x0, x1, dtype=np.float32) - float(u[idx])
            ys = np.arange(y0, y1, dtype=np.float32) - float(v[idx])
            dx, dy = np.meshgrid(xs, ys)
            quad = inv[0, 0] * dx * dx + (inv[0, 1] + inv[1, 0]) * dx * dy + inv[1, 1] * dy * dy
            alpha = float(opacities[idx]) * np.exp(-0.5 * quad)
            alpha = np.clip(alpha, 0.0, 1.0)
            if float(alpha.max()) < 1e-4:
                continue
            patch = image[y0:y1, x0:x1]
            patch *= 1.0 - alpha[..., None]
            patch += colors[idx][None, None, :] * alpha[..., None]

    return np.clip(image, 0.0, 1.0)


def _background_fill_value(background: str) -> float:
    if background == "white":
        return 1.0
    if background == "mid-gray":
        return 0.5
    return 0.0


def _make_ui_path(ply_path: str) -> str:
    filename = os.path.basename(ply_path)
    if COMFYUI_OUTPUT_FOLDER and ply_path.startswith(COMFYUI_OUTPUT_FOLDER):
        return os.path.relpath(ply_path, COMFYUI_OUTPUT_FOLDER)
    return filename


# ── Node ─────────────────────────────────────────────────────────────────────

class SharpShotRenderNode:
    _ply_cache: dict = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_path": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Path to a Gaussian Splatting PLY file (from SHARP Gaussian Splat node)",
                }),
                "interactive_state": ("STRING", {
                    "default": "{}",
                    "multiline": True,
                }),
                "output_width": (
                    "INT",
                    {"default": 1024, "min": 64, "max": 4096, "step": 64,
                     "tooltip": "Rendered image width"},
                ),
                "output_height": (
                    "INT",
                    {"default": 1024, "min": 64, "max": 4096, "step": 64,
                     "tooltip": "Rendered image height"},
                ),
                "gaussian_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01,
                     "tooltip": "Global scale multiplier for Gaussian sizes"},
                ),
                "max_gaussians": (
                    "INT",
                    {"default": 0, "min": 0, "max": 500000, "step": 1000,
                     "tooltip": "Max splats to render (0 = unlimited). Higher values = slower but more detail"},
                ),
                "background": (
                    ["black", "mid-gray", "white"],
                    {"default": "black",
                     "tooltip": "Background color for empty areas"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render"
    CATEGORY = "3D"
    OUTPUT_NODE = True

    def render(self, ply_path: str, interactive_state: str, output_width: int, output_height: int,
               gaussian_scale: float, max_gaussians: int, background: str):
        if not ply_path:
            raise ValueError("No PLY path provided")
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"PLY file not found: {ply_path}")

        ply_data = _load_gaussian_ply(ply_path)
        xyz = ply_data["xyz"]

        camera_state = _parse_interactive_state(interactive_state)
        if abs(camera_state["distance"]) <= 1e-5:
            camera_state = _build_framed_state(xyz)

        width = max(int(output_width), 1)
        height = max(int(output_height), 1)

        print(f"[SHARP Shot] Rendering {width}x{height} from '{os.path.basename(ply_path)}' "
              f"(yaw={camera_state['yaw_deg']:.1f}, pitch={camera_state['pitch_deg']:.1f}, dist={camera_state['distance']:.2f})")

        image = _render_gaussians(
            ply_path=ply_path,
            camera_state=camera_state,
            width=width,
            height=height,
            gaussian_scale=gaussian_scale,
            max_gaussians=0 if max_gaussians <= 0 else max_gaussians,
            background=background,
            ply_data=ply_data,
        )
        image_tensor = torch.from_numpy(image.astype(np.float32)[None, ...])

        ui_data = {
            "ply_file": [_make_ui_path(ply_path)],
            "filename": [os.path.basename(ply_path)],
            "preview_camera_state": [json.dumps(camera_state)],
            "render_size": [json.dumps({"width": width, "height": height})],
            "gaussian_scale": [gaussian_scale],
            "max_gaussians": [max_gaussians],
        }
        return {"ui": ui_data, "result": (image_tensor,)}


NODE_CLASS_MAPPINGS = {"SharpShotRender": SharpShotRenderNode}
NODE_DISPLAY_NAME_MAPPINGS = {"SharpShotRender": "SHARP Shot Render"}


class SharpCameraRenderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_path": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Path to a Gaussian Splatting PLY file (from SHARP Gaussian Splat node)",
                }),
                "pivot_x": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0e6, "max": 1.0e6, "step": 0.01,
                     "tooltip": "Orbit pivot X in scene units"},
                ),
                "pivot_y": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0e6, "max": 1.0e6, "step": 0.01,
                     "tooltip": "Orbit pivot Y in scene units"},
                ),
                "pivot_z": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0e6, "max": 1.0e6, "step": 0.01,
                     "tooltip": "Orbit pivot Z in scene units"},
                ),
                "yaw_deg": (
                    "FLOAT",
                    {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.5,
                     "tooltip": "Horizontal orbit angle in degrees"},
                ),
                "pitch_deg": (
                    "FLOAT",
                    {"default": 0.0, "min": -89.0, "max": 89.0, "step": 0.5,
                     "tooltip": "Vertical orbit angle in degrees"},
                ),
                "roll_deg": (
                    "FLOAT",
                    {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5,
                     "tooltip": "Camera roll in degrees"},
                ),
                "distance": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0e6, "step": 0.01,
                     "tooltip": "Distance from pivot (0 = auto-frame)"},
                ),
                "fov_degrees": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 160.0, "step": 0.5,
                     "tooltip": "Horizontal field of view in degrees (0 = auto)"},
                ),
                "output_width": (
                    "INT",
                    {"default": 1024, "min": 64, "max": 4096, "step": 64,
                     "tooltip": "Rendered image width"},
                ),
                "output_height": (
                    "INT",
                    {"default": 1024, "min": 64, "max": 4096, "step": 64,
                     "tooltip": "Rendered image height"},
                ),
                "gaussian_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01,
                     "tooltip": "Global scale multiplier for Gaussian sizes"},
                ),
                "max_gaussians": (
                    "INT",
                    {"default": 0, "min": 0, "max": 500000, "step": 1000,
                     "tooltip": "Max splats to render (0 = unlimited)"},
                ),
                "background": (
                    ["black", "mid-gray", "white"],
                    {"default": "black",
                     "tooltip": "Background color for empty areas"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render"
    CATEGORY = "3D"

    def render(self, ply_path: str, pivot_x: float, pivot_y: float, pivot_z: float,
               yaw_deg: float, pitch_deg: float, roll_deg: float, distance: float,
               fov_degrees: float, output_width: int, output_height: int,
               gaussian_scale: float, max_gaussians: int, background: str):
        if not ply_path:
            raise ValueError("No PLY path provided")
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"PLY file not found: {ply_path}")

        ply_data = _load_gaussian_ply(ply_path)
        xyz = ply_data["xyz"]

        camera_state = {
            "pivot_x": pivot_x, "pivot_y": pivot_y, "pivot_z": pivot_z,
            "yaw_deg": yaw_deg, "pitch_deg": pitch_deg, "roll_deg": roll_deg,
            "distance": distance,
            "vp_fx": 0.0, "vp_fy": 0.0, "vp_width": 0.0, "vp_height": 0.0,
        }
        if abs(distance) <= 1e-5:
            auto = _build_framed_state(xyz)
            auto["pivot_x"] = pivot_x
            auto["pivot_y"] = pivot_y
            auto["pivot_z"] = pivot_z
            camera_state = auto

        width = max(int(output_width), 1)
        height = max(int(output_height), 1)

        if fov_degrees > 0:
            f_px = width / (2.0 * math.tan(math.radians(fov_degrees) * 0.5))
            camera_state["vp_fx"] = f_px
            camera_state["vp_fy"] = f_px
            camera_state["vp_width"] = float(width)
            camera_state["vp_height"] = float(height)

        print(f"[SHARP Camera] Rendering {width}x{height} from '{os.path.basename(ply_path)}' "
              f"(yaw={camera_state['yaw_deg']:.1f}, pitch={camera_state['pitch_deg']:.1f}, dist={camera_state['distance']:.2f})")

        image = _render_gaussians(
            ply_path=ply_path,
            camera_state=camera_state,
            width=width,
            height=height,
            gaussian_scale=gaussian_scale,
            max_gaussians=0 if max_gaussians <= 0 else int(max_gaussians),
            background=background,
            ply_data=ply_data,
        )
        image_tensor = torch.from_numpy(image.astype(np.float32)[None, ...])
        return (image_tensor,)


NODE_CLASS_MAPPINGS["SharpCameraRender"] = SharpCameraRenderNode
NODE_DISPLAY_NAME_MAPPINGS["SharpCameraRender"] = "SHARP Camera Render"

"""ScanNet++ iPhone frames for the demos: RGB, camera, ground-truth mesh, depth.

The model only ever sees the RGB frame. The mesh and the iPhone depth are for
display: depth marks which mesh vertices the input view can see, so they can be
drawn brighter. Everything is expressed in the start-camera frame, which is the
frame predictions come out in.

Frames must be extracted to <scene>/iphone/rgb/<frame>.jpg.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import lz4.block
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

# The frames shown on the project page: the viewpoint each render was taken from, and
# the intents as (output file stem, intent text). "visible" and "background" override
# the display defaults below for scenes that were rendered differently.
EXAMPLES = [
    {"name": "scene_1", "scene": "6f1848d1e3", "frame": "frame_000011",
     "camera": {"position": (-0.05139213, -0.48892426, -0.63161210),
                "wxyz": (0.98680673, -0.16101765, 0.00710049, -0.01534228),
                "fov": 1.43989663},
     "visible": (0.05, 3.0),
     "intents": [("cook", "Cook"),
                 ("leave", "Leave"),
                 ("to_livingroom", "To living room"),
                 ("to_toilet", "To toilet")]},
    {"name": "scene_2", "scene": "09c1414f1b", "frame": "frame_003943",
     "camera": {"position": (0.22049394, -1.70343947, -1.26252328),
                "wxyz": (0.95998475, -0.27775865, 0.00994147, -0.03435955),
                "fov": 0.88139127},
     "visible": (0.08, 2.0),
     "intents": [("findmybike", "Find my bike"),
                 ("lookoutside", "Look outside"),
                 ("watchTV", "Watch TV"),
                 ("whatsontable", "What's on table")]},
    {"name": "scene_3", "scene": "7eac902fd5", "frame": "frame_002039",
     "camera": {"position": (0.06697301, -0.91320278, -1.35596086),
                "wxyz": (0.98258183, -0.18579176, 0.00070404, -0.00372339),
                "fov": 0.88139127},
     "visible": (0.05, 3.0),
     "intents": [("leavebathroom", "Leave bathroom"),
                 ("lookatmirror", "Look at mirror"),
                 ("usethetowel", "Use the towel"),
                 ("usetoilet", "Use toilet")]},
    {"name": "scene_4", "scene": "7e09430da7", "frame": "frame_001245",
     "camera": {"position": (-0.55918207, -0.26082059, -1.47870710),
                "wxyz": (0.99993544, -0.00000050, -0.00000001, 0.01136302),
                "fov": 1.05592420},
     "intents": [("findbed", "Find bed"),
                 ("lookoutside", "Look outside"),
                 ("whatsondesk", "What's on desk")]},
    {"name": "scene_5", "scene": "8e6ff28354", "frame": "frame_008097",
     "camera": {"position": (0.20660932, -1.20444951, -2.10719733),
                "wxyz": (0.99202663, -0.12084590, 0.00432532, -0.03550663),
                "fov": 0.88139127},
     "background": (70, 70, 70),
     "intents": [("exit", "Exit"),
                 ("findmonitor", "Find monitor"),
                 ("lookatwhiteboard", "What is written on the whiteboard"),
                 ("whatsonsofa", "What's on sofa")]},
    {"name": "scene_6", "scene": "8a35ef3cfe", "frame": "frame_003903",
     "camera": {"position": (-0.29141129, -1.71275071, -1.48707391),
                "wxyz": (0.92403637, -0.38229323, -0.00112647, 0.00272278),
                "fov": 0.88139127},
     "intents": [("findfridge", "Look at the fridge"),
                 ("findthestove", "Find the stove"),
                 ("gotodiningtable", "Go to dining table"),
                 ("lookinsidesink", "Look inside the sink")]},
]

# Display settings of the project-page renders.
BACKGROUND = (180, 180, 180)
BRIGHTNESS = 0.55
VISIBLE = (0.03, 3.0)       # brighten vertices within this many metres of the
                            # frame's depth points, times this much
MAX_VISIBLE_RADIUS_M = 0.2
MAX_HEIGHT_M = 2.3          # hide the ceiling, so it doesn't block the view
POINT_SIZE = 0.015
START_SCALE = 0.21
SAMPLE_SCALE = 0.18
LINE_WIDTH = 8.4


def find_example(scene: str, frame: str | None = None) -> dict | None:
    """The project-page example for a scene, if any. frame=None matches any frame."""
    for example in EXAMPLES:
        if example["scene"] == scene and frame in (None, example["frame"]):
            return example
    return None


def load_depth_frame(scene_root: Path, frame_idx: int, h: int = 192,
                     w: int = 256) -> np.ndarray:
    """One frame of iphone/depth.bin, in metres, with NaN where invalid.

    The file is a stream of [u32 size][lz4 block] chunks, each holding h*w uint16
    millimetres.
    """
    with open(scene_root / "iphone" / "depth.bin", "rb") as f:
        for _ in range(frame_idx):
            f.seek(struct.unpack("<I", f.read(4))[0], 1)
        blob = f.read(struct.unpack("<I", f.read(4))[0])
    raw = lz4.block.decompress(blob, uncompressed_size=h * w * 2)
    mm = np.frombuffer(raw, dtype=np.uint16).reshape(h, w)
    metres = mm.astype(np.float32) / 1000.0
    metres[mm == 0] = np.nan
    return metres


def depth_points(scene_root: Path, frame: str, K: np.ndarray,
                 image_size: tuple[int, int]) -> np.ndarray:
    """Unproject the frame's depth, doubled in resolution, into its camera frame."""
    depth = load_depth_frame(scene_root, int(frame.split("_")[-1]))
    h, w = depth.shape[0] * 2, depth.shape[1] * 2
    filled = np.where(np.isfinite(depth), depth, 0.0)
    valid = np.isfinite(depth).astype(np.float32)
    filled = np.asarray(Image.fromarray(filled).resize((w, h), Image.BILINEAR))
    valid = np.asarray(Image.fromarray(valid).resize((w, h), Image.BILINEAR))
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(valid > 0.5, filled / np.maximum(valid, 1e-6), np.nan)

    sx, sy = w / image_size[0], h / image_size[1]
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    keep = np.isfinite(z) & (z > 0.1) & (z < 15.0)
    x = (uu - K[0, 2] * sx) / (K[0, 0] * sx) * z
    y = (vv - K[1, 2] * sy) / (K[1, 1] * sy) * z
    return np.stack([x[keep], y[keep], z[keep]], axis=-1)


def available_frames(scene_root: Path) -> list[str]:
    return sorted(p.stem for p in (scene_root / "iphone" / "rgb").glob("*.jpg"))


class ScanNetFrame:
    """One iPhone frame and its scene's mesh, in the start-camera frame."""

    def __init__(self, scene_root, frame: str):
        scene_root = Path(scene_root)
        self.image_path = scene_root / "iphone" / "rgb" / f"{frame}.jpg"
        if not self.image_path.is_file():
            raise FileNotFoundError(
                f"no {self.image_path}. Extracted frames here: "
                f"{available_frames(scene_root) or 'none'}")
        meta = json.loads((scene_root / "iphone" / "pose_intrinsic_imu.json").read_text())
        # aligned_pose is camera-to-world, OpenCV convention, in the z-up mesh frame.
        T_world_cam = np.asarray(meta[frame]["aligned_pose"], dtype=np.float64)
        K = np.asarray(meta[frame]["intrinsic"], dtype=np.float64)

        self.rgb = np.asarray(Image.open(self.image_path).convert("RGB"))
        H, W = self.rgb.shape[:2]
        self.fov = float(2.0 * np.arctan2(H / 2.0, K[1, 1]))
        self.aspect = W / H

        mesh = trimesh.load(str(scene_root / "scans" / "mesh_aligned_0.05.ply"),
                            process=False, force="mesh")
        vertices = np.asarray(mesh.vertices)
        T_cam_world = np.linalg.inv(T_world_cam)
        self.points = vertices @ T_cam_world[:3, :3].T + T_cam_world[:3, 3]
        self.faces = np.asarray(mesh.faces)
        self.heights = vertices[:, 2]
        self.colours = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.float32)
        self.up = T_cam_world[:3, 2]    # world +z, seen from the start camera

        # Distances beyond the largest radius we offer come back as inf.
        tree = cKDTree(depth_points(scene_root, frame, K, (W, H)))
        self.depth_distance, _ = tree.query(
            self.points, k=1, distance_upper_bound=MAX_VISIBLE_RADIUS_M)

    def display_colours(self, brightness: float, radius: float, boost: float) -> np.ndarray:
        colours = self.colours * brightness
        colours[self.depth_distance <= radius] *= boost
        return np.clip(colours, 0, 255).astype(np.uint8)

    def display_points(self, brightness: float, radius: float, boost: float,
                       max_height: float) -> tuple[np.ndarray, np.ndarray]:
        """Mesh vertices and their colours, with the ceiling cut off."""
        keep = self.heights <= max_height
        colours = self.display_colours(brightness, radius, boost)
        return self.points[keep].astype(np.float32), colours[keep]

    def display_mesh(self, brightness: float, radius: float, boost: float,
                     max_height: float) -> trimesh.Trimesh:
        """The mesh itself, with the ceiling cut off."""
        keep = self.heights <= max_height
        # Drop any face touching a removed vertex, then renumber the rest.
        faces = self.faces[keep[self.faces].all(axis=1)]
        new_index = np.cumsum(keep) - 1
        return trimesh.Trimesh(
            vertices=self.points[keep], faces=new_index[faces],
            vertex_colors=self.display_colours(brightness, radius, boost)[keep],
            process=False)

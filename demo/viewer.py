#!/usr/bin/env python3
"""Interactive LIME demo in viser: type an intent, see where the camera should go.

With no input given, it opens the images in examples/, with a menu to switch between
them when there are several. Each is drawn as the point cloud saved next to it,
<name>_points.npz:
  python demo/viewer.py

One image of your own. Its <name>_points.npz is drawn if there is one, otherwise you
see the cameras only:
  python demo/viewer.py --image room.jpg

On a ScanNet++ frame, predictions are drawn into the scene's ground-truth mesh:
  python demo/viewer.py --scene-root /path/to/scannetpp/data/09c1414f1b \\
      --frame frame_003943

All load weights/lime_v1 unless --checkpoint says otherwise.

Then open http://localhost:8080. The model only ever sees the RGB image and the
intent: point clouds and meshes are for display. The start camera sits at the origin,
and each sampled target camera is drawn in viridis.
"""
from __future__ import annotations

import argparse
import getpass
import json
import logging
import socket
import threading
import time
from pathlib import Path

import numpy as np
import viser
import viser.transforms as vtf
from PIL import Image, ImageOps

from common import flatten_rgba, quiet_browser_preconnects, sample_colours, set_background
from lime_camera import LIME
from lime_camera.models.lime import DTYPES
from scannetpp import (BACKGROUND, BRIGHTNESS, MAX_HEIGHT_M, MAX_VISIBLE_RADIUS_M,
                       POINT_SIZE, SAMPLE_SCALE, START_SCALE, VISIBLE, ScanNetFrame,
                       available_frames, find_example)

logger = logging.getLogger("viewer")

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE.parent / "weights" / "lime_v1"
EXAMPLES = HERE.parent / "examples"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
CLOUD_POINT_SIZE = 0.012    # metres; the saved clouds are about a pixel per point
MEAN_COLOUR = (255, 40, 40)
START_COLOUR = (0, 200, 255)
LINE_WIDTH = 4.0            # lighter than the page's renders, which are twice as wide
AXES_RATIO = 1.0            # axes length, relative to the camera's frustum scale
AXES_THICKNESS = 0.03       # axes radius, relative to their length
MAX_IMAGE_SIDE = 1024       # the start camera's image is only a texture
DEFAULT_VIEW_FOV = 0.9      # radians, for inputs without a saved viewpoint


def read_intents(path: Path) -> list[str]:
    """Read a preset list: either a JSON array or one intent per line."""
    text = path.read_text().strip()
    try:
        items = json.loads(text)
        if isinstance(items, list):
            return [str(x).strip() for x in items if str(x).strip()]
    except json.JSONDecodeError:
        pass
    return [line.strip() for line in text.splitlines() if line.strip()]


def display_image(path: Path) -> np.ndarray:
    """The image as the model sees it, upright, at texture size."""
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    return np.asarray(image)


def example_images(folder: Path) -> list[Path]:
    """The images in folder, numbered ones first, in numeric order."""
    images = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(images, key=lambda p: (not p.stem.isdigit(),
                                         int(p.stem) if p.stem.isdigit() else 0, p.stem))


def load_image_input(path: Path, fov_deg: float, intents: Path | None = None) -> dict:
    """An image and what sits next to it: preset intents and a saved point cloud.

    A point cloud comes with the vertical FOV it was estimated with, which then replaces
    fov_deg so the start camera lines up with it.
    """
    start_image = display_image(path)
    sidecar = path.with_name(f"{path.stem}_command.json")
    intents = intents or (sidecar if sidecar.is_file() else None)
    points = colours = None
    fov = float(np.deg2rad(fov_deg))
    cloud = path.with_name(f"{path.stem}_points.npz")
    if cloud.is_file():
        with np.load(cloud) as data:
            points = data["points"].astype(np.float32)
            colours = data["colours"]
            fov = float(data["fov_y"])
    return {"title": path.name, "image_path": path, "start_image": start_image,
            "fov": fov, "aspect": start_image.shape[1] / start_image.shape[0],
            "presets": read_intents(intents) if intents else [],
            "points": points, "colours": colours}


def default_view(up: np.ndarray) -> dict:
    """A viewpoint behind and above the start camera, looking past it."""
    up = up / np.linalg.norm(up)
    forward = np.array([0.0, 0.0, 1.0])
    forward = forward - (forward @ up) * up     # level with the floor
    if np.linalg.norm(forward) < 1e-3:          # looking straight up or down
        forward = np.array([0.0, -1.0, 0.0])
    forward /= np.linalg.norm(forward)
    position = up * 1.0 - forward * 1.5
    # An OpenCV camera: z along the view direction, y down, x right.
    z = forward - position
    z /= np.linalg.norm(z)
    x = np.cross(-up, z)
    x /= np.linalg.norm(x)
    rotation = np.stack([x, np.cross(z, x), z], axis=1)
    return {"position": position, "wxyz": vtf.SO3.from_matrix(rotation).wxyz,
            "fov": DEFAULT_VIEW_FOV}


def how_to_open(host: str, port: int) -> str:
    """Where to point a browser, here or from another computer. viser takes the next free
    port when the requested one is busy, so port is the one it actually got."""
    lines = ["", "LIME demo is running.",
             f"  On this machine:   open http://localhost:{port}"]
    if host in ("0.0.0.0", "::"):
        lines.append(f"  On your network:   open http://{socket.gethostname()}:{port}")
    ssh = f"ssh -N -L {port}:localhost:{port} {getpass.getuser()}@{socket.gethostname()}"
    lines += [
        f"  From another computer, over SSH: run this there, then open http://localhost:{port}",
        f"      {ssh}",
        "    (use the address you normally ssh to; VS Code Remote-SSH forwards it for you)",
        "Stop it with Ctrl+C.", ""]
    return "\n".join(lines)


def parse_args(ap: argparse.ArgumentParser):
    ap.add_argument("--checkpoint", type=Path, default=WEIGHTS,
                    help="Checkpoint folder. Default: weights/lime_v1, where "
                         "scripts/download_weights.sh puts the released one.")
    ap.add_argument("--scene-root", type=Path, default=None,
                    help="ScanNet++ scene folder, for its RGB frame and mesh.")
    ap.add_argument("--frame", default=None,
                    help="Frame in --scene-root, e.g. frame_003943. Optional for "
                         "scenes on the project page.")
    ap.add_argument("--image", type=Path, default=None,
                    help="Any image, instead of the examples.")
    ap.add_argument("--intents", type=Path, default=None,
                    help="Preset intents, one per line. Defaults to the project page's "
                         "for its scenes, or to <image>_command.json next to the image.")
    ap.add_argument("--render", choices=("mesh", "points"), default="mesh",
                    help="Draw the scene as the lit mesh (default) or as its vertices, "
                         "as on the project page, which is faster to update.")
    ap.add_argument("--fov-deg", type=float, default=60.0,
                    help="Vertical FOV to draw cameras with, for an image without a "
                         "saved point cloud. Display only.")
    ap.add_argument("--samples", type=int, default=5, help="Initial pose samples K.")
    ap.add_argument("--steps", type=int, default=10, help="Initial Euler steps.")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--dtype", choices=tuple(DTYPES), default="bf16")
    ap.add_argument("--attn", default="sdpa",
                    help="sdpa (default, no extra deps) or flash_attention_2")
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 (default) serves this machine and SSH tunnels only; "
                         "0.0.0.0 also serves your network, without a password.")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--screenshot-dir", type=Path,
                    default=HERE.parent / "vis" / "screenshots")
    args = ap.parse_args()
    if args.scene_root is not None and args.image is not None:
        ap.error("pass at most one of --scene-root and --image")
    if args.image is not None and not args.image.is_file():
        ap.error(f"no such image: {args.image}")
    return args


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    args = parse_args(ap)
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    # ---- Input ----
    # Each input: title, image_path, start_image, fov, aspect, presets, points, colours.
    frame, example = None, None
    if args.scene_root is not None:
        example = find_example(args.scene_root.name, args.frame)
        frame_name = args.frame or (example["frame"] if example else None)
        if frame_name is None:
            ap.error(f"pass --frame; extracted frames: {available_frames(args.scene_root)}")
        logger.info("loading %s %s", args.scene_root.name, frame_name)
        try:
            frame = ScanNetFrame(args.scene_root, frame_name)
        except FileNotFoundError as e:
            ap.error(str(e))
        if args.intents is not None:
            presets = read_intents(args.intents)
        else:
            presets = [intent for _, intent in example["intents"]] if example else []
        title = f"{args.scene_root.name} {frame_name}"
        inputs = {title: {"title": title, "image_path": frame.image_path,
                          "start_image": frame.rgb, "fov": frame.fov, "aspect": frame.aspect,
                          "presets": presets, "points": None, "colours": None}}
    else:
        paths = [args.image] if args.image is not None else example_images(EXAMPLES)
        if not paths:
            ap.error(f"no images in {EXAMPLES}; pass --image")
        inputs = {path.name: load_image_input(path, args.fov_deg, args.intents)
                  for path in paths}
    current = {"input": next(iter(inputs.values()))}

    # ---- Model ----
    try:
        model = LIME.from_pretrained(args.checkpoint, args.dtype, args.attn)
    except FileNotFoundError as e:
        ap.error(str(e))

    # ---- Scene ----
    quiet_browser_preconnects()
    server = viser.ViserServer(host=args.host, port=args.port)
    server.gui.configure_theme(control_width="large")
    server.scene.world_axes.visible = False
    background = (example or {}).get("background", BACKGROUND)
    set_background(server, background)
    # Orbit around gravity when we know it; otherwise the start camera's up (OpenCV).
    up = frame.up if frame is not None else np.array([0.0, -1.0, 0.0])
    view = example["camera"] if example else default_view(up)
    if frame is not None:
        # The environment map lighting the mesh is z-up, but this frame's up is gravity.
        axis = np.eye(3)[np.argmin(np.abs(up))]
        x = axis - (axis @ up) * up
        x /= np.linalg.norm(x)
        server.scene.configure_environment_map(environment_wxyz=vtf.SO3.from_matrix(
            np.stack([x, np.cross(up, x), up], axis=1)).wxyz)
    # The browser's camera takes its roll from the up direction, so it must be square
    # with the view's rotation. Then the order these updates arrive in can't change it.
    view_right = vtf.SO3(np.asarray(view["wxyz"])).as_matrix()[:, 0]
    orbit_up = up - (up @ view_right) * view_right

    def apply_view(client):
        # Don't use scene.set_up_direction here. It can reach the browser after the
        # camera below and rotate it around the origin, typically when a heavy mesh
        # delays the browser.
        client.camera.up_direction = orbit_up
        client.camera.fov = view["fov"]
        client.camera.position = np.asarray(view["position"])
        client.camera.wxyz = np.asarray(view["wxyz"])

    server.on_client_connect(apply_view)

    # ---- GUI ----
    status = server.gui.add_markdown("")

    def set_status(message: str):
        status.content = f"**{current['input']['title']}**<br>{message}"

    set_status("Idle")
    example_in = None
    if len(inputs) > 1:
        example_in = server.gui.add_dropdown("Example", options=tuple(inputs),
                                             initial_value=current["input"]["title"])
    with server.gui.add_folder("Run"):
        intent_in = server.gui.add_text("Intent", initial_value="")
        run_button = server.gui.add_button("Run", color="green")
        samples_in = server.gui.add_slider("Samples (K)", min=1, max=32, step=1,
                                           initial_value=args.samples)
        steps_in = server.gui.add_slider("Euler steps", min=1, max=50, step=1,
                                         initial_value=args.steps)
        seed_in = server.gui.add_number("Seed (-1: random)", initial_value=-1,
                                        min=-1, step=1)
        keep_in = server.gui.add_checkbox("Keep showing earlier results",
                                          initial_value=False)
    presets_folder = server.gui.add_folder("Preset intents")
    preset_buttons = []
    results_folder = server.gui.add_folder("Results")
    with results_folder:
        clear_button = server.gui.add_button("Clear all")

    with server.gui.add_folder("Cameras", expand_by_default=False):
        start_scale_in = server.gui.add_slider("Start size", min=0.02, max=1.0, step=0.01,
                                               initial_value=START_SCALE)
        sample_scale_in = server.gui.add_slider("Sample size", min=0.02, max=1.0, step=0.01,
                                                initial_value=SAMPLE_SCALE)
        line_width_in = server.gui.add_slider("Line width", min=0.5, max=10.0, step=0.1,
                                              initial_value=LINE_WIDTH)
        show_mean_in = server.gui.add_checkbox("Show mean pose", initial_value=False)
        start_axes_in = server.gui.add_checkbox("Start camera axes", initial_value=True)
        predicted_axes_in = server.gui.add_checkbox("Predicted camera axes",
                                                    initial_value=False)

    if frame is not None:
        radius0, boost0 = (example or {}).get("visible", VISIBLE)
        with server.gui.add_folder("Scene", expand_by_default=False):
            render_in = server.gui.add_dropdown("Draw as", options=("points", "mesh"),
                                                initial_value=args.render)
            brightness_in = server.gui.add_slider("Brightness", min=0.1, max=3.0, step=0.05,
                                                  initial_value=BRIGHTNESS)
            point_size_in = server.gui.add_slider("Point size", min=0.002, max=0.05,
                                                  step=0.001, initial_value=POINT_SIZE)
            max_height_in = server.gui.add_slider("Ceiling cut (m)", min=0.0, max=4.0,
                                                  step=0.05, initial_value=MAX_HEIGHT_M)
            radius_in = server.gui.add_slider("Visible radius (m)", min=0.0,
                                              max=MAX_VISIBLE_RADIUS_M, step=0.005,
                                              initial_value=radius0)
            boost_in = server.gui.add_slider("Visible boost", min=1.0, max=4.0, step=0.05,
                                             initial_value=boost0)
    elif any(item["points"] is not None for item in inputs.values()):
        with server.gui.add_folder("Scene", expand_by_default=False):
            point_size_in = server.gui.add_slider("Point size", min=0.002, max=0.05,
                                                  step=0.001, initial_value=CLOUD_POINT_SIZE)
    with server.gui.add_folder("View", expand_by_default=False):
        background_in = server.gui.add_slider("Background", min=0, max=255, step=1,
                                              initial_value=int(background[0]))
        view_fov_in = server.gui.add_slider("Field of view (deg)", min=10.0, max=120.0,
                                            step=0.5,
                                            initial_value=float(np.degrees(view["fov"])))
        screenshot_height_in = server.gui.add_number("Screenshot height", initial_value=1440,
                                                     min=240, max=4320, step=1)
        screenshot_button = server.gui.add_button("Save screenshot")

    # ---- Cameras ----
    def add_camera(name: str, T: np.ndarray, colour, scale: float, show_axes: bool, **kwargs):
        """A frustum with its axes; returns (frustum, axes) scene handles."""
        frustum = server.scene.add_camera_frustum(
            name, fov=current["input"]["fov"], aspect=current["input"]["aspect"],
            scale=scale, line_width=float(line_width_in.value), color=colour,
            wxyz=vtf.SO3.from_matrix(T[:3, :3]).wxyz, position=T[:3, 3], **kwargs)
        axes = server.scene.add_frame(
            f"{name}/axes", axes_length=scale * AXES_RATIO,
            axes_radius=scale * AXES_RATIO * AXES_THICKNESS, visible=show_axes)
        return frustum, axes

    start = {"camera": None}

    def draw_start_camera():
        if start["camera"] is not None:
            for handle in reversed(start["camera"]):
                handle.remove()
        start["camera"] = add_camera(
            "/start", np.eye(4), START_COLOUR, float(start_scale_in.value),
            bool(start_axes_in.value), image=current["input"]["start_image"])

    draw_start_camera()

    # ---- Scene geometry ----
    scene_node = {"handle": None}

    def draw_scene(_event=None):
        if scene_node["handle"] is not None:
            scene_node["handle"].remove()
            scene_node["handle"] = None
        if frame is not None:
            settings = (float(brightness_in.value), float(radius_in.value),
                        float(boost_in.value), float(max_height_in.value))
            if render_in.value == "points":
                points, colours = frame.display_points(*settings)
                scene_node["handle"] = server.scene.add_point_cloud(
                    "/scene", points=points, colors=colours,
                    point_size=float(point_size_in.value))
            else:
                scene_node["handle"] = server.scene.add_mesh_trimesh(
                    "/scene", frame.display_mesh(*settings))
        elif current["input"]["points"] is not None:
            scene_node["handle"] = server.scene.add_point_cloud(
                "/scene", points=current["input"]["points"],
                colors=current["input"]["colours"], point_size=float(point_size_in.value))

    def update_point_size(_event):
        if scene_node["handle"] is not None and (frame is None or render_in.value == "points"):
            scene_node["handle"].point_size = float(point_size_in.value)

    if frame is not None:
        for control in (render_in, brightness_in, max_height_in, radius_in, boost_in):
            control.on_update(draw_scene)
    if frame is not None or any(item["points"] is not None for item in inputs.values()):
        point_size_in.on_update(update_point_size)
    draw_scene()

    # ---- Predictions ----
    results = []            # each: samples, mean, folder, show checkbox
    run_lock = threading.Lock()
    counter = {"n": 0}

    def when_idle(action):
        """Run a change to the results, unless a prediction is running."""
        if not run_lock.acquire(blocking=False):
            set_status("Busy: wait for the current run to finish")
            return
        try:
            action()
        finally:
            run_lock.release()

    def show_result(result):
        shown = bool(result["show"].value)
        for frustum, _ in result["samples"]:
            frustum.visible = shown
        result["mean"][0].visible = shown and bool(show_mean_in.value)

    def remove_result(result):
        for frustum, axes in result["samples"] + [result["mean"]]:
            axes.remove()
            frustum.remove()
        result["folder"].remove()
        results.remove(result)

    def add_result(prediction):
        index = counter["n"]
        counter["n"] += 1
        scale = float(sample_scale_in.value)
        colours = sample_colours(len(prediction.pose9_samples))
        show_axes = bool(predicted_axes_in.value)
        samples = [add_camera(f"/pred/{index}/sample_{j}", T, colour, scale, show_axes)
                   for j, (T, colour) in enumerate(zip(prediction.T_samples, colours))]
        mean = add_camera(f"/pred/{index}/mean", prediction.T_mean, MEAN_COLOUR, scale,
                          show_axes, visible=bool(show_mean_in.value))

        move = np.linalg.norm(prediction.pose9_mean[:3])
        spread = np.linalg.norm(prediction.pose9_std[:3])
        with results_folder:
            folder = server.gui.add_folder(f"{index}: {prediction.intent[:40]}")
        with folder:
            server.gui.add_markdown(
                f"{prediction.observation_gain}<br>"
                f"Mean move {move:.2f} m, spread {spread:.2f} m")
            show = server.gui.add_checkbox("Show", initial_value=True)
            remove = server.gui.add_button("Remove")
        result = {"samples": samples, "mean": mean, "folder": folder, "show": show}
        show.on_update(lambda _e: show_result(result))
        remove.on_click(lambda _e: when_idle(lambda: remove_result(result)))

        if not keep_in.value:
            for earlier in results:
                earlier["show"].value = False
                show_result(earlier)
        results.append(result)

    def run(intent: str):
        intent = intent.strip()
        if not intent:
            set_status("Type an intent first")
            return
        if not run_lock.acquire(blocking=False):
            set_status("Busy: wait for the current run to finish")
            return
        try:
            k, n_steps, seed = (int(samples_in.value), int(steps_in.value),
                                int(seed_in.value))
            set_status(f"Running: {intent[:60]}")
            run_button.disabled = True
            if example_in is not None:
                example_in.disabled = True
            prediction = model.predict(
                current["input"]["image_path"], intent, n_samples=k, n_steps=n_steps,
                seed=None if seed < 0 else seed, max_new_tokens=args.max_new_tokens)
            add_result(prediction)
            total = sum(prediction.timings.values())
            set_status(f"Done in {total:.2f} s (K={k}, {n_steps} steps)")
            logger.info("%r: %.2f s, %s", intent, total,
                        {name: round(t, 3) for name, t in prediction.timings.items()})
        except Exception as e:
            logger.exception("prediction failed")
            set_status(f"Error: {e}")
        finally:
            run_button.disabled = False
            if example_in is not None:
                example_in.disabled = False
            run_lock.release()

    def launch(intent: str):
        threading.Thread(target=run, args=(intent,), daemon=True).start()

    run_button.on_click(lambda _e: launch(intent_in.value))

    def fill_presets():
        for button in preset_buttons:
            button.remove()
        preset_buttons.clear()
        with presets_folder:
            for text in current["input"]["presets"]:
                button = server.gui.add_button(text[:60])
                button.on_click(lambda _e, t=text: launch(t))
                preset_buttons.append(button)
        presets_folder.visible = bool(preset_buttons)

    fill_presets()

    def clear_results():
        for result in list(results):
            remove_result(result)
        set_status("Cleared")

    clear_button.on_click(lambda _e: when_idle(clear_results))

    # ---- Switching examples ----
    def switch_to(name: str):
        current["input"] = inputs[name]
        clear_results()
        draw_start_camera()
        draw_scene()
        fill_presets()
        intent_in.value = ""
        for client in server.get_clients().values():
            apply_view(client)
        set_status("Idle")

    def on_example(_event):
        # Setting a value from Python calls this too, including the reset just below.
        if example_in.value == current["input"]["title"]:
            return
        if not run_lock.acquire(blocking=False):
            # Runs disable the menu, so this only happens in a race with a run starting.
            example_in.value = current["input"]["title"]
            set_status("Busy: wait for the current run to finish")
            return
        try:
            switch_to(example_in.value)
        finally:
            run_lock.release()

    if example_in is not None:
        example_in.on_update(on_example)

    # ---- Appearance ----
    def update_cameras(_event):
        width = float(line_width_in.value)
        start_scale, sample = float(start_scale_in.value), float(sample_scale_in.value)
        pairs = [(start["camera"], start_scale)]
        for result in results:
            pairs += [(camera, sample) for camera in result["samples"] + [result["mean"]]]
        for (frustum, axes), scale in pairs:
            frustum.scale = scale
            frustum.line_width = width
            axes.axes_length = scale * AXES_RATIO
            axes.axes_radius = scale * AXES_RATIO * AXES_THICKNESS

    def update_visibility(_event):
        start["camera"][1].visible = bool(start_axes_in.value)
        for result in results:
            for _, axes in result["samples"] + [result["mean"]]:
                axes.visible = bool(predicted_axes_in.value)
            show_result(result)

    for control in (start_scale_in, sample_scale_in, line_width_in):
        control.on_update(update_cameras)
    for control in (show_mean_in, start_axes_in, predicted_axes_in):
        control.on_update(update_visibility)

    # ---- View ----
    def current_background():
        value = int(background_in.value)
        return (value, value, value)

    background_in.on_update(lambda _e: set_background(server, current_background()))

    @view_fov_in.on_update
    def _set_view_fov(_event):
        for client in server.get_clients().values():
            client.camera.fov = float(np.radians(view_fov_in.value))

    @screenshot_button.on_click
    def _screenshot(event):
        client = event.client or next(iter(server.get_clients().values()), None)
        if client is None:
            return
        height = int(screenshot_height_in.value)
        width = int(round(height * client.camera.aspect))
        rgba = client.get_render(height=height, width=width, transport_format="png")
        args.screenshot_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(current["input"]["title"]).stem.replace(" ", "_")
        n = 0
        while (args.screenshot_dir / f"{stem}_{n}.png").exists():
            n += 1
        out = args.screenshot_dir / f"{stem}_{n}.png"
        Image.fromarray(flatten_rgba(rgba, current_background())).save(out)
        out.with_suffix(".json").write_text(json.dumps({
            "position": client.camera.position.tolist(),
            "wxyz": client.camera.wxyz.tolist(),
            "fov": float(client.camera.fov),
        }, indent=1))
        set_status(f"Saved {out}")

    print(how_to_open(args.host, server.get_port()), flush=True)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()

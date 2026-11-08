"""Étape 6 : viewer web interactif (viser).

Les gaussiennes sont rendues directement dans le navigateur (WebGL).
On tape une requête, et les gaussiennes pertinentes passent en rouge.
"""
import time
from pathlib import Path

import numpy as np
import torch
import viser

from .gaussians import load_splats, sh_to_rgb
from .query import SceneQuery, clean_query, highlight
from .scene import load_scene


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = torch.nn.functional.normalize(q, dim=-1).unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)


def run_viewer(scene_dir: Path, port: int = 8080, device: str = "cuda"):
    splats, _, up = load_splats(scene_dir / "gs" / "splats.pt", device)
    # On n'envoie pas au navigateur les gaussiennes quasi transparentes.
    opacity = torch.sigmoid(splats["opacities"])
    keep = opacity > 0.02
    R = quat_to_rotmat(splats["quats"][keep])
    s2 = torch.exp(splats["scales"][keep]) ** 2
    covariances = (R * s2[:, None, :]) @ R.transpose(1, 2)
    rgb = sh_to_rgb(splats["sh0"][keep, 0])
    base_opacity = opacity[keep]
    print(f"{keep.sum().item()} gaussiennes affichées sur {len(keep)}")

    query = SceneQuery(scene_dir, device)
    frames, xyz, _ = load_scene(scene_dir)
    cam0 = frames[0].cam_center
    forward = frames[0].viewmat[2, :3]  # axe z de la caméra exprimé dans le monde
    look_at = cam0 + forward * np.linalg.norm(np.median(xyz, 0) - cam0)

    server = viser.ViserServer(port=port)
    server.scene.set_up_direction(tuple(float(v) for v in up))
    splat = server.scene.add_gaussian_splats(
        "/scene",
        centers=splats["means"][keep].cpu().numpy(),
        covariances=covariances.cpu().numpy(),
        rgbs=rgb.cpu().numpy(),
        opacities=base_opacity[:, None].cpu().numpy(),
    )

    server.gui.add_markdown("### Scène 3D interrogeable\nTape un objet, en français ou en anglais.")
    text = server.gui.add_text("Requête", "Où est le renard ?")
    level = server.gui.add_dropdown(
        "Niveau", ("auto", "small", "large"), initial_value="auto",
        hint="small = parties d'objets (oreille), large = objets entiers (renard)",
    )
    ensemble = server.gui.add_checkbox(
        "Objet entier", False, hint="Moyenne « {} » / « une photo de {} » : mieux pour les objets, moins bien pour les parties",
    )
    search = server.gui.add_button("Chercher")
    threshold = server.gui.add_slider("Seuil", 0.0, 1.0, 0.01, 0.5)
    isolate = server.gui.add_checkbox("Isoler", False)
    reset = server.gui.add_button("Réinitialiser")
    status = server.gui.add_markdown("")
    state = {"scores": None, "level": ""}

    def refresh():
        if state["scores"] is None:
            colors, opac = rgb, base_opacity
        else:
            colors, hit = highlight(rgb, state["scores"], threshold.value)
            opac = base_opacity * hit if isolate.value else base_opacity
            status.content = f"**{hit.sum().item()}** gaussiennes au-dessus du seuil"
        splat.rgbs = colors.cpu().numpy()
        splat.opacities = opac[:, None].cpu().numpy()

    @search.on_click
    def _(_):
        t0 = time.time()
        scores, state["level"] = query.scores(
            text.value, None if level.value == "auto" else level.value, ensemble.value
        )
        state["scores"] = scores[keep]
        refresh()
        status.content += (
            f" · « {clean_query(text.value)} » (niveau {state['level']}) en {time.time() - t0:.2f} s"
        )

    @reset.on_click
    def _(_):
        state["scores"] = None
        status.content = ""
        refresh()

    threshold.on_update(lambda _: refresh())
    isolate.on_update(lambda _: refresh())

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        # Point de vue de départ : celui de la première caméra.
        client.camera.position = tuple(float(v) for v in cam0)
        client.camera.look_at = tuple(float(v) for v in look_at)
        client.camera.up_direction = tuple(float(v) for v in up)

    print(f"Viewer prêt : http://localhost:{port}")
    while True:
        time.sleep(1)

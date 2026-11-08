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


def pick_gaussian(origin: torch.Tensor, direction: torch.Tensor, centers: torch.Tensor, radius: torch.Tensor,
                  opacity: torch.Tensor) -> int | None:
    """Première gaussienne assez opaque traversée par le rayon (approximation : sphère de rayon 2·s_max)."""
    d = direction / direction.norm()
    v = centers - origin
    t = v @ d                                   # distance le long du rayon
    perp = (v - t[:, None] * d).norm(dim=1)     # distance au rayon
    hit = (t > 0) & (perp < 2 * radius) & (opacity > 0.3)
    if not hit.any():
        return None
    t = torch.where(hit, t, torch.full_like(t, float("inf")))
    return int(t.argmin())


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
    keep_idx = torch.nonzero(keep)[:, 0]
    centers_t = splats["means"][keep]
    radius_t = torch.exp(splats["scales"][keep]).max(dim=1).values
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

    server.gui.add_markdown("### Scène 3D interrogeable\nTape un objet, en français ou en anglais, "
                            "ou clique sur un objet pour trouver ce qui lui ressemble.")
    text = server.gui.add_text("Requête", "", hint="ex. : où est la chaise rouge ?")
    level = server.gui.add_dropdown(
        "Niveau", ("auto", "small", "large"), initial_value="auto",
        hint="small = parties d'objets (oreille), large = objets entiers (renard)",
    )
    ensemble = server.gui.add_checkbox(
        "Objet entier", False, hint="Moyenne « {} » / « une photo de {} » : mieux pour les objets, moins bien pour les parties",
    )
    search = server.gui.add_button("Chercher")
    pick = server.gui.add_button("Sélectionner en cliquant", hint="puis clique sur un objet dans la scène")
    threshold = server.gui.add_slider("Seuil", 0.0, 1.0, 0.01, 0.5)
    isolate = server.gui.add_checkbox("Isoler", False)
    reset = server.gui.add_button("Réinitialiser")
    status = server.gui.add_markdown("")
    state = {"scores": None, "label": ""}

    def refresh():
        if state["scores"] is None:
            colors, opac = rgb, base_opacity
            status.content = state["label"]
        else:
            colors, hit = highlight(rgb, state["scores"], threshold.value)
            opac = base_opacity * hit if isolate.value else base_opacity
            status.content = f"**{hit.sum().item()}** gaussiennes au-dessus du seuil · {state['label']}"
        splat.rgbs = colors.cpu().numpy()
        splat.opacities = opac[:, None].cpu().numpy()

    @search.on_click
    def _(_):
        t0 = time.time()
        scores, lvl = query.scores(text.value, None if level.value == "auto" else level.value, ensemble.value)
        state["scores"] = scores[keep]
        state["label"] = f"« {clean_query(text.value)} » (niveau {lvl}) en {time.time() - t0:.2f} s"
        refresh()

    @pick.on_click
    def _(_):
        state["label"] = "Clique sur un objet dans la scène…"
        status.content = state["label"]

        @server.scene.on_pointer_event(event_type="click")
        def _(event: viser.ScenePointerEvent):
            server.scene.remove_pointer_callback()  # un seul clic
            o = torch.tensor(event.ray_origin, dtype=torch.float32, device=device)
            d = torch.tensor(event.ray_direction, dtype=torch.float32, device=device)
            j = pick_gaussian(o, d, centers_t, radius_t, base_opacity)
            if j is None:
                state["label"] = "Rien sous le curseur : réessaie sur un objet."
                refresh()
                return
            lvl = "large" if level.value == "auto" else level.value
            state["scores"] = query.similar(int(keep_idx[j]), lvl)[keep]
            state["label"] = f"objets semblables au point cliqué (niveau {lvl})"
            refresh()

    @reset.on_click
    def _(_):
        state["scores"] = None
        state["label"] = ""
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

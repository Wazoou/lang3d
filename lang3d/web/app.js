// Démo statique : scène .splat + scores de requêtes précalculés (un octet par gaussienne et par requête).
import * as SPLAT from "https://cdn.jsdelivr.net/npm/gsplat@1.2.9/dist/index.es.js";

const canvas = document.getElementById("canvas");
const renderer = new SPLAT.WebGLRenderer(canvas);
const scene = new SPLAT.Scene();
const camera = new SPLAT.Camera();
renderer.backgroundColor = new SPLAT.Color32(13, 16, 24, 255);

const $ = (id) => document.getElementById(id);
const meta = await (await fetch("queries.json")).json();
const scores = new Uint8Array(await (await fetch("queries.bin")).arrayBuffer());
const splat = await SPLAT.Loader.LoadAsync("scene.splat", scene, (p) => {
  $("loading").textContent = `Chargement de la scène… ${Math.round(p * 100)} %`;
});
$("loading").remove();

const n = splat.data.vertexCount;
const original = splat.data.colors.slice();  // RGBA d'origine, pour revenir en arrière
const state = { query: -1, threshold: meta.threshold, isolate: false };

const [cx, cy, cz] = meta.camera.position;
camera.position = new SPLAT.Vector3(cx, cy, cz);
const controls = new SPLAT.OrbitControls(camera, canvas);
controls.setCameraTarget(new SPLAT.Vector3(...meta.camera.target));

// Même règle que lang3d.query.highlight : au-dessus du seuil, mélange vers le rouge ; sinon gris assombri.
function recolor() {
  const colors = splat.data.colors;
  if (state.query < 0) {
    colors.set(original);
  } else {
    const off = state.query * n, t = state.threshold;
    let hits = 0;
    for (let i = 0; i < n; i++) {
      const s = scores[off + i] / 255, k = 4 * i;
      if (s > t) {
        const a = Math.min(1, 0.5 + 0.5 * (s - t) / Math.max(1 - t, 1e-6));
        colors[k] = original[k] * (1 - a) + 255 * a;
        colors[k + 1] = original[k + 1] * (1 - a) + 26 * a;
        colors[k + 2] = original[k + 2] * (1 - a) + 26 * a;
        colors[k + 3] = original[k + 3];
        hits++;
      } else {
        const g = (original[k] + original[k + 1] + original[k + 2]) / 6;
        colors[k] = colors[k + 1] = colors[k + 2] = g;
        colors[k + 3] = state.isolate ? 0 : original[k + 3];
      }
    }
    $("status").textContent = `${hits.toLocaleString("fr-FR")} gaussiennes surlignées · niveau ${meta.queries[state.query].level}`;
  }
  splat.data.changed = true;
}

// Les buffers partent vers un worker pendant la mise à jour : on attend qu'ils soient revenus.
let pending = false;
function requestRecolor() {
  if (pending) return;
  pending = true;
  const tryNow = () => (splat.data.detached ? requestAnimationFrame(tryNow) : (recolor(), (pending = false)));
  tryNow();
}

meta.queries.forEach((q, i) => {
  const b = document.createElement("button");
  b.className = "chip";
  b.textContent = q.label;
  b.setAttribute("aria-pressed", "false");
  b.onclick = () => {
    state.query = state.query === i ? -1 : i;
    document.querySelectorAll(".chip").forEach((c, j) => c.setAttribute("aria-pressed", String(j === state.query)));
    if (state.query < 0) $("status").textContent = "";
    requestRecolor();
  };
  $("chips").appendChild(b);
});

$("threshold").value = state.threshold;
$("thv").textContent = state.threshold.toFixed(2);
$("threshold").oninput = (e) => {
  state.threshold = parseFloat(e.target.value);
  $("thv").textContent = state.threshold.toFixed(2);
  if (state.query >= 0) requestRecolor();
};
$("isolate").onchange = (e) => {
  state.isolate = e.target.checked;
  if (state.query >= 0) requestRecolor();
};

const onResize = () => renderer.setSize(window.innerWidth, window.innerHeight);
window.addEventListener("resize", onResize);
onResize();

window.lang3d = { renderer, scene, camera, controls, splat };  // pour le débogage depuis la console

const frame = () => {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
};
requestAnimationFrame(frame);

// Démo statique : scène .splat + scores de requêtes précalculés (un octet par gaussienne et par requête)
// + un vecteur de sens compressé par gaussienne pour la sélection au clic.
import * as SPLAT from "https://cdn.jsdelivr.net/npm/gsplat@1.2.9/dist/index.es.js";

const canvas = document.getElementById("canvas");
const renderer = new SPLAT.WebGLRenderer(canvas);
const scene = new SPLAT.Scene();
const camera = new SPLAT.Camera();
renderer.backgroundColor = new SPLAT.Color32(13, 16, 24, 255);

const $ = (id) => document.getElementById(id);
const meta = await (await fetch("queries.json")).json();
const queryScores = new Uint8Array(await (await fetch("queries.bin")).arrayBuffer());
const features = new Int8Array(await (await fetch("features.bin")).arrayBuffer());
const splat = await SPLAT.Loader.LoadAsync("scene.splat", scene, (p) => {
  $("loading").textContent = `Chargement de la scène… ${Math.round(p * 100)} %`;
});
$("loading").remove();

const n = splat.data.vertexCount, dim = meta.feature_dim;
// Copies : les tableaux de gsplat.js partent vers un worker pendant les mises à jour.
const original = splat.data.colors.slice();
const positions = splat.data.positions.slice();
const scales = splat.data.scales.slice();
const state = { scores: null, label: "", threshold: meta.threshold, isolate: false, heat: false };

// Point de vue initial = première caméra de la vidéo. OrbitControls place la caméra en
// cible + r · (sin α cos β, −sin β, −cos α cos β) : on inverse cette formule.
const [px, py, pz] = meta.camera.position, [tx, ty, tz] = meta.camera.target;
const ox = px - tx, oy = py - ty, oz = pz - tz;
const radius = 1.8 * Math.hypot(ox, oy, oz);  // recul : le champ de vision est plus étroit que celui du téléphone
const controls = new SPLAT.OrbitControls(camera, canvas, Math.atan2(ox, -oz), Math.asin(-oy / Math.hypot(ox, oy, oz)), radius,
                                         true, new SPLAT.Vector3(tx, ty, tz));

// Même règle que lang3d.query.highlight : au-dessus du seuil, mélange vers le rouge ; sinon gris assombri.
function recolor() {
  const colors = splat.data.colors;
  if (!state.scores) {
    colors.set(original);
    $("status").textContent = state.label;
  } else {
    const t = state.threshold, s = state.scores;
    let hits = 0;
    for (let i = 0; i < n; i++) {
      const v = s[i] / 255, k = 4 * i;
      if (state.heat) {
        // Carte de chaleur (comme lang3d.query.heatmap) : du gris au rouge selon le score.
        const a = Math.pow(v, 0.7), g = (original[k] + original[k + 1] + original[k + 2]) / 6;
        colors[k] = g * (1 - a) + 255 * a;
        colors[k + 1] = g * (1 - a) + 26 * a;
        colors[k + 2] = g * (1 - a) + 26 * a;
        colors[k + 3] = state.isolate && v <= t ? 0 : original[k + 3];
        if (v > t) hits++;
      } else if (v > t) {
        const a = Math.min(1, 0.5 + 0.5 * (v - t) / Math.max(1 - t, 1e-6));
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
    $("status").textContent = `${hits.toLocaleString("fr-FR")} gaussiennes surlignées · ${state.label}`;
  }
  // gsplat.js ne relit les données d'un objet qu'après un événement « objectChanged ».
  splat.data.changed = true;
  splat.raiseChangeEvent();
}

// Les buffers partent vers un worker pendant la mise à jour : on attend qu'ils soient revenus.
let pending = false;
function requestRecolor() {
  if (pending) return;
  pending = true;
  const tryNow = () => (splat.data.detached ? requestAnimationFrame(tryNow) : (recolor(), (pending = false)));
  tryNow();
}

function setChip(i) {
  document.querySelectorAll(".chip").forEach((c, j) => c.setAttribute("aria-pressed", String(j === i)));
}

meta.queries.forEach((q, i) => {
  const b = document.createElement("button");
  b.className = "chip";
  b.textContent = q.label;
  b.setAttribute("aria-pressed", "false");
  b.onclick = () => {
    const active = b.getAttribute("aria-pressed") === "true";
    setChip(active ? -1 : i);
    state.scores = active ? null : queryScores.subarray(i * n, (i + 1) * n);
    state.label = active ? "" : `« ${q.label} » · niveau ${q.level}`;
    requestRecolor();
  };
  $("chips").appendChild(b);
});

// --- Sélection au clic : première gaussienne opaque sur le rayon, puis ressemblance des vecteurs de sens.
const norms = new Float32Array(n);
for (let i = 0; i < n; i++) {
  let s = 0;
  for (let k = 0; k < dim; k++) s += features[i * dim + k] ** 2;
  norms[i] = Math.sqrt(s);
}

function pick(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const x = (2 * (clientX - rect.left)) / rect.width - 1, y = 1 - (2 * (clientY - rect.top)) / rect.height;
  const d = camera.screenPointToRay(x, y), o = camera.position;
  let best = -1, bestT = Infinity;
  for (let i = 0; i < n; i++) {
    if (original[4 * i + 3] < 77 || norms[i] === 0) continue;  // opacité < 0,3 ou sans vecteur de sens
    const vx = positions[3 * i] - o.x, vy = positions[3 * i + 1] - o.y, vz = positions[3 * i + 2] - o.z;
    const t = vx * d.x + vy * d.y + vz * d.z;
    if (t <= 0 || t >= bestT) continue;
    const r = 2 * Math.max(scales[3 * i], scales[3 * i + 1], scales[3 * i + 2]);
    if (vx * vx + vy * vy + vz * vz - t * t < r * r) { best = i; bestT = t; }
  }
  return best;
}

function similarTo(j) {
  const cos = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    if (norms[i] === 0) { cos[i] = -1; continue; }
    let s = 0;
    for (let k = 0; k < dim; k++) s += features[i * dim + k] * features[j * dim + k];
    cos[i] = s / (norms[i] * norms[j]);
  }
  // Même normalisation que SceneQuery.similar : 0 = médiane de la scène, 1 = identique.
  const sample = Array.from({ length: 20000 }, () => cos[(Math.random() * n) | 0]).filter((v) => v > -1).sort((a, b) => a - b);
  const med = sample[sample.length >> 1];
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.round(255 * Math.min(1, Math.max(0, (cos[i] - med) / (1 - med))));
  return out;
}

let down = null;
canvas.addEventListener("mousedown", (e) => (down = e.button === 0 ? [e.clientX, e.clientY] : null));
canvas.addEventListener("mouseup", (e) => {
  if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;  // c'était un glisser
  const j = pick(e.clientX, e.clientY);
  if (j < 0) return;
  setChip(-1);
  state.scores = similarTo(j);
  state.label = "objets semblables au point cliqué";
  requestRecolor();
});

$("threshold").value = state.threshold;
$("thv").textContent = state.threshold.toFixed(2);
$("threshold").oninput = (e) => {
  state.threshold = parseFloat(e.target.value);
  $("thv").textContent = state.threshold.toFixed(2);
  if (state.scores) requestRecolor();
};
$("isolate").onchange = (e) => {
  state.isolate = e.target.checked;
  if (state.scores) requestRecolor();
};
$("heat").onchange = (e) => {
  state.heat = e.target.checked;
  if (state.scores) requestRecolor();
};

const onResize = () => renderer.setSize(window.innerWidth, window.innerHeight);
window.addEventListener("resize", onResize);
onResize();

window.lang3d = { renderer, scene, camera, controls, splat, pick, similarTo };  // pour le débogage depuis la console

const frame = () => {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
};
requestAnimationFrame(frame);

// just-matching demo: pick images and see their matches, with either matcher.
// - LightGlue (default): ALIKED keypoints and descriptors from just-keypoints, every pair matched with adaptive
//   depth and width. Its weights (Apache-2.0) are served next to the page as lightglue-aliked-fp16.jmtw.
// - FlexGlue: RaCo keypoints, the shortlist (covisibility levels 1 and 2), and matches with early stop (level 3)
//   on the pairs it keeps.
// Keypoints come from just-keypoints' module (jkp/), the matcher from jmt.js. Both run on WebGPU in this page.
//
// Where things are: next to the page by default (the build's web/ folder); the documentation site sets other
// places in <script type="application/json" id="demo-config">: lib (jmt.js and its module), jkp, samples.
// FlexGlue's weights are never served by the page: the visitor downloads flexglue.pth from FlexGlue's release and
// loads it (or a .jmtw file) from their own disk; the page keeps it in the browser's private file system for the
// next visit only when the visitor ticks "Keep in this browser". URL parameters:
// run=1 runs once the sample set is loaded (and weights are at hand); weights=URL fetches the weights instead
// (a .pth or .jmtw file, or a parts manifest), for local testing; jkp=DIR for just-keypoints' module and its
// raco.jkpw and aliked-n16.jkpw; lightglue=URL for LightGlue's weights; model=flexglue starts with FlexGlue.
const params = new URLSearchParams(location.search);
const CONFIG = JSON.parse(document.getElementById('demo-config')?.textContent || '{}');
const dir = d => d.replace(/\/?$/, '/');
const LIB = dir(CONFIG.lib ?? './');
const JKP_DIR = dir(params.get('jkp') ?? CONFIG.jkp ?? 'jkp/');
const SAMPLES_DIR = dir(CONFIG.samples ?? 'samples/');
const WEIGHTS_URL = params.get('weights');
const LIGHTGLUE_URL = params.get('lightglue') ?? CONFIG.lightglue ?? 'lightglue-aliked-fp16.jmtw';
const STORED = 'flexglue-weights';   // the visitor's weights file, in the origin private file system
const jmt = await import(new URL(`${LIB}jmt.js`, location.href).href);
const SAMPLES = ['tum-1', 'tum-2', 'tum-3', 'tum-4', 'kitti-1', 'kitti-2', 'kitti-3', 'kitti-4',
                 'kitti-revisit-1', 'kitti-revisit-2', 'kitti-alike-1', 'kitti-alike-2'];
const MAX_SIDE = 1600;   // larger images are scaled down before keypoints and matching
const MAX_IMAGES = 48;
const MIN_COVISIBILITY = 0.8;
const KEYPOINTS = 2048;
// Sequential ramp (one hue, light to dark) for the level-1 matrix.
const RAMP = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf',
              '#1c5cab', '#184f95', '#104281', '#0d366b'];

const $ = id => document.getElementById(id);
const status = $('status'), progress = $('progress');

const state = {
  images: [],          // { name, bitmap, data, kp (RaCo xy), tKp, lg ({ xy, desc } from ALIKED), tLg, feat, tDescribe }
  extractors: {},      // by model: RaCo for FlexGlue, ALIKED for LightGlue
  lightglueWeights: null,
  matcher: null, matcherKey: '',
  weights: null,       // { name, size, blob or url, stored }: the visitor's FlexGlue weights
  timings: null,
  level1: null,        // Float32Array [n][n]
  shortlist: null,     // { n, pairs, scores, rerank }
  results: new Map(),  // 'i,j' -> match result of the current settings
  resultKey: '',
  selected: null,      // [i, j]
  busy: false,
  hasF16: false,
};

// ---- small helpers

const median = xs => {
  if (!xs.length) return NaN;
  const s = [...xs].sort((a, b) => a - b), m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const fmtMs = x => (Number.isFinite(x) ? (x >= 100 ? x.toFixed(0) : x.toFixed(1)) : '–');
const fmt3 = x => (Number.isFinite(x) ? x.toFixed(3) : '–');
const pairKey = (i, j) => (i < j ? `${i},${j}` : `${j},${i}`);
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function setStatus(text, bad = false) {
  status.textContent = text;
  status.className = bad ? 'bad' : '';
}

function setProgress(done, total) {
  progress.hidden = total == null;
  if (total != null) progress.value = total ? done / total : 0;
}

function darkMode() {
  const t = document.documentElement.dataset.theme;
  if (t) return t === 'dark';
  return matchMedia('(prefers-color-scheme: dark)').matches;
}

// Colour of t in [0, 1] on the ramp; in dark mode low values recede toward the dark surface.
function rampColor(t) {
  const ramp = darkMode() ? [...RAMP].reverse() : RAMP;
  const x = Math.min(Math.max(t, 0), 1) * (ramp.length - 1), i = Math.min(Math.floor(x), ramp.length - 2), f = x - i;
  const c = k => [1, 3, 5].map(o => parseInt(ramp[k].slice(o, o + 2), 16));
  const a = c(i), b = c(i + 1);
  return `rgb(${a.map((v, n) => Math.round(v + (b[n] - v) * f)).join(',')})`;
}

// Dark or light ink, whichever reads on an rgb(...) fill.
function contrastInk(fill) {
  const [r, g, b] = fill.match(/\d+/g).map(v => Number(v) / 255).map(c => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.3 ? '#1d1d1f' : '#ffffff';
}

// ---- images

async function bitmapOf(blob) {
  let bitmap = await createImageBitmap(blob);
  const side = Math.max(bitmap.width, bitmap.height);
  if (side > MAX_SIDE) {
    const s = MAX_SIDE / side;
    const scaled = await createImageBitmap(bitmap, { resizeWidth: Math.round(bitmap.width * s), resizeHeight: Math.round(bitmap.height * s),
                                                     resizeQuality: 'high' });
    bitmap.close();
    bitmap = scaled;
  }
  return bitmap;
}

function invalidate() {
  state.level1 = null;
  state.shortlist = null;
  state.results.clear();
  state.selected = null;
  renderMatrix();
  renderPairs();
  renderPair();
}

async function addImages(entries) {  // [{ name, blob }]
  for (const { name, blob } of entries) {
    if (state.images.length >= MAX_IMAGES) {
      setStatus(`At most ${MAX_IMAGES} images.`, true);
      break;
    }
    try {
      const bitmap = await bitmapOf(blob);
      state.images.push({ name, bitmap, data: jmt.rgba(bitmap), kp: null, tKp: 0, lg: null, tLg: 0, feat: null, tDescribe: 0 });
    } catch (e) {
      setStatus(`Cannot read ${name}: ${e.message}`, true);
    }
  }
  invalidate();
  renderThumbs();
}

function clearImages() {
  for (const im of state.images) {
    im.feat?.destroy();
    im.bitmap.close();
  }
  state.images = [];
  invalidate();
  renderThumbs();
}

async function loadSamples() {
  clearImages();
  setStatus('Loading the sample images…');
  const entries = await Promise.all(SAMPLES.map(async name => {
    const r = await fetch(`${SAMPLES_DIR}${name}.jpg`);
    if (!r.ok) throw new Error(`${SAMPLES_DIR}${name}.jpg: ${r.status}`);
    return { name, blob: await r.blob() };
  }));
  await addImages(entries);
  setStatus(`${state.images.length} images. Press Run.`);
}

function renderThumbs() {
  const box = $('thumbs');
  box.replaceChildren(...state.images.map((im, i) => {
    const div = document.createElement('div');
    div.className = 'thumb';
    div.title = `${i + 1}: ${im.name} (${im.bitmap.width} x ${im.bitmap.height})`;
    const c = document.createElement('canvas');
    c.width = 128;
    c.height = 96;
    const g = c.getContext('2d');
    const s = Math.max(c.width / im.bitmap.width, c.height / im.bitmap.height);
    const w = im.bitmap.width * s, h = im.bitmap.height * s;
    g.drawImage(im.bitmap, (c.width - w) / 2, (c.height - h) / 2, w, h);
    const label = document.createElement('span');
    label.textContent = i + 1;
    div.append(c, label);
    return div;
  }));
}

// ---- modules

async function fetchWithProgress(url, label) {
  return jmt.fetchWeights(url, {
    onProgress(got, total) {
      setStatus(total ? `${label}: ${(got / 1e6).toFixed(0)} of ${(total / 1e6).toFixed(0)} MB…` : `${label}…`);
      if (total) setProgress(got, total);
    },
  });
}

// The keypoints each matcher takes: RaCo's for FlexGlue, ALIKED's with their descriptors for LightGlue.
async function ensureExtractor(model, t) {
  if (state.extractors[model]) return;
  setStatus(`Loading just-keypoints (${model === 'lightglue' ? 'ALIKED' : 'RaCo'})…`);
  const t0 = performance.now();
  const jkp = await import(new URL(`${JKP_DIR}jkp.js`, location.href).href);
  state.extractors[model] = model === 'lightglue'
    ? await jkp.create({ weightsUrl: `${JKP_DIR}aliked-n16.jkpw`, topK: KEYPOINTS })
    : await jkp.create({ weightsUrl: `${JKP_DIR}raco.jkpw`, topK: KEYPOINTS, skipRanker: true, skipCovariance: true });
  t.loadKeypoints = performance.now() - t0;
}

const xyOf = (im, model) => (model === 'lightglue' ? im.lg.xy : im.kp);

// ---- FlexGlue's weights, from the visitor

// A .pth checkpoint is a zip archive; a .jmtw file starts with JKPW.
async function weightsKind(blob) {
  const head = new Uint8Array(await blob.slice(0, 4).arrayBuffer());
  if (head[0] === 0x50 && head[1] === 0x4b && head[2] === 3 && head[3] === 4) return 'pth';
  if (String.fromCharCode(...head) === 'JKPW') return 'jmtw';
  return null;
}

async function storedDir() {
  try {
    return await navigator.storage.getDirectory();
  } catch {
    return null;   // no private file system here (private window, blocked storage): the file is used once
  }
}

function showWeights(note) {
  const w = state.weights, el = $('weights-status');
  $('weights-keep').checked = !!w?.stored;
  $('weights-keep').disabled = !!w?.url;
  $('weights-panel').classList.toggle('need', !w);
  el.classList.toggle('ok', !!w);
  el.textContent = w ? `Loaded: ${w.name}${w.size ? `, ${(w.size / 1e6).toFixed(0)} MB` : ''}${w.stored ? ', kept in this browser' : ''}.` +
                       (note ? ' ' + note : '')
                     : note || 'Not loaded.';
}

// Copies the loaded file into the private file system; false when this browser cannot keep it.
async function storeWeights() {
  const w = state.weights, dir = await storedDir();
  try {
    if (!dir || !w?.blob) throw new Error('no storage');
    const handle = await dir.getFileHandle(STORED, { create: true });
    const out = await handle.createWritable();
    await out.write(w.blob);
    await out.close();
    localStorage.setItem(STORED, JSON.stringify({ name: w.name, size: w.size }));
    w.stored = true;
    return true;
  } catch {
    return false;
  }
}

async function unstoreWeights() {
  try {
    await (await storedDir())?.removeEntry(STORED);
  } catch { /* nothing stored */ }
  try {
    localStorage.removeItem(STORED);
  } catch { /* storage blocked */ }
  if (state.weights) state.weights.stored = false;
}

async function useWeightsFile(file) {
  if (state.busy) return;
  const kind = await weightsKind(file);
  if (!kind) {
    showWeights(`${file.name} is not a FlexGlue weights file (flexglue.pth or .jmtw).`);
    return;
  }
  const keep = $('weights-keep').checked;
  dropMatcher();
  await unstoreWeights();   // a copy of an earlier file is stale now
  state.weights = { name: file.name, size: file.size, blob: file, stored: false };
  if (keep) {
    showWeights('Keeping a copy in this browser…');
    showWeights((await storeWeights()) ? '' : 'This browser cannot keep it: load it again next visit.');
  } else showWeights();
  setStatus(state.images.length >= 2 ? `${state.images.length} images. Press Run.` : 'Add at least two images.');
}

// "Keep in this browser" ticked or unticked once a file is loaded.
async function keepWeights(keep) {
  if (!state.weights || state.weights.url) return;
  if (state.busy) {
    $('weights-keep').checked = !keep;
    return;
  }
  if (!keep) {
    await unstoreWeights();
    return showWeights();
  }
  showWeights('Keeping a copy in this browser…');
  showWeights((await storeWeights()) ? '' : 'This browser cannot keep it.');
}

async function restoreWeights() {
  if (WEIGHTS_URL) {
    state.weights = { name: WEIGHTS_URL.split('/').pop(), size: 0, url: WEIGHTS_URL, stored: false };
    showWeights('Fetched from the URL given.');
    return;
  }
  const dir = await storedDir();
  if (!dir) return showWeights();
  try {
    const file = await (await dir.getFileHandle(STORED)).getFile();
    const meta = JSON.parse(localStorage.getItem(STORED) || '{}');
    if (!(await weightsKind(file))) throw new Error('not weights');
    state.weights = { name: meta.name || 'flexglue.pth', size: file.size, blob: file, stored: true };
  } catch {
    state.weights = null;
  }
  showWeights();
}

function dropMatcher() {
  for (const im of state.images) {
    im.feat?.destroy();
    im.feat = null;
  }
  state.matcher?.destroy();
  state.matcher = null;
  state.matcherKey = '';
}

async function ensureMatcher(s, t) {
  const { model, precision, size } = s;
  const key = `${model}/${precision}/${model === 'lightglue' ? '' : size}`;
  if (state.matcher && state.matcherKey === key) return;
  dropMatcher();
  let t0 = performance.now(), weights, name;
  if (model === 'lightglue') {
    name = 'LightGlue weights';
    if (!state.lightglueWeights) state.lightglueWeights = await fetchWithProgress(LIGHTGLUE_URL, name);
    weights = state.lightglueWeights;
  } else {
    const w = state.weights;
    name = w.name;
    setStatus(`Reading ${w.name}…`);
    weights = w.url ? await fetchWithProgress(w.url, `FlexGlue weights`) : new Uint8Array(await w.blob.arrayBuffer());
  }
  t.loadWeights = performance.now() - t0;
  t.weightsMB = weights.length / 1e6;
  setProgress(null);
  setStatus('Creating the matcher…');
  t0 = performance.now();
  try {
    state.matcher = await jmt.create({ weights, precision, inputSize: size });
  } catch (e) {
    throw new Error(`${name}: ${e.message}`);
  }
  t.create = performance.now() - t0;
  state.matcherKey = key;
}

// ---- the run

function settings() {
  const early = $('early').value;
  return {
    model: document.querySelector('input[name=model]:checked').value,
    adaptive: $('adaptive').value === 'on',
    precision: document.querySelector('input[name=precision]:checked').value,
    size: Number($('size').value),
    layers: Number($('layers').value),
    k: Math.max(1, Number($('k').value) || 1),
    neighbours: Math.max(0, Number($('neighbours').value) || 0),
    rerank: $('rerank').checked,
    early: early === 'off' ? null : Number(early),
  };
}

function matchOptions(s) {
  if (s.model === 'lightglue') return { layers: s.layers, depthConfidence: s.adaptive ? 0.95 : 0, widthConfidence: s.adaptive ? 0.99 : 0 };
  return s.early == null ? { layers: s.layers } : { layers: s.layers, minCovisibility: MIN_COVISIBILITY, earlyStopDecay: s.early };
}

async function matchPair(i, j, s) {
  const a = state.images[i], b = state.images[j];
  const t0 = performance.now();
  const r = await state.matcher.match(a.feat, xyOf(a, s.model), b.feat, xyOf(b, s.model), matchOptions(s));
  r.ms = performance.now() - t0;
  r.model = s.model;
  r.layers = s.layers;
  r.checked = s.model === 'flexglue' && s.early != null;
  if (s.model === 'lightglue' && state.level1) {
    const n = state.images.length;
    state.level1[i * n + j] = state.level1[j * n + i] = r.n;  // the matrix shows match counts
  }
  state.results.set(pairKey(i, j), r);
  return r;
}

async function run() {
  if (state.busy) return null;
  const n = state.images.length;
  if (n < 2) {
    setStatus('Add at least two images.', true);
    return null;
  }
  const s = settings();
  if (s.model === 'flexglue' && !state.weights) {
    askForWeights();
    return null;
  }
  state.busy = true;
  $('run').disabled = true;
  const t = { ...(state.timings?.persist ?? {}) };
  const tRun = performance.now();
  try {
    await ensureExtractor(s.model, t);
    await ensureMatcher(s, t);
    t.persist = { loadKeypoints: t.loadKeypoints, loadWeights: t.loadWeights, weightsMB: t.weightsMB, create: t.create };
    t.model = s.model;
    invalidate();
    const lg = s.model === 'lightglue';
    for (const [i, im] of state.images.entries()) {
      if ((lg ? im.lg : im.kp) && im.feat) continue;
      setStatus(`Image ${i + 1} of ${n}: keypoints and features…`);
      setProgress(i, n);
      if (lg && !im.lg) {
        const t0 = performance.now();
        const k = await state.extractors.lightglue.extract(im.data);
        im.lg = { xy: k.xy, desc: k.desc };
        im.tLg = performance.now() - t0;
      } else if (!lg && !im.kp) {
        const t0 = performance.now();
        im.kp = (await state.extractors.flexglue.extract(im.data)).xy;
        im.tKp = performance.now() - t0;
      }
      if (!im.feat) {
        const t0 = performance.now();
        im.feat = lg ? await state.matcher.describeKeypoints(im.data, im.lg.xy, im.lg.desc) : await state.matcher.describe(im.data);
        await state.matcher.finish();
        im.tDescribe = performance.now() - t0;
      }
    }
    t.keypoints = median(state.images.map(im => (lg ? im.tLg : im.tKp)));
    t.describe = median(state.images.map(im => im.tDescribe));
    if (lg) return await runLightGlue(s, t, tRun);
    const feats = state.images.map(im => im.feat);
    setStatus('Similarity of every pair…');
    let t0 = performance.now();
    state.level1 = await state.matcher.level1(feats);
    t.level1 = performance.now() - t0;
    setStatus('Picking the pairs to match…');
    t0 = performance.now();
    state.shortlist = { ...(await state.matcher.shortlist(feats, { k: s.k, neighbours: s.neighbours, rerank: s.rerank })), rerank: s.rerank,
                        model: 'flexglue' };
    t.shortlist = performance.now() - t0;
    state.resultKey = JSON.stringify(s);
    renderMatrix();
    renderPairs();
    const m = state.shortlist.n;
    for (let p = 0; p < m; p++) {
      const i = state.shortlist.pairs[2 * p], j = state.shortlist.pairs[2 * p + 1];
      setStatus(`Matching pair ${p + 1} of ${m}: ${i + 1} – ${j + 1}…`);
      setProgress(p, m);
      await matchPair(i, j, s);
      renderPairs();
    }
    setProgress(null);
    const results = [...state.results.values()];
    const full = results.filter(r => r.stopLayer >= r.layers);
    const stopped = results.filter(r => r.stopLayer < r.layers);
    t.matchFull = median(full.map(r => r.ms));
    t.matchStop = median(stopped.map(r => r.ms));
    t.nFull = full.length;
    t.nStop = stopped.length;
    t.total = performance.now() - tRun;
    t.n = n;
    t.nShort = m;
    t.rerank = s.rerank;
    state.timings = t;
    renderTimings();
    renderMatrix();
    // Show the best pair that passed.
    let best = null;
    for (let p = 0; p < m && !best; p++) {
      const i = state.shortlist.pairs[2 * p], j = state.shortlist.pairs[2 * p + 1];
      if (state.results.get(pairKey(i, j)).n > 0) best = [i, j];
    }
    if (best || m) select(...(best ?? [state.shortlist.pairs[0], state.shortlist.pairs[1]]));
    const rejected = results.filter(r => r.checked && r.covisibility < MIN_COVISIBILITY).length;
    setStatus(`Matched ${m} of ${n * (n - 1) / 2} pairs in ${(t.total / 1000).toFixed(1)} s` +
              (s.early != null ? `: ${m - rejected} same place, ${rejected} different places.` : '.'));
    return { n, pairs: m, rejected, stopped: stopped.length, timings: t };
  } catch (e) {
    setProgress(null);
    setStatus('Error: ' + e.message, true);
    console.error(e);
    return { error: e.message };
  } finally {
    state.busy = false;
    $('run').disabled = false;
  }
}

// LightGlue has no shortlist: every pair is matched, and the matrix shows how many matches each pair got.
async function runLightGlue(s, t, tRun) {
  const n = state.images.length, m = n * (n - 1) / 2;
  const pairs = new Int32Array(2 * m);
  for (let i = 0, p = 0; i < n; i++) for (let j = i + 1; j < n; j++, p++) pairs[2 * p] = i, pairs[2 * p + 1] = j;
  state.shortlist = { n: m, pairs, scores: new Float32Array(m).fill(NaN), rerank: false, model: 'lightglue' };
  state.level1 = new Float32Array(n * n).fill(0);
  state.resultKey = JSON.stringify(s);
  renderMatrix();
  renderPairs();
  for (let p = 0; p < m; p++) {
    const i = pairs[2 * p], j = pairs[2 * p + 1];
    setStatus(`Matching pair ${p + 1} of ${m}: ${i + 1} – ${j + 1}…`);
    setProgress(p, m);
    await matchPair(i, j, s);
    if (p % 8 === 7 || p === m - 1) renderPairs(), renderMatrix();
  }
  setProgress(null);
  const results = [...state.results.values()];
  const full = results.filter(r => r.stopLayer >= r.layers), stopped = results.filter(r => r.stopLayer < r.layers);
  t.matchFull = median(full.map(r => r.ms));
  t.matchStop = median(stopped.map(r => r.ms));
  t.nFull = full.length;
  t.nStop = stopped.length;
  t.total = performance.now() - tRun;
  t.n = n;
  t.nShort = m;
  state.timings = t;
  renderTimings();
  renderMatrix();
  // Show the pair with the most matches.
  let best = null;
  for (let p = 0; p < m; p++) {
    const i = pairs[2 * p], j = pairs[2 * p + 1];
    if (!best || state.results.get(pairKey(i, j)).n > state.results.get(pairKey(...best)).n) best = [i, j];
  }
  if (best) await select(...best);
  setStatus(`Matched all ${m} pairs in ${(t.total / 1000).toFixed(1)} s.`);
  return { n, pairs: m, rejected: 0, stopped: stopped.length, timings: t };
}

// Run without weights: say so next to the button and point at the weights panel.
function askForWeights() {
  showWeights();
  setStatus("FlexGlue's weights are needed first: download them, then load them.", true);
  const panel = $('weights-panel');
  panel.scrollIntoView({ behavior: 'smooth', block: 'center' });
  panel.classList.remove('flash');
  void panel.offsetWidth;   // restart the animation
  panel.classList.add('flash');
  $('weights-open').focus({ preventScroll: true });
}

// A pair picked in the matrix or the table: shown, and matched first when it has no result yet.
async function select(i, j) {
  if (i > j) [i, j] = [j, i];
  state.selected = [i, j];
  renderPairs();
  if (!state.results.has(pairKey(i, j)) && state.matcher && state.images[i]?.feat && state.images[j]?.feat) {
    if (state.busy) return renderPair();
    state.busy = true;
    $('run').disabled = true;
    try {
      setStatus(`Matching ${i + 1} – ${j + 1} (not picked)…`);
      await matchPair(i, j, JSON.parse(state.resultKey));
      setStatus(`Matched ${state.images[i].name} – ${state.images[j].name}.`);
      renderMatrix();
    } catch (e) {
      setStatus('Error: ' + e.message, true);
    } finally {
      state.busy = false;
      $('run').disabled = false;
    }
  }
  renderPair();
}

// ---- rendering

// FlexGlue's verdict on a pair: same place (matched), or different places (no matches; most stop early, which
// only saves time).
function verdict(r) {
  if (!r || r.model === 'lightglue') return '';
  if (!r.checked) return '<span class="tag" title="Early stop is off: FlexGlue keeps the matches of every pair">not checked</span>';
  if (r.covisibility >= MIN_COVISIBILITY) return '<span class="tag pass" title="FlexGlue judges that the images show the same place">same place</span>';
  const how = r.stopLayer < r.layers ? `stopped after ${r.stopLayer} of ${r.layers} layers, which saves time` : `after all ${r.layers} layers`;
  return `<span class="tag" title="FlexGlue judges that the images show different places (${how}), so the pair gets no matches">different places</span>`;
}

function level2Of(i, j) {
  const sl = state.shortlist;
  if (!sl?.rerank) return NaN;
  for (let p = 0; p < sl.n; p++) if (sl.pairs[2 * p] === i && sl.pairs[2 * p + 1] === j) return sl.scores[p];
  return NaN;
}

function renderPairs() {
  const body = $('pairs').tBodies[0];
  const sl = state.shortlist;
  $('th-l2').hidden = !sl?.rerank;
  if (!sl) {
    body.innerHTML = '<tr><td colspan="9" class="l">Run to see the pairs worth matching.</td></tr>';
    return;
  }
  const n = state.images.length;
  const rows = [];
  for (let p = 0; p < sl.n; p++) {
    const i = sl.pairs[2 * p], j = sl.pairs[2 * p + 1];
    const r = state.results.get(pairKey(i, j));
    const sel = state.selected && state.selected[0] === i && state.selected[1] === j;
    rows.push(`<tr data-i="${i}" data-j="${j}"${sel ? ' class="sel"' : ''}><td>${p + 1}</td>` +
      `<td class="l">${i + 1} ${state.images[i].name} – ${j + 1} ${state.images[j].name}</td>` +
      `<td>${sl.model === 'lightglue' ? '–' : fmt3(state.level1[i * n + j])}</td>${sl.rerank ? `<td>${fmt3(sl.scores[p])}</td>` : ''}` +
      `<td>${r ? (r.model === 'lightglue' ? '–' : fmt3(r.covisibility)) : '…'}</td><td>${r ? `${r.stopLayer} / ${r.layers}` : ''}</td>` +
      `<td>${r ? r.n : ''}</td><td>${r ? fmtMs(r.ms) : ''}</td><td class="l">${verdict(r)}</td></tr>`);
  }
  body.innerHTML = rows.join('') || '<tr><td colspan="9" class="l">No pairs kept.</td></tr>';
}

function renderTimings() {
  const t = state.timings;
  if (!t) return;
  const lg = t.model === 'lightglue';
  const rows = lg ? [
    ['just-keypoints module and ALIKED weights', t.loadKeypoints],
    [`LightGlue weights read (${fmtMs(t.weightsMB)} MB)`, t.loadWeights],
    ['Matcher created (weights converted and uploaded to the GPU)', t.create],
    ['Keypoints and descriptors, per image', t.keypoints],
    ['Keypoints to the GPU, per image', t.describe],
    [`Match, all layers (${t.nFull} pairs)`, t.matchFull],
    [`Match, stopped early (${t.nStop} pairs)`, t.matchStop],
    ['This run, in all', t.total],
  ] : [
    ['just-keypoints module and RaCo weights', t.loadKeypoints],
    [`FlexGlue weights read (${fmtMs(t.weightsMB)} MB)`, t.loadWeights],
    ['Matcher created (weights converted and uploaded to the GPU)', t.create],
    ['Keypoints, per image', t.keypoints],
    ['Features (describe), per image', t.describe],
    [`Similarity, all ${t.n * (t.n - 1) / 2} pairs`, t.level1],
    [t.rerank ? `Picking pairs, with re-rank (${t.nShort} picked)` : `Picking pairs (${t.nShort})`, t.shortlist],
    [`Match, all layers (${t.nFull} pairs)`, t.matchFull],
    [`Match, stopped early (${t.nStop} pairs)`, t.matchStop],
    ['This run, in all', t.total],
  ];
  $('timings').tBodies[0].innerHTML = rows.map(([k, v]) => `<tr><td class="l">${k}</td><td>${fmtMs(v)} ms</td></tr>`).join('');
}

function renderMatrix() {
  const svg = $('matrix'), n = state.images.length, L = state.level1;
  $('legend').hidden = $('matrix-hint').hidden = !L;
  if (!L || n < 2) {
    svg.setAttribute('viewBox', '0 0 10 10');
    svg.setAttribute('width', 0);
    svg.innerHTML = '';
    return;
  }
  // Each pair once: row i (images 1 to n - 1), column j > i (images 2 to n).
  const cell = Math.max(10, Math.min(32, Math.floor(420 / (n - 1)))), pad = 22, size = pad + (n - 1) * cell;
  const X = j => pad + (j - 1) * cell, Y = i => pad + i * cell;
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) lo = Math.min(lo, L[i * n + j]), hi = Math.max(hi, L[i * n + j]);
  const span = hi - lo || 1;
  const kept = new Set();
  const sl = state.shortlist, lg = sl?.model === 'lightglue';
  if (sl && !lg) for (let p = 0; p < sl.n; p++) kept.add(pairKey(sl.pairs[2 * p], sl.pairs[2 * p + 1]));
  const out = [];
  const every = cell >= 16 ? 1 : Math.ceil(16 / cell);
  for (let k = 0; k < n - 1; k++) {
    if (k % every === 0) {
      out.push(`<text x="${X(k + 1) + cell / 2}" y="${pad - 8}" text-anchor="middle">${k + 2}</text>`);
      out.push(`<text x="${pad - 6}" y="${Y(k) + cell / 2 + 3.5}" text-anchor="end">${k + 1}</text>`);
    }
  }
  const gap = cell >= 14 ? 2 : 1, marks = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const x = X(j), y = Y(i);
      const fill = rampColor((L[i * n + j] - lo) / span);
      if (kept.has(pairKey(i, j))) marks.push([x + cell / 2, y + cell / 2, contrastInk(fill)]);
      out.push(`<rect class="cell" data-i="${i}" data-j="${j}" x="${x + gap / 2}" y="${y + gap / 2}" width="${cell - gap}" ` +
               `height="${cell - gap}" rx="2" fill="${fill}"/>`);
    }
  }
  const radius = Math.max(2, cell * 0.16);
  for (const [x, y, ink] of marks) out.push(`<circle cx="${x}" cy="${y}" r="${radius}" fill="${ink}" pointer-events="none"/>`);
  if (state.selected) {
    const [i, j] = state.selected;
    out.push(`<rect x="${X(j) - 1}" y="${Y(i) - 1}" width="${cell + 2}" height="${cell + 2}" rx="3" ` +
             `fill="none" stroke="${css('--accent')}" stroke-width="2" pointer-events="none"/>`);
  }
  svg.setAttribute('viewBox', `0 0 ${size + 2} ${size + 2}`);
  svg.setAttribute('width', size + 2);
  svg.innerHTML = out.join('');
  $('lmin').textContent = lg ? lo.toFixed(0) : lo.toFixed(3);
  $('lmax').textContent = lg ? hi.toFixed(0) : hi.toFixed(3);
  $('legend-label').textContent = lg ? 'matches' : 'similarity';
  $('legend-picked').hidden = lg;
  const stops = Array.from({ length: 9 }, (_, k) => rampColor(k / 8));
  $('ramp').style.background = `linear-gradient(to right, ${stops.join(',')})`;
}

function tipHtml(i, j) {
  const n = state.images.length, a = Math.min(i, j), b = Math.max(i, j);
  const sl = state.shortlist;
  let rank = -1;
  if (sl) for (let p = 0; p < sl.n; p++) if (sl.pairs[2 * p] === a && sl.pairs[2 * p + 1] === b) rank = p;
  const r = state.results.get(pairKey(a, b));
  if (sl?.model === 'lightglue') {
    const head = `<b>${a + 1} ${state.images[a].name} – ${b + 1} ${state.images[b].name}</b>`;
    return r ? `${head}<br><b>${r.n}</b> matches after ${r.stopLayer} of ${r.layers} layers` : `${head}<br><span class="m">not matched yet</span>`;
  }
  const lines = [`<b>${a + 1} ${state.images[a].name} – ${b + 1} ${state.images[b].name}</b>`,
                 `similarity <b>${fmt3(state.level1[a * n + b])}</b>` + (rank >= 0 ? `, picked #${rank + 1}` : ', <span class="m">not picked</span>')];
  const l2 = level2Of(a, b);
  if (Number.isFinite(l2)) lines.push(`re-rank <b>${fmt3(l2)}</b>`);
  if (r) {
    const same = !r.checked || r.covisibility >= MIN_COVISIBILITY;
    lines.push(`overlap <b>${fmt3(r.covisibility)}</b>: ` + (same ? `<b>${r.n}</b> matches` :
               `different places${r.stopLayer < r.layers ? `, stopped after ${r.stopLayer} of ${r.layers} layers` : ''}`));
  } else lines.push('<span class="m">click to match</span>');
  return lines.join('<br>');
}

function wireMatrix() {
  const svg = $('matrix'), tip = $('tip');
  svg.addEventListener('pointermove', e => {
    const c = e.target.closest?.('.cell');
    if (!c || c.dataset.i === c.dataset.j) {
      tip.hidden = true;
      return;
    }
    tip.innerHTML = tipHtml(Number(c.dataset.i), Number(c.dataset.j));
    tip.hidden = false;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = `${Math.min(e.clientX + 14, innerWidth - w - 8)}px`;
    tip.style.top = `${Math.min(e.clientY + 14, innerHeight - h - 8)}px`;
  });
  svg.addEventListener('pointerleave', () => (tip.hidden = true));
  svg.addEventListener('click', e => {
    const c = e.target.closest?.('.cell');
    if (c && c.dataset.i !== c.dataset.j) select(Number(c.dataset.i), Number(c.dataset.j)).then(renderMatrix);
  });
  $('pairs').addEventListener('click', e => {
    const tr = e.target.closest('tr[data-i]');
    if (tr) select(Number(tr.dataset.i), Number(tr.dataset.j)).then(renderMatrix);
  });
}

function renderPair() {
  const canvas = $('view'), stats = $('pair-stats');
  const sel = state.selected;
  const r = sel && state.results.get(pairKey(...sel));
  const width = canvas.parentElement.clientWidth - 32;
  const dpr = devicePixelRatio || 1;
  if (!sel || !r) {
    stats.innerHTML = `<div class="stat"><div class="k">${sel ? 'Matched once the run is over.' : state.shortlist ? 'Click a pair in the grid or the table.' : 'Press Run, then click a pair.'}</div></div>`;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(120 * dpr);
    canvas.style.height = '120px';
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
    return;
  }
  const [i, j] = sel, A = state.images[i], B = state.images[j];
  const n = state.images.length;
  const l2 = level2Of(i, j);
  const stat = (k, v) => `<div class="stat"><div class="v">${v}</div><div class="k">${k}</div></div>`;
  if (r.model === 'lightglue') stats.innerHTML = stat(`${A.name} – ${B.name}`, `${i + 1} – ${j + 1}`) + stat('matches', r.n) +
    stat('layers run', `${r.stopLayer} / ${r.layers}`) + stat('match time', `${fmtMs(r.ms)} ms`);
  else stats.innerHTML = `<div class="stat"><div class="v" style="font-size:14px;padding-top:4px">${verdict(r) || '–'}</div><div class="k">verdict</div></div>` +
    stat(`${A.name} – ${B.name}`, `${i + 1} – ${j + 1}`) + stat('matches', r.n) +
    stat('similarity', fmt3(state.level1[i * n + j])) + (Number.isFinite(l2) ? stat('re-rank', fmt3(l2)) : '') +
    stat('overlap', fmt3(r.covisibility)) + stat('layers run', `${r.stopLayer} / ${r.layers}`) + stat('match time', `${fmtMs(r.ms)} ms`);
  const aA = A.bitmap.width / A.bitmap.height, aB = B.bitmap.width / B.bitmap.height, gap = 8;
  const stacked = aA >= 1.5 && aB >= 1.5;
  let rectA, rectB, height;
  if (stacked) {
    const hA = width / aA, hB = width / aB;
    rectA = [0, 0, width, hA];
    rectB = [0, hA + gap, width, hB];
    height = hA + gap + hB;
  } else {
    const h = Math.min((width - gap) / (aA + aB), 520);
    const wA = h * aA, wB = h * aB, x0 = (width - wA - gap - wB) / 2;
    rectA = [x0, 0, wA, h];
    rectB = [x0 + wA + gap, 0, wB, h];
    height = h;
  }
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  canvas.style.height = `${height}px`;
  const g = canvas.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, width, height);
  g.drawImage(A.bitmap, ...rectA);
  g.drawImage(B.bitmap, ...rectB);
  const sA = rectA[2] / A.bitmap.width, sB = rectB[2] / B.bitmap.width;
  const kA = xyOf(A, r.model), kB = xyOf(B, r.model);
  const pa = k => [rectA[0] + (kA[2 * k] + 0.5) * sA, rectA[1] + (kA[2 * k + 1] + 0.5) * sA];
  const pb = k => [rectB[0] + (kB[2 * k] + 0.5) * sB, rectB[1] + (kB[2 * k + 1] + 0.5) * sB];
  if ($('show-kp').checked) {
    for (const [kp, p] of [[kA, pa], [kB, pb]]) {
      g.fillStyle = css('--kp');
      g.strokeStyle = css('--kp-ring');
      g.lineWidth = 0.75;
      for (let k = 0; k < kp.length / 2; k++) {
        const [x, y] = p(k);
        g.beginPath();
        g.arc(x, y, 1.6, 0, 2 * Math.PI);
        g.fill();
        g.stroke();
      }
    }
  }
  // Every match as a dot at both ends; lines for an even subsample of them, with a dark halo so they read on
  // any image.
  const all = [];
  for (let m = 0; m < r.n; m++) all.push([pa(r.matches[2 * m]), pb(r.matches[2 * m + 1])]);
  const limit = $('lines').value === 'all' ? all.length : Number($('lines').value);
  const step = Math.max(1, all.length / limit), lines = [];
  for (let x = 0; Math.floor(x) < all.length && lines.length < limit; x += step) lines.push(all[Math.floor(x)]);
  for (const [w, color] of [[2.4, 'rgba(0,0,0,0.4)'], [1, css('--match')]]) {
    g.lineWidth = w;
    g.strokeStyle = color;
    g.beginPath();
    for (const [[x0, y0], [x1, y1]] of lines) {
      g.moveTo(x0, y0);
      g.lineTo(x1, y1);
    }
    g.stroke();
  }
  const dot = Math.max(1.4, Math.min(2.4, width / 400));
  for (const [size, color] of [[dot + 1.6, 'rgba(0,0,0,0.45)'], [dot, css('--match')]]) {
    g.fillStyle = color;
    for (const [[x0, y0], [x1, y1]] of all) {
      g.fillRect(x0 - size / 2, y0 - size / 2, size, size);
      g.fillRect(x1 - size / 2, y1 - size / 2, size, size);
    }
  }
}

// Shows the settings of the chosen matcher. Its results and features are dropped: the images are matched
// anew with the other one.
function applyModel() {
  const model = settings().model;
  for (const el of document.querySelectorAll('[data-model]')) el.hidden = el.dataset.model !== model;
  if (state.busy) return;
  dropMatcher();
  invalidate();
  showWeights();
  setStatus(model === 'lightglue' ? 'LightGlue: its weights come with the page. Press Run.'
                                   : state.weights ? 'FlexGlue. Press Run.' : "FlexGlue's weights are needed to run: download them, then load them.");
}

// ---- start

async function init() {
  const banner = $('banner');
  const problems = [];
  if (!jmt.webgpuSupported()) problems.push('This browser has no WebGPU.');
  if (typeof WebAssembly.Suspending !== 'function') problems.push('This browser has no JSPI (WebAssembly.Suspending).');
  let adapterInfo = 'no WebGPU adapter';
  if (jmt.webgpuSupported()) {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (adapter) {
      const i = adapter.info || {};
      adapterInfo = [i.vendor, i.architecture, i.description].filter(Boolean).join(' / ') || 'WebGPU adapter';
      state.hasF16 = adapter.features.has('shader-f16');
    } else problems.push('No WebGPU adapter was found.');
  }
  if (problems.length) {
    banner.textContent = problems.join(' ') + ' The demo needs both: Chrome or Edge 137 or later.';
    banner.hidden = false;
  }
  $('adapter').textContent = `GPU: ${adapterInfo}; shader-f16: ${state.hasF16 ? 'yes' : 'no'}.`;
  if (!state.hasF16) {
    const f16 = document.querySelector('input[name=precision][value=fp16]');
    f16.disabled = true;
    document.querySelector('input[name=precision][value=fp32]').checked = true;
    $('precision-hint').textContent = 'This GPU has no shader-f16: FP32 only.';
    $('precision-hint').hidden = false;
  }

  $('weights-open').addEventListener('click', () => !state.busy && $('weights-file').click());
  $('weights-file').addEventListener('change', async e => {
    if (e.target.files[0]) await useWeightsFile(e.target.files[0]);
    e.target.value = '';
  });
  $('weights-keep').addEventListener('change', e => keepWeights(e.target.checked));
  await restoreWeights();

  $('samples').addEventListener('click', () => !state.busy && loadSamples());
  $('clear').addEventListener('click', () => !state.busy && clearImages());
  $('open').addEventListener('click', () => !state.busy && $('file').click());
  $('file').addEventListener('change', async e => {
    await addImages([...e.target.files].map(f => ({ name: f.name.replace(/\.[^.]+$/, ''), blob: f })));
    e.target.value = '';
    setStatus(`${state.images.length} images. Press Run.`);
  });
  const drop = $('drop');
  for (const target of [drop, document.body]) {
    target.addEventListener('dragover', e => {
      e.preventDefault();
      drop.classList.add('over');
    });
    target.addEventListener('dragleave', () => drop.classList.remove('over'));
    target.addEventListener('drop', async e => {
      e.preventDefault();
      drop.classList.remove('over');
      if (state.busy) return;
      const dropped = [...e.dataTransfer.files];
      const weights = dropped.find(f => /\.(pth|jmtw)$/i.test(f.name));
      if (weights) await useWeightsFile(weights);
      const files = dropped.filter(f => f.type.startsWith('image/'));
      if (!files.length) return;
      await addImages(files.map(f => ({ name: f.name.replace(/\.[^.]+$/, ''), blob: f })));
      setStatus(`${state.images.length} images. Press Run.`);
    });
  }
  if (params.get('model') === 'flexglue') document.querySelector('input[name=model][value=flexglue]').checked = true;
  for (const r of document.querySelectorAll('input[name=model]')) r.addEventListener('change', applyModel);
  applyModel();
  $('run').addEventListener('click', run);
  $('show-kp').addEventListener('change', renderPair);
  $('lines').addEventListener('change', renderPair);
  addEventListener('resize', () => renderPair());
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => (renderMatrix(), renderPair()));
  wireMatrix();
  renderPair();
  await loadSamples();
  const flex = settings().model === 'flexglue';
  if (flex && !state.weights) setStatus("FlexGlue's weights are needed to run: download them, then load them.");
  if (params.get('run') === '1' && (!flex || state.weights)) await run();
}

const ready = init().catch(e => setStatus('Error: ' + e.message, true));
window.jmtDemo = { state, run, select, ready };

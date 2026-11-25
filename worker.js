// Speech → English worker. Everything runs inside the browser (WebGPU if available, else WASM/CPU).
// Models are downloaded once, cached by the browser, and then work fully offline.
//
// Two modes:
//   fast     – Whisper translates speech straight to English (one small model).
//   accurate – Whisper writes down the speech in its own script, then NLLB-200 (Meta's
//              200-language translator) translates it to English. Falls back to "fast"
//              automatically when Whisper's transcript comes out in the wrong script.
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

env.allowLocalModels = false;
env.useBrowserCache = true;

const MT_MODEL = 'Xenova/nllb-200-distilled-600M';

// Unicode script ranges → NLLB language codes
const SCRIPTS = [
  ['gurmukhi',   /[਀-੿]/g, 'pan_Guru'],
  ['gujarati',   /[઀-૿]/g, 'guj_Gujr'],
  ['bengali',    /[ঀ-৿]/g, 'ben_Beng'],
  ['devanagari', /[ऀ-ॿ]/g, 'hin_Deva'],
  ['tamil',      /[஀-௿]/g, 'tam_Taml'],
  ['telugu',     /[ఀ-౿]/g, 'tel_Telu'],
  ['kannada',    /[ಀ-೿]/g, 'kan_Knda'],
  ['malayalam',  /[ഀ-ൿ]/g, 'mal_Mlym'],
  ['han',        /[㐀-鿿]/g, 'zho_Hans'],
  ['latin',      /[A-Za-z]/g,        'eng_Latn'],
];
// Which script each selectable language must be written in
const EXPECTED = {
  punjabi: 'gurmukhi', marathi: 'devanagari', hindi: 'devanagari', tamil: 'tamil', telugu: 'telugu',
  kannada: 'kannada', chinese: 'han', bengali: 'bengali', gujarati: 'gujarati', malayalam: 'malayalam',
};

function dominantScript(text) {
  let best = null, n = 0;
  for (const [name, re, code] of SCRIPTS) {
    const c = (text.match(re) || []).length;
    if (c > n) { n = c; best = { name, code }; }
  }
  return best;
}

let asr = null, asrKey = null, mt = null, mtLoading = null, device = 'wasm', loading = Promise.resolve();

async function hasWebGPU() {
  try { return !!(navigator.gpu && await navigator.gpu.requestAdapter()); } catch { return false; }
}

function tracker(label) {
  const files = {};
  return (p) => {
    if (p.status === 'progress' && p.total) {
      files[p.file] = [p.loaded, p.total];
      const [l, t] = Object.values(files).reduce((a, [x, y]) => [a[0] + x, a[1] + y], [0, 0]);
      self.postMessage({ type: 'progress', label, pct: Math.round((l / t) * 100), mb: Math.round(t / 1e6) });
    }
  };
}

// Speech model (needed in both modes)
async function load(model, loadId) {
  device = (await hasWebGPU()) ? 'webgpu' : 'wasm';
  const key = model + ':' + device;
  if (!asr || asrKey !== key) {
    if (asr) { try { await asr.dispose(); } catch {} asr = null; }
    self.postMessage({ type: 'status', text: `Loading speech model (${model.split('-').pop()})…` });
    asr = await pipeline('automatic-speech-recognition', model, {
      device,
      dtype: device === 'webgpu' ? { encoder_model: 'fp32', decoder_model_merged: 'q4' }
                                 : { encoder_model: 'q8', decoder_model_merged: 'q8' },
      progress_callback: tracker('Speech model'),
    });
    asrKey = key;
  }
  self.postMessage({ type: 'ready', loadId, device: device === 'webgpu' ? 'GPU (WebGPU)' : 'CPU' });
}

// Translator for Accurate mode. Loads in the background; Fast mode keeps working meanwhile.
function loadMT() {
  if (mt) { self.postMessage({ type: 'mt-ready' }); return; }
  if (mtLoading) return;
  mtLoading = pipeline('translation', MT_MODEL, { device: 'wasm', dtype: 'q8', progress_callback: tracker('Accurate-mode translator') })
    .then((m) => { mt = m; self.postMessage({ type: 'mt-ready' }); })
    .catch((e) => { mtLoading = null; self.postMessage({ type: 'mt-error', text: String((e && e.message) || e) }); });
}

async function whisper(audio, language, task) {
  const opts = { task, chunk_length_s: 30, stride_length_s: 5 };
  if (language && language !== 'auto') opts.language = language;
  return ((await asr(audio, opts)).text || '').trim();
}

async function processAudio({ audio, language, mode, showOriginal }) {
  if (mode !== 'accurate' || !mt) {
    const english = await whisper(audio, language, 'translate');
    const original = showOriginal ? await whisper(audio, language, 'transcribe') : '';
    return { english, original, via: mode === 'accurate' ? 'fast-while-downloading' : 'whisper' };
  }
  const original = await whisper(audio, language, 'transcribe');
  const script = dominantScript(original);
  const ok = script && (language === 'auto' ? true : script.name === EXPECTED[language]);
  if (!ok) {
    // Whisper wrote it in the wrong script (common for Punjabi): use direct speech translation.
    return { english: await whisper(audio, language, 'translate'), original, via: 'whisper-fallback' };
  }
  if (script.code === 'eng_Latn') return { english: original, original: '', via: 'nllb' };
  let src = script.code;
  if (script.name === 'devanagari' && language === 'marathi') src = 'mar_Deva';
  const out = await mt(original, { src_lang: src, tgt_lang: 'eng_Latn', max_new_tokens: 256 });
  return { english: (out[0]?.translation_text || '').trim(), original, via: 'nllb' };
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'load') {
      await (loading = load(data.model, data.loadId));
    } else if (data.type === 'loadMT') {
      loadMT();
    } else if (data.type === 'process') {
      await loading.catch(() => {}); // never process while the speech model is switching
      const t0 = performance.now();
      const r = await processAudio(data);
      self.postMessage({ type: 'result', id: data.id, ...r, ms: Math.round(performance.now() - t0) });
    }
  } catch (e) {
    self.postMessage({ type: 'error', text: String((e && e.message) || e), id: data.id });
  }
};

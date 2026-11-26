// Speech → English worker. Everything runs inside the browser (WebGPU if available, else WASM/CPU).
// Models are downloaded once, cached by the browser, and then work fully offline.
//
// Two modes:
//   fast     – Whisper translates speech straight to English (one small model).
//   accurate – Whisper writes down the speech in its own script, then NLLB-200 (Meta's
//              200-language translator) translates it to English. Falls back to "fast"
//              automatically when Whisper's transcript comes out in the wrong script.
import { pipeline, env, Tensor } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

env.allowLocalModels = false;
env.useBrowserCache = true;

// Two translators for Accurate mode:
//   full  – NLLB-200 (~900 MB download, ~3 GB RAM): laptops/desktops only
//   light – OPUS-MT many-languages→English (~110 MB, ~300 MB RAM): phones and low-memory devices
const MT_MODELS = { full: 'Xenova/nllb-200-distilled-600M', light: 'Xenova/opus-mt-mul-en' };
let mtKind = null;

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
async function load(model, loadId, forceWasm) {
  device = (!forceWasm && await hasWebGPU()) ? 'webgpu' : 'wasm';
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
function loadMT(kind) {
  if (mt && mtKind === kind) { self.postMessage({ type: 'mt-ready' }); return; }
  if (mtLoading) return;
  mtLoading = (async () => {
    await unloadMT();
    const m = await pipeline('translation', MT_MODELS[kind], { device: 'wasm', dtype: 'q8', progress_callback: tracker('Accurate-mode translator') });
    mt = m; mtKind = kind; mtLoading = null;
    self.postMessage({ type: 'mt-ready' });
  })().catch((e) => { mtLoading = null; self.postMessage({ type: 'mt-error', text: String((e && e.message) || e) }); });
}
// Free the translator's memory (used on phones when switching to Fast)
async function unloadMT() {
  if (mt) { try { await mt.dispose(); } catch {} }
  mt = null; mtKind = null;
}

async function whisper(audio, language, task) {
  const opts = { task, chunk_length_s: 30, stride_length_s: 5 };
  if (language && language !== 'auto') opts.language = language;
  return ((await asr(audio, opts)).text || '').trim();
}

// Real language detection: ask Whisper which language token is most likely,
// limited to the languages this app supports (avoids random guesses).
const DETECT = { pa: 'punjabi', mr: 'marathi', ta: 'tamil', te: 'telugu', kn: 'kannada', zh: 'chinese',
  bn: 'bengali', gu: 'gujarati', ml: 'malayalam', hi: 'hindi', ur: 'urdu', en: 'english' };
async function detectLanguage(audio) {
  const tok = asr.tokenizer.model.tokens_to_ids;
  const { input_features } = await asr.processor(audio.subarray(0, 16000 * 30));
  const sot = tok.get('<|startoftranscript|>');
  const out = await asr.model({ input_features, decoder_input_ids: new Tensor('int64', [BigInt(sot)], [1, 1]) });
  const logits = out.logits.data;
  return Object.entries(DETECT)
    .map(([code, name]) => ({ name, score: logits[tok.get(`<|${code}|>`)] ?? -Infinity }))
    .sort((a, b) => b.score - a.score);
}

async function processAudio(req) {
  let { audio, language } = req;
  let detected = null;
  let pre = null;
  if (language === 'auto') {
    const ranked = await detectLanguage(audio);
    if (ranked[0].name === 'english') {
      const text = await whisper(audio, 'english', 'transcribe');
      return { english: text, original: '', via: 'whisper', detected: 'english' };
    }
    // Similar-sounding languages (e.g. Tamil/Kannada) score close together. Confirm by
    // transcribing with each close candidate and keeping the one written in its own script.
    const close = ranked.filter((c) => c.score > ranked[0].score - 2 && EXPECTED[c.name]).slice(0, 3);
    language = ranked[0].name;
    if (close.length > 1) {
      for (const c of close) {
        const t = await whisper(audio, c.name, 'transcribe');
        if (dominantScript(t)?.name === EXPECTED[c.name]) { language = c.name; pre = t; break; }
      }
    }
    detected = language;
  }
  return { ...(await translateAudio({ ...req, language, pre })), detected };
}

async function translateAudio({ audio, language, mode, showOriginal, pre }) {
  if (mode !== 'accurate' || !mt) {
    const english = await whisper(audio, language, 'translate');
    const original = showOriginal ? (pre ?? await whisper(audio, language, 'transcribe')) : '';
    return { english, original, via: mode === 'accurate' ? 'fast-while-downloading' : 'whisper' };
  }
  const original = pre ?? await whisper(audio, language, 'transcribe');
  const script = dominantScript(original);
  const ok = script && EXPECTED[language] && script.name === EXPECTED[language];
  if (!ok) {
    // Whisper wrote it in the wrong script (common for Punjabi): use direct speech translation.
    return { english: await whisper(audio, language, 'translate'), original, via: 'whisper-fallback' };
  }
  if (script.code === 'eng_Latn') return { english: original, original: '', via: 'nllb' };
  let out;
  if (mtKind === 'light') {
    out = await mt(original, { max_new_tokens: 256 });
  } else {
    let src = script.code;
    if (script.name === 'devanagari' && language === 'marathi') src = 'mar_Deva';
    out = await mt(original, { src_lang: src, tgt_lang: 'eng_Latn', max_new_tokens: 256 });
  }
  return { english: (out[0]?.translation_text || '').trim(), original, via: 'nllb' };
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'load') {
      await (loading = load(data.model, data.loadId, data.forceWasm));
    } else if (data.type === 'loadMT') {
      loadMT(data.kind);
    } else if (data.type === 'unloadMT') {
      if (!mtLoading) await unloadMT();
    } else if (data.type === 'process') {
      await loading.catch(() => {}); // never process while the speech model is switching
      const t0 = performance.now();
      const r = await processAudio(data);
      self.postMessage({ type: 'result', id: data.id, ...r, ms: Math.round(performance.now() - t0) });
    }
  } catch (e) {
    const text = String((e && e.message) || e);
    // A bare number (or memory message) means the AI engine crashed, usually out of memory.
    // The page restarts the engine with lighter settings when it sees fatal: true.
    const fatal = /^\d+$/.test(text.trim()) || /memory|allocation|abort/i.test(text);
    self.postMessage({ type: 'error', text, fatal, id: data.id });
  }
};

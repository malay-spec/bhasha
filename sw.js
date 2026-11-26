// Offline support: app shell + the in-browser AI runtime are cached on first visit.
// Model weights are cached separately by transformers.js (Cache Storage "transformers-cache").
const CACHE = 'bhasha-setu-v3';
const SHELL = ['./', './index.html', './worker.js', './manifest.json', './icons/icon-192.png', './icons/icon-512.png'];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});
self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== 'GET') return;
  const isShell = url.origin === self.location.origin;
  const isRuntime = url.hostname === 'cdn.jsdelivr.net';
  if (!isShell && !isRuntime) return; // models: handled by transformers.js cache
  e.respondWith((async () => {
    const cache = await caches.open(CACHE);
    const hit = await cache.match(e.request, { ignoreSearch: isShell });
    // App shell: network-first (get updates), fall back to cache offline. Runtime: cache-first.
    if (isRuntime && hit) return hit;
    try {
      // Don't let a slow mobile network hold the app hostage: use the cached copy after 3 s.
      const net = fetch(e.request);
      const res = hit ? await Promise.race([net, new Promise((r) => setTimeout(() => r(hit), 3000))]) : await net;
      if (res === hit) { net.then((r) => r.ok && cache.put(e.request, r)).catch(() => {}); return hit; }
      if (res.ok) cache.put(e.request, res.clone());
      return res;
    } catch (err) {
      if (hit) return hit;
      throw err;
    }
  })());
});

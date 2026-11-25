# Bhasha Setu — offline speech → English (PWA)

Listens through the phone/laptop mic and turns Punjabi, Marathi, Tamil, Telugu, Kannada,
Mandarin, Bengali, Gujarati, Malayalam (and Hindi) speech into English **text and/or spoken English**.
All AI runs inside the browser on the device. Internet is needed only once, to download the models.

## Run it
The mic only works on `https://` or `http://localhost`.

**Laptop (quick test):**
```
cd bhasha-setu
python3 -m http.server 8080
```
Open http://localhost:8080 in Chrome or Edge.

**Phones (and sharing with others):** put the folder on any HTTPS static host —
GitHub Pages, Netlify Drop (drag the folder onto app.netlify.com/drop), Cloudflare Pages,
or the university web server. Open the link in Chrome → menu → *Add to Home screen* / *Install app*.

## First use
1. Pick the spoken language, model size and engine. The model downloads once (progress bar shows it).
2. Tap the mic and speak; pause briefly between sentences — each pause sends that sentence for translation.
3. Toggle *Show English text* / *Speak English aloud* / *Also show original language* any time.
4. After the first download you can switch off Wi-Fi/data — it keeps working.

## Choosing settings
| Device | Speech model | Engine | Download |
|---|---|---|---|
| Phone | Base | Fast | ~80 MB |
| Phone, better accuracy | Small | Fast | ~250 MB |
| Laptop | Base or Small | Accurate | ~1–1.2 GB |

*Accurate* = Whisper writes the speech down in its own script, then Meta's NLLB-200 translator turns it into English.
If Whisper writes in the wrong script (common for Punjabi), the app automatically falls back to direct speech translation.

## Files
- `index.html` – the app (UI, mic capture, pause detection, English voice output, saved conversation)
- `worker.js` – runs the AI models (Whisper via transformers.js, NLLB-200) in a background thread
- `sw.js` – offline cache · `manifest.json`, `icons/` – installable app

## Known limits
- Whisper is weaker on Punjabi, Gujarati, Kannada and Malayalam than on Tamil, Bengali, Marathi and Mandarin; use the Small model for these.
- Translation appears after each pause (sentence by sentence), not word by word. Expect about 2–8 s per sentence on CPU; faster in browsers with WebGPU (recent Chrome/Edge).
- Spoken English uses the voices installed on the device (Settings → Text-to-speech); most phones/laptops include offline English voices.
- Conversations are stored in the browser on that device until you press *Delete all*.

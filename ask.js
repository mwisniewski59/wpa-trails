// api/ask.js — Vercel serverless function (Node runtime)
// Holds the Gemini API key server-side, builds a mode-specific prompt,
// calls Gemini, and returns a clean JSON answer the browser can render + map.
//
// SETUP (when you're ready):
//   1. Get a free key at https://aistudio.google.com  (no credit card)
//   2. In Vercel: Project → Settings → Environment Variables
//      add  GEMINI_API_KEY = your_key_here
//   3. Redeploy. That's it — the key never touches the browser.

// gemini-2.5-flash is the current free-tier-eligible model (as of mid-2026).
// Google cut free-tier limits in Dec 2025 and 2.0-flash now has near-zero free quota,
// which causes a 429 on the very first request. If you still hit 429s on the free tier,
// enabling billing in Google AI Studio unlocks Tier 1 (no minimum spend) and clears it.
// Alternatives if you want higher free limits: "gemini-2.5-flash-lite".
const GEMINI_MODEL = "gemini-2.5-flash";
const GEMINI_URL = (key) =>
  `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${key}`;

// ── Simple in-memory rate limit (per warm instance). For real public scale,
//    swap this for Vercel KV / Upstash. Good enough to blunt abuse for now. ──
const HITS = new Map(); // ip -> {count, resetAt}
const RL_MAX = 30; // requests
const RL_WINDOW_MS = 60 * 60 * 1000; // per hour per IP

function rateLimited(ip) {
  const now = Date.now();
  const rec = HITS.get(ip);
  if (!rec || now > rec.resetAt) {
    HITS.set(ip, { count: 1, resetAt: now + RL_WINDOW_MS });
    return false;
  }
  rec.count += 1;
  return rec.count > RL_MAX;
}

// ── Prompt templates per mode ────────────────────────────────────────────────
const BASE_RULES = `You are a knowledgeable, friendly guide to rail-trails in Western Pennsylvania
(the Montour Trail, Great Allegheny Passage, Panhandle Trail, Three Rivers Heritage Trail,
and Rachel Carson Trail). You answer ONLY using the trail point-of-interest (POI) data
provided in this request. Never invent places, mileages, hours, or facts not present in the data.

When a POI has a low confidence score (below 75) or a verify_before_visit flag, mention that
the visitor should confirm before relying on it. If something relevant has a correction_note,
respect it (e.g. closures). Keep a warm, plainspoken tone — like a local who rides these trails.
Distances are in miles. Surfaces are: paved asphalt, crushed limestone, or on-road/shared.`;

function buildPrompt(mode, question, pois, context) {
  const data = JSON.stringify(pois);
  const ctxLine = context && context.routeSummary
    ? `\nThe user currently has this route on their map: ${context.routeSummary}\n`
    : "";

  if (mode === "ask") {
    return `${BASE_RULES}

POI DATA:
${data}
${ctxLine}
USER QUESTION: ${question}

Answer conversationally in 2-4 sentences. If you reference specific POIs, you may name them.
Return ONLY a JSON object, no markdown fences, in exactly this shape:
{"answer": "your prose answer", "pois": ["poi_id_1","poi_id_2"]}
The "pois" array lists the id fields of any POIs you referenced that should be highlighted on the map (may be empty).`;
  }

  if (mode === "ride" || mode === "plan") {
    return `${BASE_RULES}

POI DATA:
${data}
${ctxLine}
USER REQUEST: ${question}

Recommend ONE specific ride that fits the request. Choose a real start and end from the POI data
(use trailheads/access points where possible). Pick 3-6 highlights along the way to point out —
things to see, good stops, cautions. Order the stops the way you'd actually ride them.

Return ONLY a JSON object, no markdown fences, in exactly this shape:
{
  "answer": "a warm 2-4 sentence description of the ride and why it fits",
  "ride": {
    "title": "short ride name",
    "trail": "trail key (montour, gap, panhandle, three_rivers_heritage, rachel_carson)",
    "distance_note": "e.g. ~18 miles round trip, mostly flat",
    "surface_note": "e.g. crushed limestone with a paved stretch through Peters Township"
  },
  "stops": ["poi_id_in_ride_order_1","poi_id_2","poi_id_3"],
  "highlights": [
    {"poi_id":"id","look_for":"one sentence on what to notice or do here"}
  ]
}
Every poi_id MUST exist in the POI DATA. "stops" should be the riding order used to draw the route.`;
  }

  if (mode === "along") {
    return `${BASE_RULES}

POI DATA:
${data}
${ctxLine}
USER REQUEST: ${question}

The user wants to know what to see and look for along this stretch. Narrate the notable
sights in trail order — history, views, quirks, good stops, cautions. Be specific and vivid
but grounded only in the data.

Return ONLY a JSON object, no markdown fences, in exactly this shape:
{
  "answer": "a flowing 3-5 sentence narration in trail order",
  "highlights": [
    {"poi_id":"id","look_for":"one sentence on what to notice here"}
  ]
}
Every poi_id MUST exist in the POI DATA.`;
  }

  // fallback = ask
  return buildPrompt("ask", question, pois, context);
}

// ── Pull the model's text out of Gemini's response shape ─────────────────────
function extractText(data) {
  try {
    const parts = data.candidates?.[0]?.content?.parts || [];
    return parts.map((p) => p.text || "").join("").trim();
  } catch {
    return "";
  }
}

// ── Strip ```json fences if the model adds them, then parse ──────────────────
function safeParse(text) {
  if (!text) return null;
  let t = text.trim();
  // remove leading/trailing code fences
  t = t.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/i, "").trim();
  // grab the outermost JSON object if there's stray prose
  const first = t.indexOf("{");
  const last = t.lastIndexOf("}");
  if (first !== -1 && last !== -1 && last > first) t = t.slice(first, last + 1);
  try {
    return JSON.parse(t);
  } catch {
    return null;
  }
}

export default async function handler(req, res) {
  // CORS (same-origin in production, but harmless and helps local testing)
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") return res.status(200).end();
  if (req.method !== "POST")
    return res.status(405).json({ error: "Method not allowed" });

  const key = process.env.GEMINI_API_KEY;
  if (!key)
    return res.status(500).json({
      error:
        "Server not configured yet. Add GEMINI_API_KEY in Vercel project settings.",
    });

  // rate limit by IP
  const ip =
    (req.headers["x-forwarded-for"] || "").split(",")[0].trim() ||
    req.socket?.remoteAddress ||
    "unknown";
  if (rateLimited(ip))
    return res
      .status(429)
      .json({ error: "Whoa — too many requests this hour. Try again later." });

  // parse body (Vercel usually parses JSON, but be defensive)
  let body = req.body;
  if (typeof body === "string") {
    try {
      body = JSON.parse(body);
    } catch {
      return res.status(400).json({ error: "Invalid JSON body" });
    }
  }
  const { mode = "ask", question = "", pois = [], context = null } = body || {};
  if (!question.trim())
    return res.status(400).json({ error: "No question provided" });
  if (!Array.isArray(pois) || pois.length === 0)
    return res.status(400).json({ error: "No POI data provided" });

  const prompt = buildPrompt(mode, question, pois, context);

  // Call Gemini with a couple of short retries on transient limits (429/503).
  async function callGemini() {
    const payload = {
      contents: [{ role: "user", parts: [{ text: prompt }] }],
      generationConfig: {
        temperature: 0.7,
        maxOutputTokens: 1024,
        responseMimeType: "application/json", // ask Gemini for raw JSON
      },
    };
    const delays = [0, 900, 2200]; // first try immediate, then backoff
    let last = null;
    for (let i = 0; i < delays.length; i++) {
      if (delays[i]) await new Promise((r) => setTimeout(r, delays[i]));
      const r = await fetch(GEMINI_URL(key), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (r.ok) return r;
      last = r;
      if (r.status !== 429 && r.status !== 503) return r; // non-transient: stop
    }
    return last;
  }

  try {
    const gRes = await callGemini();

    if (!gRes.ok) {
      const errText = await gRes.text();
      if (gRes.status === 429)
        return res.status(429).json({
          error:
            "Gemini's free tier is rate-limited right now (Google cut free limits in late 2025). " +
            "Wait a minute and retry. If it keeps happening, enable billing in Google AI Studio " +
            "to unlock Tier 1 — it has no minimum spend and clears this immediately.",
        });
      return res
        .status(502)
        .json({ error: "Gemini error", detail: errText.slice(0, 300) });
    }

    const data = await gRes.json();
    const text = extractText(data);
    const parsed = safeParse(text);

    if (!parsed)
      // model didn't return clean JSON — still give the user the prose
      return res.status(200).json({ answer: text || "No answer.", pois: [] });

    return res.status(200).json(parsed);
  } catch (e) {
    return res
      .status(500)
      .json({ error: "Request failed", detail: String(e).slice(0, 200) });
  }
}

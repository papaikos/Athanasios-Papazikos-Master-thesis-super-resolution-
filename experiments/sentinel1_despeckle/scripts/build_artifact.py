"""Build the self-contained HTML presentation ("SAR despeckling report"):
for each Sentinel-1 chip show Original | SAR2SAR (model 1) | SAR-BM3D (model 2),
with ENL metrics and per-panel speckle-suppression bars.
All images embedded as base64 (artifact CSP forbids external assets)."""
import os, json, base64, html, io, math
from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "out")
CHIPS = os.path.join(ROOT, "data", "chips")
metrics = json.load(open(os.path.join(OUT, "metrics.json")))
geo = {m["name"]: m for m in json.load(open(os.path.join(CHIPS, "manifest.json")))}

PRETTY = {
    "singapore_east": "Singapore Eastern Anchorage", "singapore_strait": "Singapore Strait",
    "gibraltar_strait": "Strait of Gibraltar", "fujairah_anchorage": "Fujairah Anchorage",
    "suez_gulf": "Gulf of Suez", "port_said_suez": "Port Said / Suez Approach",
    "piraeus_saronic": "Piraeus / Saronic Gulf", "hong_kong_anchorage": "Hong Kong Anchorage",
    "panama_pacific": "Panama Pacific Approach", "istanbul_marmara": "Sea of Marmara",
    "longbeach_la": "Long Beach / Los Angeles", "dover_channel": "Dover Strait",
    "port_klang_malacca": "Malacca Strait / Port Klang", "bosphorus_north": "Bosphorus (North)",
    "jebel_ali_dubai": "Jebel Ali, Dubai", "chittagong_anchor": "Chittagong Anchorage",
}
REGION = {
    "chittagong_anchor": "Bay of Bengal", "dover_channel": "English Channel",
    "port_klang_malacca": "Strait of Malacca", "istanbul_marmara": "Sea of Marmara",
    "bosphorus_north": "Black Sea approach", "panama_pacific": "Pacific / Panama",
    "singapore_east": "Singapore Strait", "singapore_strait": "Singapore Strait",
    "hong_kong_anchorage": "Pearl River Delta", "piraeus_saronic": "Aegean / Greece",
}

def b64(path, size=452):
    im = Image.open(path).convert("L").resize((size, size), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def coord(lat, lon):
    ns = "N" if lat >= 0 else "S"; ew = "E" if lon >= 0 else "W"
    return f"{abs(lat):.2f}&deg;{ns}&nbsp;{abs(lon):.2f}&deg;{ew}"

# log-scaled bar width for ENL (16..~600)
def barw(v):
    return max(4, min(100, 100 * math.log10(max(v, 1)) / math.log10(650)))

cards = []
for i, m in enumerate(metrics):
    name = m["name"]; g = geo.get(name, {})
    o = b64(os.path.join(OUT, "orig", f"{name}.png"))
    s = b64(os.path.join(OUT, "sar2sar", f"{name}.png"))
    b = b64(os.path.join(OUT, "merlin", f"{name}.png"))
    title = PRETTY.get(name, name.replace("_", " ").title())
    region = REGION.get(name, "")
    x1 = m["enl_sar2sar"] / max(m["enl_orig"], 1e-6)
    x2 = m["enl_bm3d"] / max(m["enl_orig"], 1e-6)
    cc = coord(g.get("lat", 0), g.get("lon", 0)) if g else ""
    cards.append(f"""
    <article class="rec">
      <header class="rec-h">
        <div class="rec-id"><span class="idx">{i+1:02d}</span><h2>{html.escape(title)}</h2>
          <span class="region">{html.escape(region)}</span></div>
        <dl class="tele">
          <div><dt>Scene</dt><dd class="mono">{html.escape(m['scene'])}</dd></div>
          <div><dt>Acquired</dt><dd class="mono">{html.escape(str(m['datetime'])[:10])}</dd></div>
          <div><dt>Position</dt><dd class="mono">{cc}</dd></div>
          <div><dt>Pol</dt><dd class="mono">{html.escape(str(m['band']).upper())}</dd></div>
          <div><dt>Vessels</dt><dd class="mono">~{m['n_ships']}</dd></div>
        </dl>
      </header>
      <div class="triptych">
        <figure class="pane orig">
          <img src="{o}" alt="Original Sentinel-1 GRD of {html.escape(title)}" loading="lazy">
          <figcaption><span class="lab">Original</span>
            <span class="enl">ENL <b>{m['enl_orig']:.0f}</b></span>
            <span class="bar"><i style="width:{barw(m['enl_orig']):.1f}%"></i></span></figcaption>
        </figure>
        <figure class="pane m1">
          <img src="{s}" alt="SAR2SAR despeckled {html.escape(title)}" loading="lazy">
          <figcaption><span class="lab">SAR2SAR &middot; deep</span>
            <span class="enl">ENL <b>{m['enl_sar2sar']:.0f}</b> <em>&times;{x1:.1f}</em></span>
            <span class="bar"><i style="width:{barw(m['enl_sar2sar']):.1f}%"></i></span></figcaption>
        </figure>
        <figure class="pane m2">
          <img src="{b}" alt="SAR-BM3D despeckled {html.escape(title)}" loading="lazy">
          <figcaption><span class="lab">SAR-BM3D &middot; classical</span>
            <span class="enl">ENL <b>{m['enl_bm3d']:.0f}</b> <em>&times;{x2:.1f}</em></span>
            <span class="bar"><i style="width:{barw(m['enl_bm3d']):.1f}%"></i></span></figcaption>
        </figure>
      </div>
    </article>""")

import statistics as st
avg_o = st.mean(m["enl_orig"] for m in metrics)
avg_s = st.mean(m["enl_sar2sar"] for m in metrics)
avg_b = st.mean(m["enl_bm3d"] for m in metrics)

doc = f"""<title>Sentinel-1 SAR Despeckling Report</title>
<style>
  :root {{
    --bg:#0b1013; --panel:#111a1f; --panel2:#0e161a; --line:#1f2d34; --line2:#2b3d45;
    --fg:#dfe9ec; --mut:#7d949c; --dim:#5a6f77;
    --accent:#43d6e6;           /* signal cyan */
    --m1:#eab54b;               /* SAR2SAR amber */
    --m2:#48d0dd;               /* SAR-BM3D cyan */
    --orig:#8aa0a8;
  }}
  @media (prefers-color-scheme: light) {{
    :root {{
      --bg:#eef2f1; --panel:#ffffff; --panel2:#f5f8f7; --line:#d8e0e0; --line2:#c2cfcf;
      --fg:#12222a; --mut:#5a6f77; --dim:#8497a0;
      --accent:#0e8ea0; --m1:#b7791f; --m2:#0e8ea0; --orig:#5f757d;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg:#0b1013; --panel:#111a1f; --panel2:#0e161a; --line:#1f2d34; --line2:#2b3d45;
    --fg:#dfe9ec; --mut:#7d949c; --dim:#5a6f77; --accent:#43d6e6;
    --m1:#eab54b; --m2:#48d0dd; --orig:#8aa0a8;
  }}
  :root[data-theme="light"] {{
    --bg:#eef2f1; --panel:#ffffff; --panel2:#f5f8f7; --line:#d8e0e0; --line2:#c2cfcf;
    --fg:#12222a; --mut:#5a6f77; --dim:#8497a0; --accent:#0e8ea0;
    --m1:#b7791f; --m2:#0e8ea0; --orig:#5f757d;
  }}
  * {{ box-sizing:border-box; }}
  html {{ -webkit-text-size-adjust:100%; }}
  body {{
    margin:0; background:var(--bg); color:var(--fg);
    font:15px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    --mono:ui-monospace,"SF Mono","Cascadia Code","JetBrains Mono",Menlo,Consolas,monospace;
  }}
  .mono {{ font-family:var(--mono); font-variant-numeric:tabular-nums; }}
  .wrap {{ max-width:1160px; margin:0 auto; padding:0 20px 90px; }}

  /* ---- masthead ---- */
  .mast {{ border-bottom:1px solid var(--line2); padding:34px 0 22px; margin-bottom:30px;
    background:
      repeating-linear-gradient(90deg,transparent 0 39px,color-mix(in oklab,var(--line) 55%,transparent) 39px 40px);
    background-clip:padding-box; }}
  .kicker {{ font-family:var(--mono); font-size:11.5px; letter-spacing:.28em; text-transform:uppercase;
    color:var(--accent); margin:0 0 12px; }}
  .mast h1 {{ font-size:clamp(28px,4.4vw,44px); line-height:1.04; letter-spacing:-.02em; margin:0 0 10px;
    text-wrap:balance; font-weight:640; }}
  .mast p.lede {{ margin:0; max-width:60ch; color:var(--mut); font-size:16px; }}
  .mast p.lede b {{ color:var(--fg); font-weight:600; }}

  /* ---- telemetry stat strip ---- */
  .strip {{ display:grid; grid-template-columns:repeat(4,1fr); gap:1px; margin:26px 0 4px;
    background:var(--line); border:1px solid var(--line); border-radius:10px; overflow:hidden; }}
  .strip .cell {{ background:var(--panel); padding:14px 16px; }}
  .strip .n {{ font-family:var(--mono); font-size:26px; font-variant-numeric:tabular-nums;
    letter-spacing:-.01em; line-height:1; }}
  .strip .n.a {{ color:var(--orig); }} .strip .n.b {{ color:var(--m1); }} .strip .n.c {{ color:var(--m2); }}
  .strip .k {{ font-size:11.5px; letter-spacing:.06em; text-transform:uppercase; color:var(--mut); margin-top:7px; }}

  /* ---- methods legend ---- */
  .legend {{ display:flex; flex-wrap:wrap; gap:10px 22px; margin:22px 0 34px; font-size:13.5px; color:var(--mut); }}
  .legend span {{ display:inline-flex; align-items:center; gap:8px; }}
  .swatch {{ width:11px; height:11px; border-radius:3px; box-shadow:0 0 0 1px color-mix(in oklab,#000 25%,transparent) inset; }}
  .sw-o {{ background:var(--orig); }} .sw-1 {{ background:var(--m1); }} .sw-2 {{ background:var(--m2); }}

  /* ---- primer ---- */
  .primer {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:34px; }}
  .primer .box {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px 18px; }}
  .primer h3 {{ margin:0 0 6px; font-size:13px; letter-spacing:.04em; text-transform:uppercase; color:var(--accent);
    font-family:var(--mono); }}
  .primer .box.m1 h3 {{ color:var(--m1); }} .primer .box.m2 h3 {{ color:var(--m2); }}
  .primer p {{ margin:0; font-size:14px; color:var(--fg); }}
  .primer p + p {{ margin-top:8px; color:var(--mut); font-size:13px; }}

  /* ---- records ---- */
  .rec {{ border-top:1px solid var(--line2); padding:26px 0 6px; }}
  .rec-h {{ display:flex; justify-content:space-between; align-items:flex-start; gap:22px;
    flex-wrap:wrap; margin-bottom:16px; }}
  .rec-id {{ display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; }}
  .rec-id .idx {{ font-family:var(--mono); font-size:13px; color:var(--accent); }}
  .rec-id h2 {{ margin:0; font-size:21px; letter-spacing:-.01em; font-weight:600; }}
  .rec-id .region {{ font-size:12.5px; color:var(--dim); font-family:var(--mono);
    border:1px solid var(--line2); padding:2px 8px; border-radius:20px; }}
  .tele {{ display:flex; flex-wrap:wrap; gap:4px 20px; margin:0; }}
  .tele div {{ display:flex; flex-direction:column; }}
  .tele dt {{ font-size:10px; letter-spacing:.12em; text-transform:uppercase; color:var(--dim); }}
  .tele dd {{ margin:0; font-size:12px; color:var(--mut); }}

  .triptych {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }}
  .pane {{ margin:0; }}
  .pane img {{ width:100%; height:auto; display:block; border-radius:9px;
    border:1px solid var(--line2); background:#000; }}
  .pane.m1 img {{ border-color:color-mix(in oklab,var(--m1) 45%,var(--line2)); }}
  .pane.m2 img {{ border-color:color-mix(in oklab,var(--m2) 45%,var(--line2)); }}
  figcaption {{ margin-top:9px; display:grid; grid-template-columns:1fr auto; align-items:center;
    gap:2px 10px; }}
  figcaption .lab {{ font-family:var(--mono); font-size:12px; letter-spacing:.02em; }}
  .orig figcaption .lab {{ color:var(--orig); }}
  .m1 figcaption .lab {{ color:var(--m1); }}
  .m2 figcaption .lab {{ color:var(--m2); }}
  figcaption .enl {{ font-family:var(--mono); font-size:12px; color:var(--mut);
    font-variant-numeric:tabular-nums; text-align:right; }}
  figcaption .enl b {{ color:var(--fg); font-weight:600; }}
  figcaption .enl em {{ font-style:normal; color:var(--accent); }}
  figcaption .bar {{ grid-column:1 / -1; height:3px; background:var(--line); border-radius:2px; overflow:hidden; margin-top:5px; }}
  figcaption .bar i {{ display:block; height:100%; border-radius:2px; }}
  .orig .bar i {{ background:var(--orig); }}
  .m1 .bar i {{ background:var(--m1); }}
  .m2 .bar i {{ background:var(--m2); }}

  footer {{ margin-top:44px; border-top:1px solid var(--line2); padding-top:20px;
    color:var(--dim); font-size:12.5px; line-height:1.7; }}
  footer b {{ color:var(--mut); font-weight:600; }}
  footer .mono {{ color:var(--mut); }}

  @media (max-width:860px) {{ .primer {{ grid-template-columns:1fr; }} .strip {{ grid-template-columns:repeat(2,1fr); }} }}
  @media (max-width:620px) {{ .triptych {{ grid-template-columns:1fr; }} }}
</style>

<div class="wrap">
  <div class="mast">
    <p class="kicker">Copernicus Sentinel-1 &middot; SAR Speckle Reduction</p>
    <h1>Ten radar scenes, two despecklers,<br>and the boats that survive both.</h1>
    <p class="lede">Every panel below is a <b>real Sentinel-1</b> Ground-Range-Detected amplitude chip pulled live from the Microsoft Planetary Computer archive, framed over busy shipping lanes so that <b>vessels read as bright point targets on dark water</b>. The task: strip the grainy <b>speckle</b> inherent to coherent radar without erasing the ships.</p>
    <div class="strip">
      <div class="cell"><div class="n">10</div><div class="k">Sentinel-1 scenes</div></div>
      <div class="cell"><div class="n a">{avg_o:.0f}</div><div class="k">Avg ENL &mdash; original</div></div>
      <div class="cell"><div class="n b">{avg_s:.0f}</div><div class="k">Avg ENL &mdash; SAR2SAR</div></div>
      <div class="cell"><div class="n c">{avg_b:.0f}</div><div class="k">Avg ENL &mdash; SAR-BM3D</div></div>
    </div>
    <div class="legend">
      <span><i class="swatch sw-o"></i> Original GRD</span>
      <span><i class="swatch sw-1"></i> Model&nbsp;1 &middot; SAR2SAR (deep, self-supervised)</span>
      <span><i class="swatch sw-2"></i> Model&nbsp;2 &middot; SAR-BM3D (classical, block-matching)</span>
    </div>
  </div>

  <div class="primer">
    <div class="box m1">
      <h3>Model 1 &mdash; SAR2SAR</h3>
      <p>A self-supervised convolutional network (Dalsasso&nbsp;et&nbsp;al., 2021), the reference <b>deep</b> despeckler built specifically for Sentinel-1 GRD. Pretrained weights via the <span class="mono">deepdespeckling</span> package.</p>
      <p>Behaviour: aggressive &mdash; it flattens homogeneous sea almost completely (ENL into the hundreds), at the cost of slightly softened coastlines.</p>
    </div>
    <div class="box m2">
      <h3>Model 2 &mdash; SAR-BM3D</h3>
      <p>Homomorphic <b>BM3D</b> &mdash; the landmark block-matching filter, run in the log domain so multiplicative speckle becomes additive noise it can collaboratively filter.</p>
      <p>Behaviour: conservative &mdash; strong, clean speckle reduction that keeps edges and point targets crisp, leaving more natural sea texture.</p>
    </div>
  </div>

  {''.join(cards)}

  <footer>
    <p><b>Metric &mdash; ENL (Equivalent Number of Looks)</b>, measured on the flattest water patch of each scene: ENL = (mean/std)&sup2;. Higher means smoother, more speckle removed; <span class="mono">&times;</span> is the gain over the original. All three panels of a scene share one identical dB stretch, so differences are the filters', not the display.</p>
    <p><b>Provenance.</b> Data: Copernicus Sentinel-1 (ESA), IW mode GRD, retrieved via the <span class="mono">Microsoft Planetary Computer</span> STAC API and located with each product's ground-control points. Chips are 512&times;512&nbsp;px, VV or HH polarization. Processing: <span class="mono">SAR2SAR</span> (deepdespeckling, pretrained) and homomorphic <span class="mono">SAR-BM3D</span> in log-amplitude. Contains modified Copernicus Sentinel data 2025.</p>
  </footer>
</div>
"""

path = os.path.join(ROOT, "presentation.html")
open(path, "w", encoding="utf-8").write(doc)
print("wrote", path, round(len(doc)/1e6, 2), "MB")

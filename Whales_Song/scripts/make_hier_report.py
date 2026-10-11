import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = r"c:\Users\hoang\project\stumpy\Whales_Song"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
import whale_mfcc as w  # noqa: E402

REPORTS = os.path.join(ROOT, "reports")
LEVELS = {"song": (600, 1), "theme": (95, 2), "phrase": (33, 10), "unit": (4.15, 20)}
LETTER = {"song": "S", "theme": "T", "phrase": "P", "unit": "U"}
IMAGE_WINDOW = (20175.0, 24675.0)
CLOCK0 = pd.Timestamp("2023-04-26 19:35:00")
PAGE_S = 900

TOP = {lvl: pd.read_csv(f"whale_top10_{lvl}.csv") for lvl in LEVELS}
C10, FRAME_RATE = w.load_or_compute_features(w.WAV_10H_PATH, "whale_mfcc_193500_100hz.npz")
DURATION = len(C10) / FRAME_RATE
ENERGY = w.rolling_mean(C10[:, 1], int(10 * FRAME_RATE))[:: int(FRAME_RATE)]
QUIET = np.nanpercentile(ENERGY, 10)
N_PAGES = int(np.ceil(DURATION / PAGE_S))

nb = json.load(open("Whale_Song_Motif_Hierarchy.ipynb", encoding="utf-8"))
CAPTIONS = {}
for c in nb["cells"]:
    src = "".join(c["source"])
    m = re.match(r"\*\*(Figure|Table) (\d+)\.", src)
    if c["cell_type"] == "markdown" and m:
        CAPTIONS[f"{m.group(1)} {m.group(2)}"] = src.strip()


def clock(s, fmt="%H:%M"):
    return (CLOCK0 + pd.to_timedelta(float(s), unit="s")).strftime(fmt)


def fmt_gap(s):
    if not np.isfinite(s):
        return "–"
    return f"{s:.0f} s" if s < 120 else f"{s / 60:.0f} min"


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(out)


def motif_table(level):
    tab, rows = TOP[level], []
    for mi, c in tab.groupby("motif"):
        s = np.sort(c["start_s"].to_numpy())
        seed = c.loc[c["r"].idxmax()]
        in_win = (c["start_s"] + c["length_s"] >= IMAGE_WINDOW[0]) & (c["start_s"] <= IMAGE_WINDOW[1])
        rows.append({
            "motif": f"{LETTER[level]}{mi}", "copies": len(c),
            "median r": f"{c['r'][c['r'] < 1].median():.3f}" if len(c) > 1 else "–",
            "min r": f"{c['r'].min():.3f}",
            "seed": f"{clock(seed['start_s'], '%H:%M:%S')} ({seed['start_s'] / 3600:.2f} h)",
            "first – last": f"{s[0] / 3600:.2f} – {s[-1] / 3600:.2f} h",
            "median gap": fmt_gap(np.median(np.diff(s)) if len(s) > 1 else np.nan),
            "in image window": int(in_win.sum()),
            "time covered": f"{c['length_s'].sum() / 60:.1f} min",
        })
    return pd.DataFrame(rows)


def level_table():
    rows = []
    for level in ["song", "theme", "phrase", "unit"]:
        tab = TOP[level]
        L, rate = LEVELS[level]
        counts = tab.groupby("motif").size()
        rows.append({
            "level": level, "length": f"{L:g} s", "rate": f"{rate} Hz", "m": int(round(L * rate)),
            "threshold θ": f"{tab['threshold'].iloc[0]:.3f}", "motifs": len(counts), "copies": len(tab),
            "copies per motif": ", ".join(map(str, counts.tolist())),
            "median copy r": f"{tab['r'][tab['r'] < 1].median():.3f}",
            "night covered": f"{tab['length_s'].sum() / DURATION:.0%}",
        })
    return pd.DataFrame(rows)


def page_index():
    rows = []
    for p in range(1, N_PAGES + 1):
        t0, t1 = (p - 1) * PAGE_S, min(p * PAGE_S, DURATION)
        counts, seeds, songs = [], [], []
        for level in ["song", "theme", "phrase", "unit"]:
            tab = TOP[level]
            on = tab[(tab["start_s"] >= t0) & (tab["start_s"] < t1)]
            counts.append(str(len(on)))
            seeds += [f"{LETTER[level]}{int(m)}" for m in on.loc[on["r"] >= 0.9999, "motif"]]
            if level == "song":
                songs += [f"S{int(r.motif)} c{int(r.copy)} at {clock(r.start_s)}" for r in on.itertuples()]
        notes = list(songs)
        if seeds:
            notes.append("seeds: " + ", ".join(seeds))
        for edge, name in zip(IMAGE_WINDOW, ("start", "end")):
            if t0 <= edge < t1:
                notes.append(f"original figure window {name} ({clock(edge, '%H:%M:%S')})")
        quiet = np.mean(ENERGY[int(t0):int(t1)] < QUIET)
        rows.append({
            "page": p, "clock time": f"{clock(t0)}–{clock(t1)}", "hours": f"{t0 / 3600:.1f}–{t1 / 3600:.1f}",
            "copies S / T / P / U": " / ".join(counts), "quiet": f"{quiet:.0%}", "notes": "; ".join(notes) or "–",
        })
    return pd.DataFrame(rows)


def fig(name, key):
    return f"![{key}](figures/p3_{name}.png)\n\n{CAPTIONS[key]}\n"


LEVEL_FIGS = {
    "unit": [("unit_night", 1), ("unit_window", 2), ("unit_zoom", 3)],
    "phrase": [("phrase_night", 4), ("phrase_window", 5), ("phrase_zoom", 6)],
    "theme": [("theme_night", 7), ("theme_window", 8)],
    "song": [("song_night", 9), ("song_window", 10)],
}
LEVEL_TEXT = {
    "unit": "Units are the shortest level (4.15 s at 20 Hz, m = 83). Their threshold is high (θ = 0.821) because short "
            "windows match by chance more easily, yet 1,157 copies pass it.",
    "phrase": "Phrases are 33 s windows at 10 Hz (m = 330). Four phrase motifs (P1–P4) account for 242 of the 263 "
              "copies.",
    "theme": "Themes are 95 s windows at 2 Hz (m = 190), the scale of the 70 s motifs of the original figure. Theme "
             "copies cover 31% of the night.",
    "song": "Songs are 600 s windows at 1 Hz (m = 600). Only two motifs reach the null threshold.",
}

parts = []
parts.append(f"""# Phase 3 report: top-10 motifs per level and a test of the song hierarchy

**Recording:** `SOCAL_H_76_230426_193500_df100.x.x.wav`, 10.0 hours from 2023-04-26 19:35:00
**Notebook:** `Whales_Song/Whale_Song_Motif_Hierarchy.ipynb` · **Plan:** `Whales_Song/Motif_Hierarchy_Plan.md`
**Timeline (stand-alone):** `Whales_Song/reports/Motif_Hierarchy_Timeline.pdf`

## Summary

- The analysis of the original figure ("Top-10 motifs with all copies, length 70 s") was repeated on the whole
  10-hour recording for each level of humpback song, at the representative lengths found in Phase 1 and 2:
  **unit 4.15 s, phrase 33 s, theme 95 s, song 600 s**. A copy is any non-overlapping window whose correlation with
  the motif's seed reaches the level's null threshold θ.
- **Motifs found:** 10 unit motifs (1,157 copies), 10 phrase motifs (263), 10 theme motifs (117) and 2 song motifs
  (9). One unit (U1, 650 copies), four phrases (P1–P4) and five themes (T1, T2, T3, T5, T9) recur all night.
- **Hierarchy test** (lower-level copies inside higher-level copies vs. 500 random time shifts):
  **phrase ⊃ unit, theme ⊃ phrase and theme ⊃ unit are strongly supported** (p ≤ 0.004 for containment, inside
  rate and consistency). **Song ⊃ theme is partly supported**: song copies always contain the same theme (T9) at
  the same place (consistency p = 0.006), but there are only 9 song copies. **Song ⊃ phrase is not supported**,
  because T9, the theme every song copy contains, contains none of the top-10 phrases.
- Section 5 is a **page-by-page 10-hour timeline** (page map + 40 pages of 15 minutes) showing every copy of every
  motif, so the nesting can be checked anywhere in the night.

## 1. Data and method

**Features.** MATLAB-style MFCCs (2 kHz, 12 bands, 0.1 s window, 10 ms hop = 100 Hz). The motif search uses
**c1** (MFCC coefficient 2, the coefficient of the original figure), block-averaged to each level's rate; the energy
panels use **c0** (log energy) averaged over a sliding 10 s window.

**Top-10 motifs with all copies.** For each level the matrix profile of c1 is computed once (stumpy `stump`,
exclusion zone = the full length). Motif 1's seed is the window with the best nearest-neighbour match; its copies
are all windows with correlation r ≥ θ to the seed (computed with `mass`), taken greedily without overlap. The
copies are then blanked and motif 2 is searched among the remaining windows, and so on, up to 10 motifs. Copies of
different motifs of one level never overlap.

**Threshold θ.** The top-decile nearest-neighbour correlation of a phase-randomized surrogate of c1 (same spectrum,
no repeated shapes), at the level's length and rate: a copy must be at least as similar to its seed as the best 10%
of chance matches.

**Hierarchy test.** A lower-level copy is *inside* a higher-level copy if ≥ 90% of it overlaps. For each pair of
levels: *contain rate* (fraction of higher copies containing ≥ 1 lower copy), *inside rate* (fraction of lower
copies inside some higher copy) and *consistency* (mean pairwise Jaccard similarity of the sets of lower motifs
inside the copies of the same higher motif). Null: all lower copies shifted together by one random circular offset,
500 times; one-sided p-values (minimum 1/501 ≈ 0.002).

### Notation

| symbol | meaning |
|---|---|
| c1, x(t) | MFCC coefficient 2, block-averaged to the level's analysis rate |
| c0 | MFCC log energy; its 10 s moving average is the energy panel |
| m | motif length in points = length (s) × rate (Hz) |
| r | Pearson correlation of two z-normalized windows, r = 1 − d² / (2m), d = matrix-profile distance |
| θ | null threshold of a level (minimum r of a copy) |
| S*k*, T*k*, P*k*, U*k* | song, theme, phrase, unit motif *k* (k = 1 found first) |
| c*j* | the *j*-th copy of a motif in time order |
| seed | the window a motif was grown from (r = 1); ▲ or black outline in the figures |
| A ⊃ B | a copy of B lies (≥ 90%) inside a copy of A |
| quiet | c0 10 s mean below its 10th percentile (grey) |
| time | hours since 19:35:00 on 2023-04-26, and clock time from the file timestamp |

Motif colours are shared by all levels (motif 1 dark red … motif 10 dark blue); the letter and the row tell the
levels apart. The original figure's recording (`011115+014845`) is the window **20,175–24,675 s
(01:11:15–02:26:15, 5.60–6.85 h)** of the 10-hour file and is shown separately at every level.

## 2. Motifs found at each level

**Table 1. Motifs found per level.** *m*: motif length in points; *threshold θ*: minimum r of a copy; *copies per
motif*: motif 1, 2, … in order; *median copy r*: median correlation of the copies with their seed (seeds excluded);
*night covered*: total duration of the copies divided by 10 hours.

{md_table(level_table())}

The per-level tables below list every motif. *median r* / *min r*: correlation of the copies with the seed;
*seed*: where the motif was found; *first – last*: first and last copy (hours since 19:35); *median gap*: median time
between consecutive copies; *in image window*: copies overlapping the original figure's 75 minutes.

Each level figure has three panels like the original: **(a)** c1 with the copies coloured by motif (▲ = seed;
box: m, θ and copy counts), **(b)** c0 energy with the 10th percentile of the plotted span (dashed red) and quiet
stretches in grey, **(c)** one row per motif with each copy as a rectangle, a dotted line joining copies of the
same motif, the seed outlined in black and, on the right, copies in view / in total and the median gap.
""")

for k, level in enumerate(["unit", "phrase", "theme", "song"], start=1):
    L, _ = LEVELS[level]
    parts.append(f"### 2.{k} {level.capitalize()}s ({L:g} s)\n\n{LEVEL_TEXT[level]}\n")
    parts.append(f"**Table 2.{k}. {level.capitalize()} motifs.**\n\n{md_table(motif_table(level))}\n")
    for name, n in LEVEL_FIGS[level]:
        parts.append(fig(name, f"Figure {n}"))

parts.append(f"""## 3. Does a song contain themes, a theme phrases, a phrase units?

### 3.1 All levels on one time axis

One row per level, each copy coloured by its motif number and labelled when there is room. Grey = quiet (whole-night
10th percentile); yellow = original figure window. In the zoomed views, dotted lines extend the start and end of
each song, theme and phrase copy down through the lower rows: a lower copy between two guides is inside that copy.

{fig("all_night", "Figure 11")}
{fig("all_window", "Figure 12")}
{fig("all_song1", "Figure 13")}
### 3.2 Nesting test

**Table 3. Hierarchy test** (observed / mean of 500 shifted nulls / one-sided p).

| pair | upper copies | lower copies | contain rate | inside rate | consistency |
|---|---|---|---|---|---|
| song ⊃ theme | 9 | 117 | 1.000 / 0.877 / p 0.269 | 0.154 / 0.131 / p 0.206 | **0.561 / 0.125 / p 0.006** |
| theme ⊃ phrase | 117 | 263 | **0.761 / 0.389 / p 0.004** | **0.529 / 0.224 / p 0.002** | **0.388 / 0.035 / p 0.004** |
| phrase ⊃ unit | 263 | 1157 | **0.939 / 0.425 / p 0.002** | **0.597 / 0.218 / p 0.002** | **0.425 / 0.064 / p 0.002** |
| song ⊃ phrase | 9 | 263 | 0.889 / 0.917 / p 0.832 | 0.080 / 0.142 / p 0.990 | 0.320 / 0.331 / p 0.557 |
| theme ⊃ unit | 117 | 1157 | **0.872 / 0.568 / p 0.002** | **0.583 / 0.299 / p 0.002** | **0.353 / 0.111 / p 0.004** |

Bold: p < 0.05.

{fig("null_hist", "Figure 14")}
### 3.3 Which lower motifs are inside which higher motifs

Rows: higher-level motifs (number of copies); columns: lower-level motifs. Cell text: % of the higher motif's copies
containing the lower motif. Colour: log2 enrichment over the shift null (red = more often than chance).

{fig("enrichment", "Figure 15")}
### 3.4 Barcodes: what is inside each copy

One row per copy of a higher motif (clock time, hours since 19:35, r with the seed; ★ = seed). Bottom axis: relative
position inside the copy; top axis: seconds. Blocks: lower-level copies inside, coloured and labelled by motif. If
the hierarchy holds, the blocks line up vertically.

{fig("barcode_song_theme", "Figure 16")}
{fig("barcode_theme_phrase", "Figure 17")}
{fig("barcode_phrase_unit", "Figure 18")}
## 4. Discussion and verdict

**Unit ⊂ phrase ⊂ theme: strongly supported.** Lower-level copies sit inside higher-level copies two to three times
more often than chance, and copies of the same higher motif contain the *same* lower motifs (consistency 6–10× the
null). The structure is concrete: P1 is U1 sung at four fixed positions ~7–8 s apart (Figure 18); T1 holds P1 or P2
in two fixed slots, T3 holds P3 then P3 or P4 (Figure 17); each theme has its own phrases (Figure 15b).

**Theme ⊂ song: partly supported.** Every song copy (S1 and S2) contains T9 at 0.5–0.78 of its length, usually
followed by T5, T1 or T7 at 0.75–0.98 (Figure 16), a significant consistency (p = 0.006). Containment itself is not
significant because themes cover 31% of the night and 600 s song copies almost always contain some theme. The
evidence rests on 9 song copies, and the first half of every song copy holds no top-10 motif at any level
(Figure 13).

**Song ⊃ phrase: not supported.** T9, the theme every song contains, contains none of the top-10 phrases
(Figures 15b, 17), so the chain song → theme → phrase breaks at T9.

**Caveats and next steps.**

- Single lengths per level (4.15, 33, 95, 600 s): a 33 s phrase motif fits only twice into a 95 s theme, and the true
  phrase may be shorter (Phase 1 found an ~11 s period).
- At 600 s, z-normalized correlation is dominated by slow loudness changes; detrending c1 or a lower threshold would
  give more song copies and a stronger test.
- Units are analysed at 20 Hz, so very short units (< 0.5 s) are smoothed out.
- T9 and the unexplained first half of the songs deserve their own search (e.g. phrase lengths of 10–20 s inside T9
  copies).

## 5. The 10-hour timeline

The following pages show **every copy of every motif over the whole night**. The first page is a page map (whole
night, one row per motif, page numbers on top); each following page covers **15 minutes**.
On each page:

- the thin top panel is c0 energy (10 s mean) with the whole-night 10th percentile (dashed red); grey = quiet;
- one row per motif, grouped by level (S1–S2, T1–T10, P1–P10, U1–U10; total copies in brackets on the left, level
  and length on the right);
- each copy is a rectangle at its true length; song and theme copies are labelled `c3 r=.64` (3rd copy in time
  order, r = 0.64 with the seed), phrase copies `c3`; seeds have a black outline and the label `seed`;
- dotted lines extend the start and end of every song, theme and phrase copy down through the lower rows, so a copy
  between two dotted lines of the same colour is inside that higher-level copy;
- black dashed lines mark the start and end of the original figure's window.

**Table 4. Page index.** Copies *starting* on each page per level (song / theme / phrase / unit), fraction of the
page below the whole-night quiet threshold, song copies, seeds and other landmarks.

{md_table(page_index())}

<!-- TIMELINE_START -->
""")
for p in range(N_PAGES + 1):
    label = "Page map" if p == 0 else f"Timeline page {p}"
    parts.append(f"![{label}](figures/p3_timeline_{p:02d}.png)\n")
parts.append("<!-- TIMELINE_END -->\n")

text = "\n".join(parts)
md_path = os.path.join(REPORTS, "Motif_Hierarchy_Report.md")
open(md_path, "w", encoding="utf-8").write(text)
print("wrote", md_path, len(text), "chars;", len(CAPTIONS), "captions")

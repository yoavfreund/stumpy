# Top-10 motifs per level and motif-hierarchy test (10 h file)

A new notebook reproduces the image-style analysis for each level (unit, phrase, theme, song) over the full 10-hour file: the top-10 motifs with all their copies, a c0 panel showing quiet periods, and a copy timeline. It then tests the nesting hypothesis (a song contains themes, which contain phrases, which contain units) with an aligned all-level chart, per-motif "barcode" charts and a containment test against a shift null.

## Tasks
- [x] Copy the plan to `Whales_Song/Motif_Hierarchy_Plan.md`
- [x] Add `level_threshold`, `top_motifs_with_copies`, `rolling_mean`, `containment`, `hierarchy_stats` to `whale_mfcc.py` and smoke-test them on the theme and song levels
- [x] Compute the thresholds and the top-10 motifs with copies for unit, phrase, theme and song (cached as CSV, matrix profiles as npz)
    10|- [x] Notebook part 1: image-style 3-panel figures per level (full 10 h, the image window 20,175–24,675 s, and a 5-minute zoom for units)
- [x] Notebook part 2: all-level aligned chart, barcode charts, enrichment heatmaps, shift-null test and verdict table
- [x] Execute the notebook end to end, review the figures, and write the discussion of whether the nesting hypothesis holds

## What the image shows (to reproduce per level)
1. **Series panel:** the c1 series ("MFCC coefficient 2") in grey, with the copies of the top-10 motifs coloured (one colour per motif, jet/turbo colormap). The title gives the length and the file.
2. **Energy panel:** c0 averaged over a sliding 10 s window, with a dashed line at its 10th percentile ("dips are quiet periods").
3. **Timeline panel:** rows M1..M10, labelled "(n copies)". Each copy is a coloured rectangle, and the copies of one motif are joined by a dotted line.

The image covers `SOCAL_H_76_230427_011115+014845` (75 min). In the 10 h file that is **20,175–24,675 s**, so every level also gets a zoom on exactly that window for direct comparison with the image.
    20|
## Settings (decided)
- **Feature:** motifs on **c1** (column 2), as in the image; the energy panel uses c0 (column 1). Data: `whale_mfcc_193500_100hz.npz`.
- **Motif lengths:**

| level | length | analysis rate | m (points) |
|---|---|---|---|
| unit | 4.15 s | 20 Hz | 83 |
| phrase | 33 s | 10 Hz | 330 |
| theme | 95 s | 2 Hz | 190 |
    30|| song | 600 s | 1 Hz | 600 |

  Units use 20 Hz rather than 100 Hz: a full matrix profile of 3.6 million points would take hours, while 720k points take about 13 min.
- **Copies:** all non-overlapping copies whose correlation with the motif reaches the null threshold, i.e. the top-decile r of a phase-randomized surrogate. The threshold is recomputed at each level's length *and rate*, with `length_scan` on that level's Phase 1 segments, because 20 Hz changes the unit null.
- **Number of motifs:** 10 per level (a parameter).

## Algorithm: top-k motifs with all copies (one level)

```mermaid
flowchart TD
    40|    mp["matrix profile of c1 at the level rate (cached npz)"] --> pick["seed = argmin P over unclaimed windows"]
    pick --> copies["mass(seed, T with claimed regions = NaN); greedy non-overlapping copies with r >= threshold"]
    copies --> enough{"at least 2 copies?"}
    enough -->|yes| claim["record motif; claim copies (NaN); set P = inf for windows overlapping them"]
    enough -->|no| drop["set P = inf at seed"]
    claim --> more{"k motifs found?"}
    drop --> more
    more -->|no| pick
    more -->|yes| doneNode["motif table: motif, copy, start, r"]
```
    50|
- Copies of different motifs never overlap within a level, as in the image. Levels are independent, so a theme copy may contain phrase copies.

## Hierarchy test (song contains themes, contains phrases, contains units)
- **Containment:** a lower-level copy is "inside" a higher-level copy if at least 90% of it overlaps. This is computed for song–theme, theme–phrase and phrase–unit, and also song–phrase and theme–unit.
- **Metrics per level pair:**
  - fraction of higher copies that contain at least one lower copy, and how much of each higher copy is covered;
  - **consistency:** for each higher motif, the mean pairwise Jaccard similarity of the sets of lower-motif labels inside its copies. If the thesis holds, copies of the same song contain the same themes.
  - **enrichment heatmap:** higher motif × lower motif, the fraction of copies that contain it, divided by the null expectation.
- **Null:** shift all lower-level copies circularly by one random offset (which keeps their internal spacing), 500 times. Report p-values for the containment rate and the consistency score.
    60|- **Charts:**
  1. **All-level chart:** four aligned rows (song, theme, phrase, unit) of coloured copy bands over the full 10 h, with quiet periods (c0 below its 10th percentile) shaded. Zooms: the image window, and the first copy of song motif 1.
  2. **Barcode charts:** for each higher motif, one row per copy, with the x axis as relative position inside the copy (0–1) and lower-motif copies drawn as coloured blocks. Vertically aligned blocks mean nested structure.
  3. **Verdict table:** per level pair, containment rate, consistency, null mean, and p-value.

## Code changes
- [whale_mfcc.py](whale_mfcc.py): add
  - `level_threshold(T_segments, m, rate)`, a thin wrapper around `length_scan`;
  - `top_motifs_with_copies(T, m, k, threshold, mp_cache)`, which returns a DataFrame `motif, copy, start, length, r`;
  - `rolling_mean(T, n)`;
    70|  - `containment(upper, lower, min_overlap=0.9)`;
  - `hierarchy_stats(upper, lower, n_null=500, seed=0)`.
  These reuse `block_average`, `stumpy.stump`, `stumpy.mass` and `phase_surrogate`.
- New notebook `Whales_Song/Whale_Song_Motif_Hierarchy.ipynb`:
  1. Introduction, settings table, and loading the features.
  2. Per level: threshold, top-10 motifs with copies (cached as `whale_top10_<level>.csv`), the image-style 3-panel figure for the full 10 h, and the same figure for the image window (units also get a 5-min zoom, since 4 s copies are tiny at 75 min).
  3. Hierarchy: the all-level chart, barcode charts, enrichment heatmaps, null histograms, and the verdict table.
  4. Discussion of whether the nesting holds, written after the results.
- Plotting 720k points per series uses a min/max envelope (1 s bins) for the full-night panels.

    80|## Runtime
About 25 min on the first run: unit matrix profile ~13 min, phrase ~4 min, null thresholds ~3 min, theme and song under 1 min. Everything is cached, so re-runs take well under a minute.

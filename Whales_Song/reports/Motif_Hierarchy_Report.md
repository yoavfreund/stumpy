# Phase 3 report: top-10 motifs per level and a test of the song hierarchy

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

| level | length | rate | m | threshold θ | motifs | copies | copies per motif | median copy r | night covered |
|---|---|---|---|---|---|---|---|---|---|
| song | 600 s | 1 Hz | 600 | 0.592 | 2 | 9 | 7, 2 | 0.667 | 15% |
| theme | 95 s | 2 Hz | 190 | 0.566 | 10 | 117 | 21, 17, 18, 11, 14, 7, 2, 6, 19, 2 | 0.639 | 31% |
| phrase | 33 s | 10 Hz | 330 | 0.577 | 10 | 263 | 77, 58, 62, 45, 4, 2, 5, 4, 3, 3 | 0.654 | 24% |
| unit | 4.15 s | 20 Hz | 83 | 0.821 | 10 | 1157 | 650, 14, 182, 91, 35, 132, 4, 4, 13, 32 | 0.858 | 13% |

The per-level tables below list every motif. *median r* / *min r*: correlation of the copies with the seed;
*seed*: where the motif was found; *first – last*: first and last copy (hours since 19:35); *median gap*: median time
between consecutive copies; *in image window*: copies overlapping the original figure's 75 minutes.

Each level figure has three panels like the original: **(a)** c1 with the copies coloured by motif (▲ = seed;
box: m, θ and copy counts), **(b)** c0 energy with the 10th percentile of the plotted span (dashed red) and quiet
stretches in grey, **(c)** one row per motif with each copy as a rectangle, a dotted line joining copies of the
same motif, the seed outlined in black and, on the right, copies in view / in total and the median gap.

### 2.1 Units (4.15 s)

Units are the shortest level (4.15 s at 20 Hz, m = 83). Their threshold is high (θ = 0.821) because short windows match by chance more easily, yet 1,157 copies pass it.

**Table 2.1. Unit motifs.**

| motif | copies | median r | min r | seed | first – last | median gap | in image window | time covered |
|---|---|---|---|---|---|---|---|---|
| U1 | 650 | 0.869 | 0.821 | 21:24:00 (1.82 h) | 0.41 – 9.59 h | 12 s | 104 | 45.0 min |
| U2 | 14 | 0.841 | 0.831 | 00:46:45 (5.20 h) | 4.48 – 5.65 h | 36 s | 1 | 1.0 min |
| U3 | 182 | 0.851 | 0.821 | 21:50:56 (2.27 h) | 0.38 – 9.78 h | 45 s | 15 | 12.6 min |
| U4 | 91 | 0.841 | 0.822 | 20:31:11 (0.94 h) | 0.51 – 9.62 h | 3 min | 12 | 6.3 min |
| U5 | 35 | 0.859 | 0.821 | 01:02:44 (5.46 h) | 2.80 – 6.87 h | 19 s | 0 | 2.4 min |
| U6 | 132 | 0.852 | 0.822 | 00:19:02 (4.73 h) | 0.53 – 9.81 h | 95 s | 15 | 9.1 min |
| U7 | 4 | 0.838 | 0.833 | 21:22:40 (1.79 h) | 1.79 – 9.19 h | 137 min | 0 | 0.3 min |
| U8 | 4 | 0.857 | 0.826 | 00:49:14 (5.24 h) | 5.24 – 5.96 h | 8 min | 1 | 0.3 min |
| U9 | 13 | 0.847 | 0.828 | 00:59:13 (5.40 h) | 4.60 – 5.47 h | 21 s | 0 | 0.9 min |
| U10 | 32 | 0.849 | 0.821 | 22:05:32 (2.51 h) | 2.03 – 9.20 h | 10 min | 13 | 2.2 min |

![Figure 1](figures/p3_unit_night.png)

**Figure 1. Unit motifs over the whole night.** 1,157 copies of 10 unit motifs, each 4.15 s long. **U1** alone
has 650 copies spread from 0.41 h to 9.59 h with a median gap of about 12 s: it is the unit that repeats inside
almost every phrase (see Figure 18). U3 (182), U6 (132) and U4 (91) also last the whole night. **U2, U5 and U9**
are local: their copies sit only between ~2.8 and ~6.9 h (U2 4.48–5.65 h, U9 4.60–5.47 h), so these units belong to
a part of the song that is sung only in that stretch of the night. Panel (b) shows the c0 energy rising through
the night; the quiet periods (grey) are mostly in the first hour and a half, before singing becomes continuous,
and there are no unit copies in the first 0.38 h.

![Figure 2](figures/p3_unit_window.png)

**Figure 2. Unit motifs in the original figure's 75 minutes** (01:11:15–02:26:15; x axis in seconds from its
start, as in the original). 161 unit copies fall in these 75 minutes: 104 of U1, and 12–15 each of U3, U4, U6
and U10. The U1 copies come in **clusters** (around 150–350 s, 1,000–1,100 s, 1,450–1,700 s, 2,100–2,350 s,
2,800–3,050 s, 3,650–3,800 s and 4,200–4,500 s) separated by several minutes with few U1 copies: U1 belongs to
particular phrases (P1–P4, Figure 18) and is absent while the whale sings other themes. The grey shading in (b) is
computed with this window's own 10th percentile, so it marks the relative pauses within the song (e.g. 600–750 s
and 2,400–2,700 s), and U1 clusters stop during them.

![Figure 3](figures/p3_unit_zoom.png)

**Figure 3. Five minutes with the most unit copies (01:58:11–02:03:11).** At this scale each 4.15 s copy is visible
as a coloured stretch of c1, and the repeated shape (a sharp c1 peak followed by a decay) can be compared copy by
copy. U1 recurs every ~10–20 s (18 copies in 5 minutes), with U3, U4, U6 and U10 in some of the gaps between U1
copies. During the quiet stretch at 55–95 s (grey in b) the U1 copies keep coming but other units are absent.
This regular repetition of a short unit is the pattern that "a phrase contains units" predicts.

### 2.2 Phrases (33 s)

Phrases are 33 s windows at 10 Hz (m = 330). Four phrase motifs (P1–P4) account for 242 of the 263 copies.

**Table 2.2. Phrase motifs.**

| motif | copies | median r | min r | seed | first – last | median gap | in image window | time covered |
|---|---|---|---|---|---|---|---|---|
| P1 | 77 | 0.626 | 0.579 | 20:29:58 (0.92 h) | 0.51 – 9.59 h | 4 min | 15 | 42.4 min |
| P2 | 58 | 0.665 | 0.582 | 22:00:37 (2.43 h) | 0.19 – 9.37 h | 7 min | 7 | 31.9 min |
| P3 | 62 | 0.668 | 0.578 | 22:22:26 (2.79 h) | 0.66 – 9.61 h | 9 min | 6 | 34.1 min |
| P4 | 45 | 0.668 | 0.578 | 20:17:07 (0.70 h) | 0.45 – 9.49 h | 10 min | 7 | 24.8 min |
| P5 | 4 | 0.694 | 0.602 | 21:23:59 (1.82 h) | 1.82 – 6.65 h | 58 min | 2 | 2.2 min |
| P6 | 2 | 0.813 | 0.626 | 01:14:02 (5.65 h) | 4.49 – 5.65 h | 70 min | 1 | 1.1 min |
| P7 | 5 | 0.590 | 0.587 | 00:21:27 (4.77 h) | 4.04 – 5.20 h | 13 min | 0 | 2.8 min |
| P8 | 4 | 0.630 | 0.599 | 22:20:43 (2.76 h) | 2.76 – 7.66 h | 83 min | 0 | 2.2 min |
| P9 | 3 | 0.598 | 0.580 | 23:51:55 (4.28 h) | 4.28 – 8.22 h | 118 min | 1 | 1.6 min |
| P10 | 3 | 0.684 | 0.597 | 00:44:00 (5.15 h) | 0.53 – 5.41 h | 146 min | 0 | 1.6 min |

![Figure 4](figures/p3_phrase_night.png)

**Figure 4. Phrase motifs over the whole night.** 263 copies of 10 phrase motifs, each 33 s long. Four phrases
dominate: **P1** (77 copies, median gap ~4 min), **P2** (58), **P3** (62) and **P4** (45), all present from the
first hour to the last. P5–P10 have 2–5 copies each and are mostly confined to the middle of the night
(P6 4.49–5.65 h, P7 4.04–5.20 h): rare variants or transitions.

![Figure 5](figures/p3_phrase_window.png)

**Figure 5. Phrase motifs in the original figure's window.** 39 phrase copies: P1 (15), P2 (7), P3 (6) and P4
(7) alternate across the 75 minutes, with single copies of P5, P6 (its seed, at ~180 s) and P9. Copies of the same
phrase often come in pairs or runs (P1 at ~1,000–1,060 s and ~4,250–4,400 s, P2 at ~1,500 s and ~2,050 s, P4 at
~300 s and ~1,150 s): a phrase is repeated before the song moves on, the defining property of a theme. The phrase
copies coincide with the U1 clusters of Figure 2.

![Figure 6](figures/p3_phrase_zoom.png)

**Figure 6. Ten minutes with the most phrase copies (01:27:53–01:37:53).** Each 33 s copy is now a full coloured
stretch of c1, and the repeated shape (a train of 4–6 sharp c1 peaks, i.e. units) is visible inside every copy. The
sequence is P1 P1 (back to back), P4 P4, a stretch without phrase copies (200–390 s), then P2 P1 P2 P2 P1: each
phrase is sung two or more times in a row, then the song switches. The median gap of the main phrases over the
night (4–10 min) is longer than here because after a run of a phrase the song moves on to other themes.

### 2.3 Themes (95 s)

Themes are 95 s windows at 2 Hz (m = 190), the scale of the 70 s motifs of the original figure. Theme copies cover 31% of the night.

**Table 2.3. Theme motifs.**

| motif | copies | median r | min r | seed | first – last | median gap | in image window | time covered |
|---|---|---|---|---|---|---|---|---|
| T1 | 21 | 0.667 | 0.580 | 21:59:47 (2.41 h) | 1.07 – 9.36 h | 17 min | 4 | 33.2 min |
| T2 | 17 | 0.622 | 0.573 | 22:03:12 (2.47 h) | 0.51 – 7.72 h | 16 min | 1 | 26.9 min |
| T3 | 18 | 0.685 | 0.568 | 22:16:36 (2.69 h) | 1.82 – 9.41 h | 13 min | 3 | 28.5 min |
| T4 | 11 | 0.641 | 0.576 | 02:37:34 (7.04 h) | 0.84 – 8.79 h | 49 min | 1 | 17.4 min |
| T5 | 14 | 0.653 | 0.572 | 21:22:27 (1.79 h) | 0.90 – 9.57 h | 21 min | 4 | 22.2 min |
| T6 | 7 | 0.594 | 0.575 | 01:38:30 (6.06 h) | 1.71 – 8.64 h | 27 min | 3 | 11.1 min |
| T7 | 2 | 0.663 | 0.663 | 21:47:17 (2.20 h) | 2.20 – 2.92 h | 43 min | 0 | 3.2 min |
| T8 | 6 | 0.648 | 0.581 | 02:49:01 (7.23 h) | 1.14 – 8.21 h | 64 min | 0 | 9.5 min |
| T9 | 19 | 0.610 | 0.575 | 00:29:28 (4.91 h) | 0.24 – 9.67 h | 27 min | 6 | 30.1 min |
| T10 | 2 | 0.582 | 0.582 | 20:37:25 (1.04 h) | 1.04 – 6.63 h | 335 min | 1 | 3.2 min |

![Figure 7](figures/p3_theme_night.png)

**Figure 7. Theme motifs over the whole night.** 117 copies of 10 theme motifs, each 95 s long, covering 31% of the
night. T1 (21 copies), T9 (19), T3 (18), T2 (17) and T5 (14) recur from the first to the last hour, i.e. the same
themes are sung all night. The median gap of the main themes is 13–27 min, close to the length of one song cycle.
T7 and T10 have only 2 copies.

![Figure 8](figures/p3_theme_window.png)

**Figure 8. Theme motifs in the original figure's window.** 23 theme copies: T9 (6), T1 and T5 (4 each), T3 and
T6 (3 each), and single copies of T2, T4 and T10. Each theme returns every ~15–25 minutes (about once per song
cycle). Some local orderings recur: T9 comes in pairs (~730/900 s and ~3,300/3,480 s), each pair followed within a
few minutes by T5 or T10 and then T3; T1 is followed by T6 twice (~1,500 → 1,680 s and ~2,900 → 3,050 s). The T6
seed is at ~1,680 s (black outline). The 95 s theme copies have the same scale and rhythm of repetition as the
70 s motifs of the original figure.

### 2.4 Songs (600 s)

Songs are 600 s windows at 1 Hz (m = 600). Only two motifs reach the null threshold.

**Table 2.4. Song motifs.**

| motif | copies | median r | min r | seed | first – last | median gap | in image window | time covered |
|---|---|---|---|---|---|---|---|---|
| S1 | 7 | 0.643 | 0.601 | 00:24:06 (4.82 h) | 4.82 – 8.84 h | 36 min | 4 | 70.0 min |
| S2 | 2 | 0.832 | 0.663 | 21:25:03 (1.83 h) | 1.83 – 2.07 h | 14 min | 0 | 20.0 min |

![Figure 9](figures/p3_song_night.png)

**Figure 9. Song motifs over the whole night.** Only two 600 s motifs reach the threshold: **S1** has 7 copies
between 4.82 h and 8.84 h (median gap ~36 min; four of them in the original figure's window) and **S2** has 2
copies at 1.83 h and 2.07 h, 14 minutes apart (about one song cycle). Ten-minute windows rarely correlate above
the null level because slow loudness changes dominate c1 at this scale (note the energy rise to its maximum at
~3.5 h in panel b), so the song level is the weakest part of the analysis.

![Figure 10](figures/p3_song_window.png)

**Figure 10. Song motifs in the original figure's window.** Four S1 copies overlap the 75 minutes. Compare with
the original figure, whose 70 s motifs repeat at the same rhythm: the 600 s S1 copies span several of those
shorter motifs.

## 3. Does a song contain themes, a theme phrases, a phrase units?

### 3.1 All levels on one time axis

One row per level, each copy coloured by its motif number and labelled when there is room. Grey = quiet (whole-night
10th percentile); yellow = original figure window. In the zoomed views, dotted lines extend the start and end of
each song, theme and phrase copy down through the lower rows: a lower copy between two guides is inside that copy.

![Figure 11](figures/p3_all_night.png)

**Figure 11. All four levels over the whole night.** At this scale the rows show *where* each level is active.
Units and phrases start after the first ~0.4 h, as soon as the energy rises out of the quiet first hour. Themes
cover 31% of the night and phrases 24%, so most phrase copies have a theme copy above them. The song row is sparse:
S2 twice near 2 h, S1 seven times between 4.8 h and 8.8 h. The page-by-page timeline at the end of the notebook
shows the same data at a scale where every copy can be read.

![Figure 12](figures/p3_all_window.png)

**Figure 12. All four levels in the original figure's 75 minutes.** The dotted guides from each song, theme and
phrase copy show nesting: most phrase copies fall between the guides of a theme copy, and the dense U1 clusters
sit between the guides of P1–P4 copies. Theme copies without phrase copies below them are mostly T9 (blue). Each
of the four S1 copies in view contains a T9 copy in its second half, which is what the consistency score in the
test below measures (Figure 16 shows all seven S1 copies aligned).

![Figure 13](figures/p3_all_song1.png)

**Figure 13. One song copy, zoomed (the S1 seed, 00:24:06–00:34:06, ± 1 minute).** The full 600 s copy is the top
row; the dotted lines mark its start and end and the start and end of every theme and phrase copy below it. The
nesting is visible in the second half: **T9** (~380–475 s) contains five U1 copies but no phrase copy, then **T5**
(~535–630 s) contains three back-to-back **P1** copies, each with three to four U1 copies inside. The first half of
the song (60–380 s) holds almost no motif copies at any level: that part of the song is not captured by the top-10
motifs of any level.

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

![Figure 14](figures/p3_null_hist.png)

**Figure 14. Observed vs. null.** Each column is a pair of levels, each row a score. When the red line lies far to
the right of the grey histogram, the lower level nests inside the higher level more than chance allows. For
**phrase ⊃ unit**, **theme ⊃ phrase** and **theme ⊃ unit** all three scores are beyond every null (p ≤ 0.004). For
**song ⊃ theme** only the consistency is (p = 0.006): song copies contain the *same* themes, but containing *some*
theme is not unusual because themes cover a third of the night. **Song ⊃ phrase** is at or below the null on all
three scores.

### 3.3 Which lower motifs are inside which higher motifs

Rows: higher-level motifs (number of copies); columns: lower-level motifs. Cell text: % of the higher motif's copies
containing the lower motif. Colour: log2 enrichment over the shift null (red = more often than chance).

![Figure 15](figures/p3_enrichment.png)

**Figure 15. Enrichment of lower motifs inside higher motifs.** (a) Song ⊃ theme: every copy of S1 and S2
contains T9 (100%); S1 copies also contain T5 (43%) or T1 (29%), S2 copies T5 or T7 (50% each). (b) Theme ⊃ phrase:
T1 contains P1 (52%) and P2 (67%); T3 and T4 contain P3 (72%, 82%) and P4; T5 contains P1 in every copy; T2 mixes
P2 and P4. Each theme thus has its *own* phrases, as a theme should. T9's row is empty: theme 9 contains none of the
top-10 phrases. (c) Phrase ⊃ unit: U1 is inside 67–87% of the copies of P1–P4, and each phrase adds its own
secondary units (P1: U3 38%, U6 29%; P3: U4 24%, U3 21%; P4: U6 38%, U4 27%).

### 3.4 Barcodes: what is inside each copy

One row per copy of a higher motif (clock time, hours since 19:35, r with the seed; ★ = seed). Bottom axis: relative
position inside the copy; top axis: seconds. Blocks: lower-level copies inside, coloured and labelled by motif. If
the hierarchy holds, the blocks line up vertically.

![Figure 16](figures/p3_barcode_song_theme.png)

**Figure 16. Themes inside songs.** Each row is one song copy, each block a theme copy inside it. **Every S1
copy has a T9 block at 0.51–0.75 of the song (~310–450 s)**; in five of the seven it is followed by T5 (3 copies)
or T1 (2 copies) at 0.75–0.98, and two copies have a second T9 earlier, at ~0.3–0.47. S2's two copies have the same
layout (T9 at ~0.62–0.78, then T5 or T7). This vertical alignment across copies recorded hours apart is the evidence
for "a song contains themes in a fixed order". The first half of every song copy holds no theme copy.

![Figure 17](figures/p3_barcode_theme_phrase.png)

**Figure 17. Phrases inside themes.** A 95 s theme holds at most two 33 s phrase copies (they cannot overlap). In
T1 the two slots (~0.05–0.5 and ~0.5–0.95 of the theme) hold P1 or P2, often P1 then P1 or P2 then P2. T2 usually
holds P4 in its first half and P2 or P1 in its second. In T3 the first slot holds P3 and the second P3 or P4. Blocks
line up in the same slots from copy to copy. T9 (19 copies) has no phrase blocks at all: it is made of something
the top-10 phrase motifs do not capture.

![Figure 18](figures/p3_barcode_phrase_unit.png)

**Figure 18. Units inside phrases.** The clearest nesting of all. P1 copies contain U1 at up to four fixed
positions about 7–8 s apart (~0.1, ~0.4, ~0.6 and ~0.85 of the phrase, i.e. ~3, 13, 20 and 28 s), so a P1 phrase
is essentially U1 sung four times with other sounds (often U3 or U6) in between. P2 contains U1 at ~0.2, ~0.5 and
~0.9; P3 at ~0.05, ~0.4, ~0.65 and ~0.9; P4 combines U1 at ~0.3, ~0.6 and ~0.85 with its own units U6 and U4. The
columns of blocks are straight from the top row to the bottom row, i.e. from the first to the last hour.

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

| page | clock time | hours | copies S / T / P / U | quiet | notes |
|---|---|---|---|---|---|
| 1 | 19:35–19:50 | 0.0–0.2 | 0 / 1 / 1 / 0 | 100% | – |
| 2 | 19:50–20:05 | 0.2–0.5 | 0 / 0 / 1 / 11 | 100% | – |
| 3 | 20:05–20:20 | 0.5–0.8 | 0 / 3 / 9 / 43 | 92% | seeds: P4 |
| 4 | 20:20–20:35 | 0.8–1.0 | 0 / 3 / 8 / 46 | 37% | seeds: P1, U4 |
| 5 | 20:35–20:50 | 1.0–1.2 | 0 / 5 / 9 / 32 | 45% | seeds: T10 |
| 6 | 20:50–21:05 | 1.2–1.5 | 0 / 3 / 6 / 19 | 24% | – |
| 7 | 21:05–21:20 | 1.5–1.8 | 0 / 5 / 9 / 41 | 1% | – |
| 8 | 21:20–21:35 | 1.8–2.0 | 1 / 4 / 9 / 49 | 0% | S2 c1 at 21:25; seeds: S2, T5, P5, U1, U7 |
| 9 | 21:35–21:50 | 2.0–2.2 | 1 / 4 / 5 / 34 | 0% | S2 c2 at 21:39; seeds: T7 |
| 10 | 21:50–22:05 | 2.2–2.5 | 0 / 3 / 10 / 46 | 0% | seeds: T1, T2, P2, U3 |
| 11 | 22:05–22:20 | 2.5–2.8 | 0 / 2 / 6 / 41 | 0% | seeds: T3, U10 |
| 12 | 22:20–22:35 | 2.8–3.0 | 0 / 4 / 14 / 69 | 0% | seeds: P3, P8 |
| 13 | 22:35–22:50 | 3.0–3.2 | 0 / 5 / 9 / 34 | 0% | – |
| 14 | 22:50–23:05 | 3.2–3.5 | 0 / 2 / 4 / 5 | 0% | – |
| 15 | 23:05–23:20 | 3.5–3.8 | 0 / 2 / 6 / 14 | 0% | – |
| 16 | 23:20–23:35 | 3.8–4.0 | 0 / 1 / 3 / 23 | 0% | – |
| 17 | 23:35–23:50 | 4.0–4.2 | 0 / 3 / 8 / 27 | 0% | – |
| 18 | 23:50–00:05 | 4.2–4.5 | 0 / 3 / 10 / 35 | 0% | seeds: P9 |
| 19 | 00:05–00:20 | 4.5–4.8 | 0 / 3 / 9 / 50 | 0% | seeds: U6 |
| 20 | 00:20–00:35 | 4.8–5.0 | 1 / 3 / 7 / 38 | 0% | S1 c1 at 00:24; seeds: S1, T9, P7 |
| 21 | 00:35–00:50 | 5.0–5.2 | 0 / 1 / 4 / 45 | 0% | seeds: P10, U2, U8 |
| 22 | 00:50–01:05 | 5.2–5.5 | 0 / 1 / 5 / 42 | 0% | seeds: U5, U9 |
| 23 | 01:05–01:20 | 5.5–5.8 | 2 / 3 / 5 / 23 | 0% | S1 c2 at 01:05; S1 c3 at 01:19; seeds: P6; original figure window start (01:11:15) |
| 24 | 01:20–01:35 | 5.8–6.0 | 0 / 5 / 5 / 15 | 0% | – |
| 25 | 01:35–01:50 | 6.0–6.2 | 0 / 5 / 13 / 43 | 0% | seeds: T6 |
| 26 | 01:50–02:05 | 6.2–6.5 | 2 / 3 / 5 / 32 | 0% | S1 c4 at 01:50; S1 c5 at 02:03 |
| 27 | 02:05–02:20 | 6.5–6.8 | 0 / 4 / 4 / 20 | 0% | – |
| 28 | 02:20–02:35 | 6.8–7.0 | 0 / 5 / 11 / 47 | 0% | original figure window end (02:26:15) |
| 29 | 02:35–02:50 | 7.0–7.2 | 0 / 5 / 10 / 22 | 0% | seeds: T4, T8 |
| 30 | 02:50–03:05 | 7.2–7.5 | 0 / 5 / 11 / 35 | 0% | – |
| 31 | 03:05–03:20 | 7.5–7.8 | 1 / 3 / 10 / 43 | 0% | S1 c6 at 03:06 |
| 32 | 03:20–03:35 | 7.8–8.0 | 0 / 1 / 1 / 2 | 0% | – |
| 33 | 03:35–03:50 | 8.0–8.2 | 0 / 2 / 3 / 6 | 0% | – |
| 34 | 03:50–04:05 | 8.2–8.5 | 0 / 1 / 2 / 6 | 0% | – |
| 35 | 04:05–04:20 | 8.5–8.8 | 0 / 3 / 6 / 32 | 0% | – |
| 36 | 04:20–04:35 | 8.8–9.0 | 1 / 4 / 6 / 32 | 0% | S1 c7 at 04:25 |
| 37 | 04:35–04:50 | 9.0–9.2 | 0 / 3 / 9 / 31 | 0% | – |
| 38 | 04:50–05:05 | 9.2–9.5 | 0 / 2 / 7 / 17 | 0% | – |
| 39 | 05:05–05:20 | 9.5–9.8 | 0 / 2 / 3 / 5 | 0% | – |
| 40 | 05:20–05:34 | 9.8–10.0 | 0 / 0 / 0 / 2 | 0% | – |

<!-- TIMELINE_START -->

![Page map](figures/p3_timeline_00.png)

![Timeline page 1](figures/p3_timeline_01.png)

![Timeline page 2](figures/p3_timeline_02.png)

![Timeline page 3](figures/p3_timeline_03.png)

![Timeline page 4](figures/p3_timeline_04.png)

![Timeline page 5](figures/p3_timeline_05.png)

![Timeline page 6](figures/p3_timeline_06.png)

![Timeline page 7](figures/p3_timeline_07.png)

![Timeline page 8](figures/p3_timeline_08.png)

![Timeline page 9](figures/p3_timeline_09.png)

![Timeline page 10](figures/p3_timeline_10.png)

![Timeline page 11](figures/p3_timeline_11.png)

![Timeline page 12](figures/p3_timeline_12.png)

![Timeline page 13](figures/p3_timeline_13.png)

![Timeline page 14](figures/p3_timeline_14.png)

![Timeline page 15](figures/p3_timeline_15.png)

![Timeline page 16](figures/p3_timeline_16.png)

![Timeline page 17](figures/p3_timeline_17.png)

![Timeline page 18](figures/p3_timeline_18.png)

![Timeline page 19](figures/p3_timeline_19.png)

![Timeline page 20](figures/p3_timeline_20.png)

![Timeline page 21](figures/p3_timeline_21.png)

![Timeline page 22](figures/p3_timeline_22.png)

![Timeline page 23](figures/p3_timeline_23.png)

![Timeline page 24](figures/p3_timeline_24.png)

![Timeline page 25](figures/p3_timeline_25.png)

![Timeline page 26](figures/p3_timeline_26.png)

![Timeline page 27](figures/p3_timeline_27.png)

![Timeline page 28](figures/p3_timeline_28.png)

![Timeline page 29](figures/p3_timeline_29.png)

![Timeline page 30](figures/p3_timeline_30.png)

![Timeline page 31](figures/p3_timeline_31.png)

![Timeline page 32](figures/p3_timeline_32.png)

![Timeline page 33](figures/p3_timeline_33.png)

![Timeline page 34](figures/p3_timeline_34.png)

![Timeline page 35](figures/p3_timeline_35.png)

![Timeline page 36](figures/p3_timeline_36.png)

![Timeline page 37](figures/p3_timeline_37.png)

![Timeline page 38](figures/p3_timeline_38.png)

![Timeline page 39](figures/p3_timeline_39.png)

![Timeline page 40](figures/p3_timeline_40.png)

<!-- TIMELINE_END -->

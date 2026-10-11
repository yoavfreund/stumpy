import base64, json, os

BASE = r"c:\Users\hoang\project\stumpy\Whales_Song"
OUT = os.path.join(BASE, "reports", "figures")
os.makedirs(OUT, exist_ok=True)

names = {
    "Whale_Song_Length_Range_Experiment.ipynb": [
        "p1_unit_scan", "p1_unit_periods", "p1_phrase_scan", "p1_phrase_periods", "p1_theme_scan",
        "p1_theme_periods", "p1_song_regularity", "p1_song_scan", "p1_song_periods",
    ],
    "Whale_Song_Multiscale_Forward_Motifs.ipynb": [
        "p2_lengths", "p2_timeline_c0", "p2_timeline_c1", "p2_overlap_c0", "p2_overlap_c1",
        "p2_overlays_c0", "p2_overlays_c1", "p2_excess_hist",
    ],
}
for nb_name, labels in names.items():
    nb = json.load(open(os.path.join(BASE, nb_name), encoding="utf-8"))
    imgs = [o["data"]["image/png"] for c in nb["cells"] if c["cell_type"] == "code"
            for o in c.get("outputs", []) if "data" in o and "image/png" in o["data"]]
    assert len(imgs) == len(labels), (nb_name, len(imgs))
    for label, img in zip(labels, imgs):
        open(os.path.join(OUT, label + ".png"), "wb").write(base64.b64decode(img))
print(sorted(os.listdir(OUT)))

import pandas as pd
pd.set_option("display.width", 200)
for band in ("unit", "phrase", "theme", "song"):
    for f in ("c0", "c1"):
        s = pd.read_csv(os.path.join(BASE, f"whale_scan_{band}_{f}.csv"))
        print(band, f)
        print(s[["length_s", "best_r", "top_r", "null_top_r", "excess"]].round(3).T.to_string(header=False))

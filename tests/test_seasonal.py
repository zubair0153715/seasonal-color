"""Tests: synthetic skin swatches -> undertone/depth/season classification."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image

from seasonal import analyze, classify_undertone, classify_depth, season_for, PALETTES

def swatch(rgb, path, size=200):
    Image.new("RGB", (size, size), rgb).save(path)

tmp = tempfile.mkdtemp()

# warm peach skin -> warm undertone
p1 = os.path.join(tmp, "warm.png")
swatch((235, 180, 140), p1)
r1 = analyze(p1)
print(r1["undertone"], r1["depth"], r1["season"])
assert r1["undertone"] == "warm", r1
assert r1["season"] in PALETTES

# cool pinkish skin -> cool undertone
p2 = os.path.join(tmp, "cool.png")
swatch((235, 190, 200), p2)
r2 = analyze(p2)
print(r2["undertone"], r2["depth"], r2["season"])
assert r2["undertone"] == "cool", r2

# deep skin -> deep depth
p3 = os.path.join(tmp, "deep.png")
swatch((90, 60, 50), p3)
r3 = analyze(p3)
print(r3["undertone"], r3["depth"], r3["season"])
assert r3["depth"] == "deep", r3

# unit checks
assert classify_undertone((240, 180, 130), 25.0) == "warm"
assert classify_undertone((230, 185, 205), 346.0) == "cool"
assert classify_depth((250, 230, 220)) == "light"
assert classify_depth((80, 55, 45)) == "deep"
assert season_for("warm", "light") == "Light Spring"
assert season_for("cool", "deep") == "Deep Winter"
assert len(PALETTES) == 12 and all(len(v) == 8 for v in PALETTES.values())

# no-skin image -> graceful error
p4 = os.path.join(tmp, "noskin.png")
swatch((10, 10, 200), p4)  # pure blue, no skin
r4 = analyze(p4)
assert "error" in r4, r4

print("ALL SEASONAL-COLOR TESTS PASSED")

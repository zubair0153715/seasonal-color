# seasonal-color

Selfie → skin undertone → **12-season color palette** (PIL + numpy, no ML model, MIT).

The personal-color-analysis corner of GitHub is a dead zone (best repo: 14 stars,
abandoned 2020). This fills it with a lightweight, explainable library.

## How it works

1. HSV skin segmentation (classic chroma-range mask)
2. Circular mean of skin hue → undertone (orange hues = warm, pink hues = cool)
3. Lightness → depth (light / medium / deep)
4. (undertone, depth) → one of 12 seasons → curated 8-color palette

## Usage

```bash
pip install pillow numpy
python -c "
from seasonal import analyze
import json
print(json.dumps(analyze('selfie.jpg'), indent=1))
"
```

Output: `season`, `undertone`, `depth`, `skin_rgb`, `palette`, `best`, `avoid`.

## Test

```bash
python tests/test_seasonal.py   # synthetic swatches: warm/cool/deep + edge cases
```

## License

MIT — see LICENSE.

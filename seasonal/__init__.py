"""seasonal-color: selfie -> skin undertone -> 12-season color palette.

No ML model needed: classic HSV skin segmentation + undertone heuristics.
Dependencies: pillow, numpy.

Usage:
    from seasonal import analyze
    result = analyze("selfie.jpg")
    # result: {season, undertone, depth, palette:[hex...], best:[hex...], avoid:[hex...]}
"""
import colorsys

import numpy as np
from PIL import Image

# 12-season palettes (hex). Curated sets following standard seasonal theory.
PALETTES = {
    "Light Spring":  ["#F9E2D2", "#F5CBA7", "#FADBD8", "#AED6F1", "#A9DFBF", "#F9E79F", "#D5F5E3", "#FCF3CF"],
    "Warm Spring":   ["#F0B27A", "#E59866", "#F5CBA7", "#F9E79F", "#ABEBC6", "#AED6F1", "#FAD7A0", "#EDBB99"],
    "Clear Spring":  ["#FF6F61", "#FFA07A", "#FFD700", "#40E0D0", "#00BFFF", "#7CFC00", "#FF69B4", "#FFFFFF"],
    "Light Summer":  ["#F5CBA7", "#FADBD8", "#D6EAF8", "#D5F5E3", "#E8DAEF", "#FCF3CF", "#D1F2EB", "#FEF9E7"],
    "Cool Summer":   ["#AED6F1", "#A9CCE3", "#D6DBDF", "#D5DBDB", "#E8DAEF", "#FADBD8", "#A9DFBF", "#D4EFDF"],
    "Soft Summer":   ["#A9CCE3", "#AEB6BF", "#B2BABB", "#C39BD3", "#A9DFBF", "#F5CBA7", "#D7BDE2", "#D6DBDF"],
    "Soft Autumn":   ["#C19A6B", "#A67B5B", "#8B6F47", "#7D6608", "#6E5100", "#935116", "#7E5109", "#B7950B"],
    "Warm Autumn":   ["#B9770E", "#935116", "#7E5109", "#784212", "#6E2C00", "#7D6608", "#9A7D0A", "#B7950B"],
    "Deep Autumn":   ["#7E5109", "#784212", "#6E2C00", "#641E16", "#4A235A", "#1A5276", "#0E6655", "#7D6608"],
    "Deep Winter":   ["#000000", "#2C3E50", "#4A235A", "#641E16", "#7B241C", "#1A5276", "#0E6251", "#FFFFFF"],
    "Cool Winter":   ["#2C3E50", "#1A5276", "#4A235A", "#6C3483", "#1F618D", "#0E6655", "#FFFFFF", "#000000"],
    "Clear Winter":  ["#000000", "#FFFFFF", "#FF0000", "#0000FF", "#FF00FF", "#00FFFF", "#FFFF00", "#800080"],
}

def _skin_mask(hsv):
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return (h <= 25) & (s >= 30) & (s <= 170) & (v >= 60)

def sample_skin(image_path, max_side=400):
    """Return (median skin RGB, mean skin hue in degrees) or (None, None)."""
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((max_side, max_side))
    arr = np.asarray(img)
    hsv = np.asarray(img.convert("HSV")).astype(int)
    # PIL HSV: H 0-255 maps 0-360; scale to degrees
    h = hsv[:, :, 0] * 360 / 255
    s = hsv[:, :, 1] * 100 / 255
    v = hsv[:, :, 2] * 100 / 255
    mask = ((h <= 30) | (h >= 330)) & (s >= 12) & (v >= 20)
    skin = arr[mask]
    if len(skin) < 50:
        return None, None
    hues = h[mask]
    ang = np.deg2rad(hues)  # circular mean of hue
    mean_h = float(np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360)
    return np.median(skin, axis=0), mean_h

def classify_undertone(skin_rgb, mean_hue=None):
    """warm / cool / neutral. Hue-based: orange hues = warm, pink hues = cool."""
    if mean_hue is not None:
        if mean_hue >= 333:
            return "cool"
        if mean_hue <= 27:
            return "warm"
        return "neutral"
    r, g, b = [float(x) for x in skin_rgb]
    warmth = (r - b) / 255.0            # >0 leans warm
    yellowness = (r + g - 2 * b) / 510.0
    score = 0.6 * warmth + 0.4 * yellowness
    if score > 0.06:
        return "warm"
    if score < -0.02:
        return "cool"
    return "neutral"

def classify_depth(skin_rgb):
    r, g, b = [float(x) / 255 for x in skin_rgb]
    _, l, _ = colorsys.rgb_to_hls(r, g, b)
    if l > 0.72:
        return "light"
    if l < 0.45:
        return "deep"
    return "medium"

def season_for(undertone, depth):
    table = {
        ("warm", "light"): "Light Spring",
        ("warm", "medium"): "Warm Spring",
        ("warm", "deep"): "Deep Autumn",
        ("cool", "light"): "Light Summer",
        ("cool", "medium"): "Cool Summer",
        ("cool", "deep"): "Deep Winter",
        ("neutral", "light"): "Light Spring",
        ("neutral", "medium"): "Soft Summer",
        ("neutral", "deep"): "Soft Autumn",
    }
    return table[(undertone, depth)]

def analyze(image_path):
    try:
        skin, mean_hue = sample_skin(image_path)
    except (OSError, ValueError) as e:
        return {"error": f"cannot read image: {e}"}
    if skin is None:
        return {"error": "no skin detected — use a clear, front-lit selfie"}
    undertone = classify_undertone(skin, mean_hue)
    depth = classify_depth(skin)
    season = season_for(undertone, depth)
    palette = PALETTES[season]
    return {
        "season": season,
        "undertone": undertone,
        "depth": depth,
        "skin_rgb": [int(x) for x in skin],
        "palette": palette,
        "best": palette[:4],
        "avoid": palette[-2:],
    }

if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(analyze(sys.argv[1]), indent=1))

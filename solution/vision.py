"""Q2/Q5: actual OpenCV checks and local Tesseract LSTM extraction."""
import copy
import hashlib
import time
from collections import OrderedDict
from pathlib import Path
import cv2
import numpy as np
import pytesseract
from .records import extract_record


def grayscale(image):
    if image is None or not isinstance(image, np.ndarray) or image.size == 0 or image.dtype != np.uint8:
        raise ValueError("Nonempty uint8 image required")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] in (3, 4):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
    raise ValueError("Expected grayscale, BGR or BGRA image")


def check_quality(image, blur_threshold=100., min_brightness=70., max_brightness=250.):
    gray = grayscale(image)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(np.percentile(gray, 99) - np.percentile(gray, 1))
    reasons = []
    if blur < blur_threshold:
        reasons.append("blurred or no visible text")
    if brightness < min_brightness:
        reasons.append("underexposed")
    if brightness > max_brightness:
        reasons.append("overexposed or mostly blank")
    if contrast < 30:
        reasons.append("low contrast")
    return {"is_usable": not reasons, "blur_score": round(blur, 3),
            "brightness": round(brightness, 3), "contrast": round(contrast, 3),
            "reason": "; ".join(reasons) if reasons else "Passed heuristic quality checks"}


def make_samples(directory):
    """Reproducible synthetic fixtures, not photographs or handwriting benchmarks."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    image = np.full((560, 1200), 245, np.uint8)
    from .records import SAMPLE_TEXT
    for i, line in enumerate(SAMPLE_TEXT.splitlines()):
        cv2.putText(image, line, (55, 95 + i * 90), cv2.FONT_HERSHEY_SIMPLEX, 1.25, 15, 2, cv2.LINE_AA)
    samples = {"clear": image, "blurred": cv2.GaussianBlur(image, (35, 35), 10),
               "dark": (image.astype(float) * .18).astype(np.uint8),
               "rotated": cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
               "blank": np.full_like(image, 245)}
    for name, data in samples.items():
        cv2.imwrite(str(directory / f"{name}.png"), data)
    return samples


def preprocess(image):
    gray = grayscale(image)
    h, w = gray.shape
    if max(h, w) > 2200:
        gray = cv2.resize(gray, None, fx=2200/max(h, w), fy=2200/max(h, w), interpolation=cv2.INTER_AREA)
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def run_tesseract(image, psm=6):
    data = pytesseract.image_to_data(image, lang="eng", config=f"--oem 1 --psm {psm}",
                                    output_type=pytesseract.Output.DICT, timeout=20)
    lines, words = OrderedDict(), []
    for i, text in enumerate(data["text"]):
        if text.strip() and float(data["conf"][i]) >= 0:
            key = (data["page_num"][i], data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append(text)
            words.append({"text": text, "confidence": float(data["conf"][i]) / 100,
                          "box": [int(data[k][i]) for k in ("left", "top", "width", "height")]})
    return {"text": "\n".join(" ".join(line) for line in lines.values()), "words": words,
            "mean_confidence": sum(w["confidence"] for w in words) / len(words) if words else 0.}


class OCRPipeline:
    """Bounded per-instance memory cache. No remote API or patient data upload."""
    def __init__(self, max_cache=32):
        if max_cache < 0:
            raise ValueError("max_cache must be nonnegative")
        self.cache = OrderedDict()
        self.max_cache = max_cache

    def process(self, image, project="demo"):
        start = time.perf_counter()
        gray = grayscale(image)
        if gray.size > 25_000_000:
            raise ValueError("Image exceeds 25 million pixels")
        key = (project, gray.shape, hashlib.sha256(gray.tobytes()).hexdigest(), "v1")
        if key in self.cache:
            out = copy.deepcopy(self.cache[key])
            self.cache.move_to_end(key)
            out.update(cache_hit=True, latency_ms=round((time.perf_counter()-start)*1000, 3))
            return out
        quality = check_quality(gray)
        out = {"quality": quality, "cache_hit": False, "api_cost_usd": 0., "attempts": []}
        # Exposure failures can conceal evidence; ask for recapture rather than fabricate it.
        if not quality["is_usable"]:
            out.update(route="RECAPTURE", extraction=None, structured=None)
        else:
            prepared = preprocess(gray)
            candidates = []
            # One fast pass; bounded rotation fallback only if schema/quality fails.
            for angle in (0, 90, 180, 270):
                candidate = np.rot90(prepared, angle // 90).copy()
                t0 = time.perf_counter()
                extraction = run_tesseract(candidate)
                structured = extract_record(extraction["text"])
                rank = (structured["schema_valid"], -len(structured["errors"]), extraction["mean_confidence"])
                candidates.append((rank, extraction, structured))
                out["attempts"].append({"rotation_ccw": angle, "mean_confidence": extraction["mean_confidence"],
                                        "latency_ms": round((time.perf_counter()-t0)*1000, 3)})
                if structured["schema_valid"] and extraction["mean_confidence"] >= .85:
                    break
            _, extraction, structured = max(candidates, key=lambda row: row[0])
            out.update(route="LOCAL_OCR" if structured["schema_valid"] and extraction["mean_confidence"] >= .85 else "HUMAN_REVIEW",
                       extraction=extraction, structured=structured)
        out["latency_ms"] = round((time.perf_counter()-start)*1000, 3)
        if self.max_cache:
            self.cache[key] = copy.deepcopy(out)
            while len(self.cache) > self.max_cache:
                self.cache.popitem(last=False)
        return out

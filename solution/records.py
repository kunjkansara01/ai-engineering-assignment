"""Q1, Q3 and Q4: deterministic processing with explicit audit trails."""
import math
import re
import unicodedata
from collections import Counter
from difflib import get_close_matches

DOCUMENTS = [
    {"id": "D101", "quality": .92, "time": 101},
    {"id": "D102", "quality": .43, "time": 102},
    {"id": "D103", "quality": .81, "time": 103},
    {"id": "D101", "quality": .92, "time": 104},
    {"id": "D104", "quality": .28, "time": 105},
    {"id": "D105", "quality": .74, "time": 105},
    {"id": "D102", "quality": .92, "time": 106},
]
OCR_RESULTS = [
    {"text": t, "confidence": c} for t, c in [
        ("Paracetamol", .98), ("500 mg", .96), ("twlce daily", .61),
        ("for 5 days", .94), ("with", .14), ("water", .94),
        ("Unknown Drug", .58), ("UnknownDrug", .38)]
]
SAMPLE_TEXT = "Patient: Rahul Sharma\nMedicine: Paracetamol\nStrength: 500 mg\nFrequency: Twice daily\nDuration: 5 days"


def probability(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Score must be a finite number in [0, 1]")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Score must be a finite number in [0, 1]")
    return float(value)


def process_documents(documents):
    """Latest timestamp wins; equal timestamps keep first input occurrence."""
    selected, audit = {}, []
    for index, document in enumerate(documents):
        d = dict(document)
        if not isinstance(d.get("id"), str) or not d["id"].strip():
            raise ValueError("Nonempty string id required")
        d["quality"] = probability(d.get("quality"))
        t = d.get("time")
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t):
            raise ValueError("Finite timestamp required")
        previous = selected.get(d["id"])
        if previous is None:
            selected[d["id"]] = d
        else:
            newer = t > previous["time"]
            audit.append({"id": d["id"], "incoming_index": index,
                          "kept_time": t if newer else previous["time"],
                          "discarded_time": previous["time"] if newer else t,
                          "reason": "latest_timestamp" if t != previous["time"] else "tie_keep_first"})
            if newer:
                selected[d["id"]] = d
    counts = {"ACCEPT": 0, "REVIEW": 0, "RETAKE": 0}
    for d in selected.values():
        d["category"] = "ACCEPT" if d["quality"] >= .8 else "REVIEW" if d["quality"] >= .5 else "RETAKE"
        counts[d["category"]] += 1
    return {"documents": list(selected.values()), "counts": counts,
            "average_quality": sum(d["quality"] for d in selected.values()) / len(selected) if selected else None,
            "duplicates_removed": len(documents) - len(selected), "duplicate_audit": audit}


def confidence_class(value):
    c = probability(value)
    return "HIGH_CONFIDENCE" if c >= .9 else "REVIEW_REQUIRED" if c >= .6 else "LOW_CONFIDENCE"


def clean_text(text):
    if not isinstance(text, str):
        raise ValueError("Text must be a string")
    cleaned = " ".join(unicodedata.normalize("NFC", text).split())
    return {"original_text": text, "corrected_text": cleaned,
            "reason": "Normalize Unicode NFC and whitespace" if cleaned != text else "No change",
            "semantic_change": False}


def review_ocr(results):
    records = []
    for item in results:
        row = clean_text(item["text"])
        row.update(confidence=probability(item["confidence"]),
                   classification=confidence_class(item["confidence"]),
                   proposed_text=None, correction_status="UNCHANGED")
        if row["corrected_text"] == "twlce daily":
            row.update(proposed_text="twice daily", correction_status="PENDING_REVIEW",
                       proposal_reason="Possible OCR l/i confusion; reviewer must confirm")
        records.append(row)
    return records


def approve_correction(row, corrected_text, reason, reviewer):
    if not all(isinstance(x, str) and x.strip() for x in (corrected_text, reason, reviewer)):
        raise ValueError("Correction, reason and reviewer required")
    return {**row, "corrected_text": corrected_text, "reason": reason,
            "reviewer": reviewer, "correction_status": "APPROVED", "semantic_change": True}


def lookup_medicine(name, dictionary):
    """Caller supplies a versioned trusted dictionary; fuzzy results are suggestions."""
    names = {n.casefold(): n for n in dictionary["names"]}
    key = name.strip().casefold()
    return {"original": name, "exact_match": names.get(key),
            "suggestions": [names[n] for n in get_close_matches(key, names, n=3, cutoff=.75)] if key not in names else [],
            "dictionary_version": dictionary["version"], "source": dictionary["source"],
            "status": "MATCHED" if key in names else "REVIEW_REQUIRED"}


FIELDS = {"patient": "patient_name", "medicine": "medicine_name", "strength": "strength",
          "frequency": "frequency", "duration": "duration"}
STRENGTH = re.compile(r"[1-9]\d*(?:\.\d+)?\s*(?:mcg|mg|g|ml|iu)(?:\s*/\s*(?:[1-9]\d*\s*)?(?:ml|tablet))?", re.I)
FREQUENCY = re.compile(r"(?:once|twice|three times|four times)\s+(?:daily|a day)|(?:every\s+[1-9]\d*\s+hours?)|(?:at bedtime)", re.I)
DURATION = re.compile(r"(?:for\s+)?[1-9]\d*\s+(?:days?|weeks?|months?)", re.I)


def validate_record(record):
    """Validate syntax only. Never certify a medicine or dosage as clinically safe."""
    errors = []
    for field in FIELDS.values():
        v = record.get(field)
        if not isinstance(v, str) or not v.strip():
            errors.append({"field": field, "code": "MISSING"})
        elif re.search(r"\?|unreadable|unknown|illegible|unclear", v, re.I):
            errors.append({"field": field, "code": "AMBIGUOUS"})
    for field, pattern in (("strength", STRENGTH), ("frequency", FREQUENCY), ("duration", DURATION)):
        if isinstance(record.get(field), str) and not pattern.fullmatch(record[field].strip()):
            errors.append({"field": field, "code": "INVALID_FORMAT"})
    s, f = record.get("strength"), record.get("frequency")
    if isinstance(s, str) and isinstance(f, str) and (FREQUENCY.fullmatch(s) or STRENGTH.fullmatch(f)):
        errors.append({"field": "strength/frequency", "code": "POSSIBLY_SWAPPED"})
    unexpected = set(record) - set(FIELDS.values()) - {"validation_status"}
    if unexpected:
        errors.append({"field": sorted(unexpected), "code": "UNEXPECTED_FIELDS"})
    return errors


def extract_record(text):
    """Label-based extraction with evidence, duplicate ambiguity and no guessing."""
    record = dict.fromkeys(FIELDS.values())
    evidence, errors, seen = {}, [], set()
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        if ":" not in line:
            errors.append({"line": line_number, "code": "UNPARSED_LINE", "text": line})
            continue
        label, value = line.split(":", 1)
        field = FIELDS.get(label.strip().casefold())
        if field is None:
            errors.append({"line": line_number, "code": "UNKNOWN_LABEL", "text": line})
            continue
        evidence.setdefault(field, []).append({"line": line_number, "text": line, "value": value.strip()})
        if field in seen:
            record[field] = None
            errors.append({"field": field, "code": "DUPLICATE_FIELD"})
        else:
            record[field] = value.strip() or None
            seen.add(field)
    errors.extend(validate_record(record))
    record["validation_status"] = "REVIEW_REQUIRED"
    return {"record": record, "schema_valid": not errors, "errors": errors, "evidence": evidence,
            "original_text": text}

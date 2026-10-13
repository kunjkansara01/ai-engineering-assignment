"""Q7: local transcription, keyword triage and exact act/section lookup."""
import argparse
import hashlib
import json
import re
import time
from datetime import date
from pathlib import Path
from .records import clean_text

LEGAL_TERMS = ("agreement", "contract", "paid", "payment", "threat", "cheat", "fraud", "court", "police", "section",
               "evidence", "witness", "signed", "loan", "debt", "consent", "hearing", "complaint", "fir",
               "धमकी", "धोखा", "पुलिस", "समझौता", "धारा", "अदालत", "भुगतान", "गवाह", "सबूत", "पैसे",
               "dhamki", "dhokha", "paise", "karar", "gawah", "saboot", "paisa")
NEGATION = re.compile(r"\b(?:not|never|no|didn't|wasn't|nahi|nahin|mat)\b|नहीं|नही|मत", re.I)
UNCERTAINTY = re.compile(r"\b(?:maybe|perhaps|might|unsure|possibly|shayad)\b|शायद", re.I)
SMALLTALK = re.compile(r"(?:hello|hi|good morning|how are you|nice weather today|chai chahiye|नमस्ते)[.!? ]*", re.I)
SECTION = re.compile(r"(?:(BNS|IPC|BNSS)\s+)?(?:section|sec\.?|धारा)\s*(\d+(?:\(\d+\))?)(?:\s+(BNS|IPC|BNSS))?", re.I)


def has_term(text, term):
    # Unicode word boundaries are unreliable for Hindi vowel signs.
    return term in text.casefold() if not term.isascii() else bool(re.search(r"\b" + re.escape(term) + r"\w*\b", text, re.I))


def word_error_rate(reference, hypothesis):
    def normalize(text):
        return re.sub(r"[^\w\s\u0900-\u097f]", "", text.casefold()).split()
    a, b = normalize(reference), normalize(hypothesis)
    if not a:
        return 0. if not b else None
    row = list(range(len(b) + 1))
    for i, token in enumerate(a, 1):
        nxt = [i]
        for j, other in enumerate(b, 1):
            nxt.append(min(nxt[j-1]+1, row[j]+1, row[j-1]+(token != other)))
        row = nxt
    return row[-1] / len(a)


def transcribe_local(audio_path, model_path="tiny", language=None, model=None):
    """Transcribe locally; reuse a supplied model or load one for this call."""
    from faster_whisper import WhisperModel
    from .benchmark import rss_mb
    start, cpu0 = time.perf_counter(), time.process_time()
    t0 = time.perf_counter()
    if model is None:
        model = WhisperModel(model_path, device="cpu", compute_type="int8", cpu_threads=2)
    load_ms = (time.perf_counter()-t0)*1000
    iterator, info = model.transcribe(str(audio_path), language=language, task="transcribe", beam_size=1,
                                      vad_filter=True, condition_on_previous_text=False, word_timestamps=True)
    segments = [{"id": str(s.id), "start": s.start, "end": s.end, "text": s.text,
                 "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob,
                 "words": [{"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                           for w in (s.words or [])]} for s in iterator]
    elapsed = time.perf_counter()-start
    return {"segments": segments, "language": info.language, "language_probability": info.language_probability,
            "audio_seconds": info.duration, "model": model_path, "api_cost_usd": 0.,
            "latency_ms": elapsed*1000, "model_load_ms": load_ms,
            "real_time_factor": elapsed/info.duration if info.duration else None,
            "cpu_seconds": time.process_time()-cpu0,
            "rss_after_mb": rss_mb(),
            "transcript": " ".join(s["text"].strip() for s in segments)}


class LegalPipeline:
    def __init__(self, database):
        self.database = database

    def analyze(self, segments, incident_date=None, jurisdiction="India"):
        start = time.perf_counter()
        if incident_date is not None:
            date.fromisoformat(incident_date)  # Validate the caller's date without inferring one.
        all_segments, facts, dates, parties, references, candidates = [], [], [], [], [], []
        db = {(p["act"], p["section"]): p for p in self.database["provisions"]}
        for n, segment in enumerate(segments):
            original = segment["text"]
            cleaned = clean_text(original)
            text = cleaned["corrected_text"]
            sid = str(segment.get("id", n))
            terms = [t for t in LEGAL_TERMS if has_term(text, t)]
            uncertain = bool(UNCERTAINTY.search(text))
            weak_asr = segment.get("avg_logprob", 0) < -1 or segment.get("no_speech_prob", 0) > .6
            # Only full small-talk matches with adequate ASR confidence are excluded.
            label = "UNCERTAIN" if weak_asr else "LEGALLY_RELEVANT" if terms else "IRRELEVANT" if SMALLTALK.fullmatch(text) else "UNCERTAIN"
            evidence = {"segment_id": sid, "start": segment.get("start"), "end": segment.get("end"), "quote": original}
            row = {**segment, **cleaned, "label": label, "negated": bool(NEGATION.search(text)),
                   "uncertain_statement": uncertain, "terms": terms}
            all_segments.append(row)
            if label != "IRRELEVANT":
                facts.append({"statement": original, "negated": row["negated"], "uncertain": uncertain or weak_asr,
                              "evidence": evidence, "status": "UNVERIFIED_STATEMENT"})
                for match in re.finditer(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b|\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b", original, re.I):
                    dates.append({"text": match.group(), "evidence": evidence, "normalized": None})
                for match in re.finditer(r"(?:Party|Complainant|Respondent|पक्ष|शिकायतकर्ता)\s*:\s*([^,;.\n]+)", original, re.I):
                    parties.append({"name": match.group(1).strip(), "evidence": evidence})
            for match in SECTION.finditer(original):
                act = (match.group(1) or match.group(3) or "").upper()
                section = match.group(2)
                provision = db.get((act, section))
                # Match both act and section; a bare number stays unverified.
                references.append({"mention": match.group(), "act": act or None, "section": section,
                                   "status": "VERIFIED_REFERENCE" if provision else "UNVERIFIED_REFERENCE",
                                   "provision": provision, "evidence": evidence,
                                   "applicability": "NOT_DETERMINED"})
            for provision in db.values():
                matches = [t for t in provision["keywords"] if has_term(original, t)]
                if matches:
                    temporal = (incident_date is not None and incident_date >= provision["effective_from"]
                                and (not provision.get("effective_to") or incident_date <= provision["effective_to"]))
                    candidates.append({"act": provision["act"], "section": provision["section"], "title": provision["title"],
                                       "source": provision["source"], "database_version": self.database["version"],
                                       "evidence": evidence, "matched_terms": matches, "negated": row["negated"],
                                       "status": "CANDIDATE_FOR_REVIEW" if temporal and jurisdiction == "India" else "DATE_OR_JURISDICTION_REVIEW",
                                       "applicability": "NOT_DETERMINED"})
        return {"segments": all_segments, "focused_segment_ids": [s.get("id", str(i)) for i,s in enumerate(all_segments) if s["label"] != "IRRELEVANT"],
                "parties": parties, "facts": facts, "dates": dates, "explicit_sections": references,
                "candidate_provisions": candidates, "database_version": self.database["version"],
                "database_sha256": hashlib.sha256(json.dumps(self.database, sort_keys=True).encode()).hexdigest(),
                "incident_date": incident_date, "jurisdiction": jurisdiction, "requires_legal_review": True,
                "latency_ms": (time.perf_counter()-start)*1000, "api_cost_usd": 0.}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--model", default="tiny")
    parser.add_argument("--language", default=None, choices=["en", "hi"])
    parser.add_argument("--reference", help="UTF-8 reference transcript for measured ASR WER")
    parser.add_argument("--incident-date", default=None)
    parser.add_argument("--database", default="data/legal_db.json")
    parser.add_argument("--output", default="outputs/audio_result.json")
    args = parser.parse_args()
    asr = transcribe_local(args.audio, args.model, args.language)
    db = json.loads(Path(args.database).read_text())
    result = {"asr": asr, "legal": LegalPipeline(db).analyze(asr["segments"], args.incident_date)}
    result["wer"] = word_error_rate(Path(args.reference).read_text(), asr["transcript"]) if args.reference else None
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

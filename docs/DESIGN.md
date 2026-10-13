# Design and question-by-question rationale

## Q1 — deduplication

Use document ID as identity and retain the latest timestamp. A repeated ID can represent a later, improved voice capture; first-occurrence or highest-quality selection could retain stale information. Input order does not matter unless timestamps tie, when the first occurrence wins deterministically. Every discarded duplicate has an audit entry. The input is not mutated. Runtime is O(n), storage O(unique IDs), with O(duplicates) audit storage. Empty input returns zero counts and a null mean. Invalid IDs, timestamps and nonfinite/out-of-range scores raise errors.

Latest-record scores are 0.92, 0.92, 0.81, 0.28 and 0.74: mean 0.734; ACCEPT 3, REVIEW 1, RETAKE 1. If first occurrence were chosen instead, the answer would change. The specification does not choose a strategy; the implementation states it explicitly. Continuous intervals are interpreted as [0.50,0.80) and [0.60,0.90), avoiding gaps at 0.795 or 0.895.

## Q2 — document image quality

OpenCV converts BGR/BGRA to grayscale, computes Laplacian variance (sharpness), mean intensity (exposure), and percentile contrast. Initial thresholds: blur variance 100, intensity range 70–250 inclusive, and contrast >=30. They distinguish our synthetic fixtures but are **heuristic starting values**, not universal calibrated thresholds. Laplacian variance depends on text density, resolution and noise. A textured/noisy image can score highly without being readable; a sparse clean page can fail the brightness rule. Calibrate on labelled camera/device data with false-reject and false-accept rates, and standardize resolution before selecting thresholds.

OCR suitability also depends on document boundaries, skew, 90-degree rotation, perspective, shadows, glare, clipping, text size and language. A production gate would measure local illumination, detect/crop the page and assess downstream OCR confidence. This implementation checks three metrics; Q5 adds bounded rotation recovery. It does not implement a trained shadow detector or general perspective correction.

## Q3 — confidence, correction and medicine lookup

Each token retains original text, normalized text, original confidence and a classification. Whitespace and Unicode NFC normalization are recorded. `twlce daily` gets a **proposal**, not silent replacement. Applying a semantic correction requires reviewer and reason; the original remains. Low confidence tokens stay in the audit trail even if they look like stop words. A production review must use the source image and neighboring words, because a missing negation or unit can change meaning.

`lookup_medicine` accepts a supplied versioned dictionary, performs exact case-insensitive matching and returns fuzzy candidates without substituting. The tiny dictionary used in tests is explicitly synthetic, not a trusted clinical vocabulary. Production should ingest an appropriately licensed and jurisdiction-relevant formulary/drug terminology from its authoritative provider, preserving concept IDs, generic/brand aliases, strengths, forms, release version and provenance. Normalize units separately, distinguish look-alike/sound-alike names, and require pharmacist/clinician verification of ambiguous matches. A name match does not validate a prescription or a dose. No real medicine dictionary integration is claimed.

## Q4 — structured records

The label parser returns the five requested fields and `validation_status: REVIEW_REQUIRED`; additional envelope fields provide errors and line evidence. Syntax-valid does not mean clinically verified. Missing values become JSON null. Repeated labels are ambiguous and yield null rather than a last-write guess. Strength, frequency and duration use full-string allowlisted patterns. Swapped-looking fields are flagged, never automatically exchanged. Unknown and unparsed lines are retained as errors. The supported grammar is intentionally narrow (e.g., numeric durations and spelled-out daily frequency); legitimate formats outside it require review.

For LLM extraction, require a fixed JSON Schema with all five keys, string-or-null types and `additionalProperties: false`. Supply the OCR text as quoted data and require a literal evidence span per non-null field. Parse JSON, run schema validation, run these deterministic domain validators, then verify every value against the source span. Reject extra keys, guessed missing fields, malformed units, conflicting spans and unsupported corrections. Allow at most one bounded schema-repair attempt and route unresolved cases to review. LLM confidence is not evidence. This assignment uses the deterministic parser instead of spending tokens on an unnecessary extraction call.

## Q5 — OCR pipeline

| Stage | Purpose / implementation | Input → output | Accuracy impact | Latency and API-cost impact |
|---|---|---|---|---|
| Decode and bounds | Validate uint8 shape and pixel limit | Array → grayscale | Reject corrupt inputs | Linear work; $0 API |
| Scoped cache | Hash image, shape, project, pipeline version; bounded LRU | Image → cached result or miss | Reuses identical work only; deep copy prevents mutation | Warm hit skips OCR; $0 |
| Quality gate | Blur, brightness, contrast | Grayscale → scores/reasons | Reject obviously unusable inputs, with false-reject risk | Cheap; prevents wasted OCR |
| Preprocessing | Grayscale, large-image resize, Otsu threshold | Image → binary page | Improves clean printed contrast; can harm handwriting | Linear work; no API |
| Intelligent routing | Usable → local OCR; failed gate → recapture | Scores → route | Abstains on poor evidence | Early exit; no LLM calls |
| AI text extraction | Tesseract OEM 1 LSTM, English, PSM 6 | Binary page → words, boxes, confidences, lines | Actual neural OCR, not canned text | One local process normally |
| Rotation fallback | Try 90/180/270 degrees only if first result weak | Failed extraction → bounded candidates | Recovers right-angle rotation | Up to four passes; 20-second timeout each |
| Validation | Label schema, domain rules, word confidence | Text → record and errors | Detects missing/swapped/ambiguous values | Cheap deterministic rules |
| Output | JSON with route, timings, evidence and review status | All stages → auditable result | No automatic clinical approval | No external API charge |

Choose among attempts by schema validity, error count and average word confidence; none is a calibrated probability of semantic correctness. General handwriting is deliberately a human-review route. No untested cloud or generative model is added merely to satisfy the word "AI": the OCR engine is an LSTM model. API cost is measured as zero calls/$0; CPU, electricity, installation and hardware are not free. Cache is per instance and project-scoped but not an authentication mechanism. Deploy each instance behind an authenticated service; production keys should also include tenant, engine version and configuration. Avoid persisting sensitive document contents unencrypted. Broad deskew, table layout, multilingual OCR and medical handwriting are future work.

Benchmark reports uncached versus cached timings separately. Five independent uncached passes and twenty warm hits use the same synthetic clear image; the rotated fixture and quality failures are functional cases. These small measurements cannot establish a global optimum of accuracy/latency/cost.

## Q6 — multiple knowledge bases

Architecture: trusted principal → session ownership/ACL check → explicit project scope → topic/follow-up resolution → metadata-filtered records → conservative lexical evidence retrieval → same-fact contradiction check → exact source answer. History is isolated per owner/session/project and bounded to 20 turns. Switching project clears the history and active topic. Permission revocation takes effect on the next request. Sources carry document ID, source path, version, project and exact quote. Documents are data and never executed.

The prototype uses multiple curated source records per project, also exported as Markdown. Ingestion into structured fact records is manual; it does not claim automatic parsing or semantic embeddings over arbitrary PDFs. The answer is extractive and costs zero LLM tokens. Unknown aspects abstain rather than recycling loosely related paragraphs. Contradictions are detected only for matching fact keys with differing values. It does not detect every natural-language contradiction.

### Failure cases and fixes (executable tests and conversations included)

| Failure | Naive outcome | Implemented fix | Test |
|---|---|---|---|
| Global retrieval mixes Atlas/Boreal leave policies | Wrong allowance or information leak | Filter by authorized project before retrieval | `test_retrieval`, `test_access_denied` |
| User changes leave topic to API | Previous topic contaminates answer | Explicit current topic replaces follow-up context | `test_topic_switch` |
| Unsupported salary/carryover detail | Plausible but fabricated answer | Missing-evidence response with no citations | `test_missing_evidence`, `test_unsupported_detail` |
| User guesses another session ID | Another user's history/evidence disclosed | Session ownership + ACL rechecked per turn | `test_cross_user_session`, `test_revoked_access` |
| Two retention policies disagree | Arbitrary source silently wins | Return contradiction and both source quotes | `test_conflicting_documents` |
| Cross-project follow-up reuses old context | Answers from previous knowledge base | Authorized explicit switch clears memory | `test_explicit_project_switch` |
| Prompt requests secret/context override | Leaks hidden context | No generative agent/tools; scope checks and refusal guard | `test_injection`, `test_prompt_cannot_switch_project` |

The five required cases are the first five; two additional cases are included. "Cross-document leakage" is interpreted as unauthorized project/context mixing and unsupported blending; citations from multiple **authorized** sources are necessary when reporting contradictions.

Production: authenticate with OIDC/JWT server-side, enforce tenant/project ACL in database queries and storage, use row-level security and permission-aware hybrid retrieval, version/reindex source chunks, bind citations to immutable hashes, evaluate recall and groundedness with held-out questions, add rate limits and encrypted audit logs, and test adversarial content. The Python identity constructor is a trust boundary, not a remote login mechanism. The phrase-based refusal is defense in depth; authorization is the actual isolation control.

## Q7 — local legal conversation processing

Whisper tiny multilingual via faster-whisper runs CPU int8 with two threads, VAD, beam size 1, word timestamps, transcription (not translation), and previous-text conditioning disabled. The model is loaded once and reused. A larger multilingual model/beam may improve difficult accents but must be benchmarked for latency and WER. VAD may drop quiet legally important speech; evaluate it on real audio before deployment. No diarization or speaker identity inference is claimed.

ASR text and timestamps are preserved. Transcript correction is limited to audited Unicode/whitespace normalization; no semantic substitution of names, negations or section numbers. Local dictionaries label utterances LEGALLY_RELEVANT, IRRELEVANT or UNCERTAIN. Only exact small-talk utterances are removed from the focused view; all original segments remain in the archive. A segment containing small talk plus a legal fact is retained. Unknown utterances are kept as UNCERTAIN. Negation, hedging and low ASR confidence are recorded; segmentation-level flags cannot resolve every grammatical scope.

Facts remain attributed, unverified statements with segment evidence; dates remain literal (ambiguous day/month order is not guessed); parties are extracted only from explicit labels. Keywords collect legal terms. Explicit references require an exact act+section match in a versioned local JSON index. `IPC section 351`, bare `section 351`, unknown subsections and nonexistent numbers are unverified; they never become BNS references automatically. Keywords can suggest only database entries, always as review candidates. Missing incident date, old incident date or other jurisdiction triggers review. Existence of a section is distinct from applicability.

The local database contains only BNS 318 and 351 titles, with official URLs and a version/hash. Official indexed excerpts were checked; full-source downloads timed out. This does not claim a complete statute database or a current amendment audit. Titles suffice to demonstrate exact-reference lookup, not full statutory analysis. Production needs a lawyer-reviewed versioned corpus of full provisions, commencement/amendment/repeal histories and case-specific applicability analysis, with immutable source snapshots.

Measurements: normalized token-level Levenshtein WER against the synthetic audio reference; actual inference wall time and real-time factor; parent CPU seconds and one-core-equivalent CPU usage; sampled peak parent-process RSS every 10 ms; legal filtering accuracy/confusion matrix on 18 synthetic transcript fixtures. RAM sampling may miss sub-10ms peaks, CPU and memory metrics exclude child processes, and benchmark memory includes already loaded libraries/models. Synthetic English speech is not representative of real multilingual legal conversation. Hindi/Hinglish are exercised only as text fixtures; audio performance remains unmeasured. An accurate legal-entity benchmark additionally needs human-labelled spans and precision/recall by language.

## Sources

- Assignment: https://docs.google.com/document/d/1WdWlP_UCgjgKN6IBde7N6qnE6y4rd6pofy15Px7N_3k/edit
- faster-whisper API and local CPU inference: https://github.com/SYSTRAN/faster-whisper
- Whisper model/research: https://github.com/openai/whisper
- Tesseract installation: https://tesseract-ocr.github.io/tessdoc/Installation.html
- Official BNS act metadata: https://www.indiacode.nic.in/handle/123456789/20062
- Official NCRB provisions/index: https://cytrain.ncrb.gov.in/staticpage/web_pages/ChaptersBNS.html and https://cytrain.ncrb.gov.in/staticpage/web_pages/IndexBNS.html

External documentation informed implementation. Synthetic fixtures were created for this project; no real patient or legal conversation was used.

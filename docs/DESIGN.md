# Implementation notes

## Q1 — deduplication

Records are keyed by document ID. The latest timestamp wins so an older capture cannot replace a newer one; timestamp ties keep the first occurrence. Output order follows the first occurrence of each ID. Each discarded duplicate gets an audit entry, and input records are unchanged. Runtime is O(n), with O(unique IDs) record storage and O(duplicates) audit storage. Empty input returns zero counts and a null mean. Invalid IDs, timestamps and nonfinite or out-of-range scores raise errors.

Q1 classifies scores as ACCEPT >=0.80, REVIEW [0.50,0.80), or RETAKE <0.50. Q3 uses HIGH_CONFIDENCE >=0.90, REVIEW_REQUIRED [0.60,0.90), or LOW_CONFIDENCE <0.60. Half-open intervals avoid gaps at values such as 0.795 and 0.895. Fixture results are in [RESULTS.md](../RESULTS.md).

## Q2 — document image quality

OpenCV converts BGR/BGRA to grayscale and checks Laplacian variance >=100, mean intensity 70–250 inclusive, and 99th-minus-1st percentile contrast >=30. These heuristic thresholds distinguish the synthetic fixtures; they are not calibrated for camera data. Noise or texture can inflate sharpness, and sparse pages can fail brightness checks. Production thresholds need labelled device data, consistent resolution, and measured false-accept and false-reject rates.

The gate does not detect page boundaries, skew, perspective, shadows, glare, clipping or text size. Q5 adds right-angle rotation recovery and OCR confidence checks. Page cropping and local illumination checks remain future work.

## Q3 — confidence, correction and medicine lookup

Each OCR item retains its original text, normalized text, confidence and classification. Normalization changes only whitespace and Unicode NFC. `twlce daily` gets a correction proposal; applying it requires a reviewer and reason. Low-confidence items remain in the audit trail. Review needs the source image and surrounding words because a missing negation or unit can change meaning.

`lookup_medicine` matches names case-insensitively and returns up to three fuzzy suggestions at a 0.75 similarity cutoff. It reports the supplied dictionary version and source without checking their authority. The test dictionary is synthetic. A clinical integration would need licensed terminology with concept IDs, aliases, strengths, forms and provenance, plus clinician review of ambiguous matches. A name match does not validate a prescription or dose.

## Q4 — structured records

The parser returns five fields, `validation_status: REVIEW_REQUIRED`, errors and line evidence. Missing, empty or repeated fields become null. Ambiguous text remains in the record and is flagged. Strength, frequency and duration must match full-string patterns; suspected swaps are flagged without exchanging values. Unknown labels and unparsed lines become errors. The grammar accepts numeric durations and spelled-out daily frequencies; other legitimate formats may fail. Passing syntax checks does not establish clinical safety.

An LLM extension could require all five keys as string-or-null values with `additionalProperties: false`, plus a literal source span for each value. It would still need JSON Schema checks, these domain validators and source-span verification. Reject guessed values, extra keys and unsupported corrections; allow at most one schema-repair attempt before review. The current implementation uses only the label parser.

## Q5 — OCR pipeline

| Stage | Purpose / implementation | Input → output | Accuracy impact | Latency and API-cost impact |
|---|---|---|---|---|
| Decode and bounds | Validate uint8 shape and pixel limit | Array → grayscale | Reject corrupt inputs | Linear work; $0 API |
| Scoped cache | Hash image, shape, project, pipeline version; bounded LRU | Image → cached result or miss | Reuses identical work only; deep copy prevents mutation | Warm hit skips OCR; $0 |
| Quality gate | Blur, brightness, contrast | Grayscale → scores/reasons | Reject obviously unusable inputs, with false-reject risk | Cheap; prevents wasted OCR |
| Preprocessing | Grayscale, large-image resize, Otsu threshold | Image → binary page | Improves clean printed contrast; can harm handwriting | Linear work; no API |
| Routing | Usable → local OCR; failed gate → recapture | Scores → route | Rejects failed quality checks | Early exit; no LLM calls |
| Text extraction | Tesseract OEM 1 LSTM, English, PSM 6 | Binary page → words, boxes, confidences, lines | Recognizes printed English | One local process normally |
| Rotation fallback | Try 90/180/270 degrees only if first result weak | Failed extraction → bounded candidates | Recovers right-angle rotation | Up to four passes; 20-second timeout each |
| Validation | Label schema, domain rules, word confidence | Text → record and errors | Detects missing/swapped/ambiguous values | Cheap deterministic rules |
| Output | JSON with route, timings, evidence and review status | All stages → auditable result | No automatic clinical approval | No external API charge |

Attempts are ranked by schema validity, fewer errors, then mean word confidence. Rotation stops when the schema passes and confidence reaches 0.85; otherwise the best attempt routes to HUMAN_REVIEW. These checks do not measure semantic correctness or detect handwriting. Handwriting accuracy is unmeasured. There are no external API calls; local compute costs are excluded from the reported $0 API charge.

The cache hashes grayscale pixels and includes shape, project and pipeline version. Deep copies protect cached results from caller mutation. Project scoping is not authentication; a deployed service would need tenant access checks and cache keys covering engine version and configuration. Sensitive stored results need encryption. Deskew, table layout, multilingual OCR and medical handwriting remain future work.

The benchmark uses five uncached passes and twenty cache hits on the same clear fixture. Rotation and quality failures are functional checks. See [RESULTS.md](../RESULTS.md) for measurements and their scope.

## Q6 — multiple knowledge bases

Each request checks session ownership and project access before resolving the topic and filtering records. Answers quote source text with document ID, path, version and project. History is limited to 20 turns per session. Switching project clears history and topic; revoked permissions take effect on the next request. Documents are returned as text, never executed.

`demo_documents()` supplies curated facts that `run_demo.py` exports as Markdown. The bot reads the records, not those files or arbitrary PDFs. It supports leave, API and retention topics, reusing the last answered topic only for pronoun-based follow-ups. Unknown content terms cause abstention. Conflicts require the same fact key with different values; broader natural-language contradictions are not detected. No LLM is called.

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

The first five rows cover the required failure cases. Multiple authorized sources may appear together when reporting a contradiction; project scope still applies to every citation.

`Principal` trusts application-supplied identity; it does not authenticate users. Production needs server-side authentication and tenant/project checks in storage and retrieval. Source versioning, immutable citations, held-out retrieval tests, rate limits and encrypted audit logs are also missing. The phrase refusal guard covers a few patterns; ownership and permission checks enforce isolation.

## Q7 — local legal conversation processing

The demo uses multilingual Whisper tiny through faster-whisper on CPU int8: two threads, VAD, beam size 1, word timestamps and no previous-text conditioning. It transcribes without translation. The demo reuses one model for both clips; `transcribe_local` loads a model on each call unless one is supplied. Larger models and beams need latency/WER evaluation. VAD may drop quiet speech. Speaker diarization and identity inference are not implemented.

Original ASR text and timestamps are retained. Corrections only normalize Unicode and whitespace. Keyword rules label segments LEGALLY_RELEVANT, IRRELEVANT or UNCERTAIN. ASR log probability below -1 or no-speech probability above 0.6 forces UNCERTAIN. Only full small-talk matches with adequate ASR confidence leave the focused view; all segments remain archived. Mixed small talk/legal facts and unknown utterances stay. Negation and hedging flags apply to whole segments and do not resolve grammatical scope.

Facts are unverified statements linked to segment evidence. Dates remain literal, including ambiguous day/month order; parties require explicit labels. References need an exact act/section match in the local index. Bare `section 351`, `IPC section 351`, unknown subsections and nonexistent numbers remain unverified. Keywords suggest only indexed entries for review. Missing or out-of-range incident dates and non-India jurisdiction trigger date/jurisdiction review. No match determines legal applicability.

The index contains only BNS 318 and 351 titles with official URLs and a version/hash. Its provenance records checks against official search excerpts; full-source downloads timed out. Full provisions and amendments were not validated. A production corpus would need legal review, full provisions, commencement/amendment/repeal histories and immutable source snapshots.

WER uses token Levenshtein distance after lowercasing and removing punctuation. The benchmark reports wall time, real-time factor, parent CPU time, RSS sampled every 10 ms, and a classification confusion matrix. Samples can miss brief memory peaks; child processes are excluded, while loaded libraries/models are included. [RESULTS.md](../RESULTS.md) records the fixture counts, measurements and language limits. Entity-level precision/recall remains unmeasured and needs human-labelled spans.

## Sources

- Assignment: https://docs.google.com/document/d/1WdWlP_UCgjgKN6IBde7N6qnE6y4rd6pofy15Px7N_3k/edit
- faster-whisper API and local CPU inference: https://github.com/SYSTRAN/faster-whisper
- Whisper model/research: https://github.com/openai/whisper
- Tesseract installation: https://tesseract-ocr.github.io/tessdoc/Installation.html
- Official BNS act metadata: https://www.indiacode.nic.in/handle/123456789/20062
- Official NCRB provisions/index: https://cytrain.ncrb.gov.in/staticpage/web_pages/ChaptersBNS.html and https://cytrain.ncrb.gov.in/staticpage/web_pages/IndexBNS.html

These sources informed the original implementation. Fixtures contain no real patient or legal conversation.

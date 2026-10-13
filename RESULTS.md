# Results report — AI engineering assignment

Prepared 2026-10-12. All measurements below were executed locally; no simulated performance numbers are presented.

## Verification

- **51 automated tests passed**, including actual Tesseract extraction, 90-degree rotation recovery, cache behavior, permission checks, topic switching, contradictions and legal-reference edge cases.
- **12/12 notebook code cells passed** sequentially in a fresh Python process using an in-process IPython shell, including dependency checks, OCR, tests and real speech inference. Notebook outputs are saved.
- The host blocks TCP/IPC socket binding, so external Jupyter-kernel execution and Google Colab hosting were not tested. The notebook is self-contained and designed for Run All after documented setup; no claim of a tested Colab session is made.
- Dependencies were installed successfully in this environment. This was not a clean virtual-machine installation test. Tesseract was already present.

## Q1 — document processing

Latest-timestamp policy: **5 unique records**, 2 duplicates removed, mean quality **0.734**. Counts: **ACCEPT 3 / REVIEW 1 / RETAKE 1**. Timestamp ties preserve the first input occurrence.

## Q2 — measured quality checks

| Synthetic fixture | Laplacian variance | Mean brightness | Usable |
|---|---:|---:|---|
| clear | 920.765 | 239.545 | True |
| blurred | 0.358 | 239.546 | False |
| dark | 29.915 | 43.001 | False |
| rotated | 920.765 | 239.545 | True |
| blank | 0.000 | 245.000 | False |

Q3 returned HIGH_CONFIDENCE 4, REVIEW_REQUIRED 1 and LOW_CONFIDENCE 3. Original text was retained; the suspected typo remained a proposal. Q4 valid, missing, unreadable, duplicated and swapped-field behaviors were tested.

## Q5 — local OCR performance

On the single synthetic printed English page, field exact match was **100% (5/5)** and normalized text WER was **0.0%**. This is not a generalization or handwriting benchmark.
Uncached median: **215.569 ms** across 5 passes. Cached median: **0.846 ms** across 20 hits. Clear and rotated pages reached local extraction; dark, blurred and blank pages were routed to recapture. API calls: **0**, API charge: **$0** (compute costs excluded).

## Q6 — retrieval and access control

Atlas answers used Atlas sources, and explicit topic changes replaced the old topic. Missing salary/carryover evidence caused abstention; inconsistent retention documents returned both citations. Unauthorized project requests, cross-user session/history access and revoked access were rejected. This is a curated lexical/extractive prototype with a trusted in-process principal, not an authenticated web service or general PDF RAG application.

## Q7 — real speech and transcript benchmarks

Model: multilingual Whisper tiny, CPU int8, 2 configured inference threads, beam size 1, VAD enabled. Model load: 386.18 ms (download excluded). The two clips are **synthetic English**, 7.62 seconds each. Clean audio was generated locally using FFmpeg/libflite voice `slt`; noisy audio adds deterministic Gaussian noise at 15 dB SNR with NumPy seed 2026. No real private conversation was used.

| Clip | WER | Inference wall time | Real-time factor | Parent CPU time | Sampled peak parent RSS |
|---|---:|---:|---:|---:|---:|
| legal_clear.wav | 0.00% | 633.92 ms | 0.083 | 1.810 s | 367.43 MB |
| legal_noisy.wav | 17.24% | 705.71 ms | 0.093 | 2.003 s | 371.00 MB |

WER uses lowercased, punctuation-normalized token Levenshtein distance against the supplied reference. The noisy recording demonstrates that legal dates and evidence descriptions can be mistranscribed; preserved originals and human review are necessary. For this fixture, the explicit “did not sign” negation survived both recordings.

Transcript relevance classification: **100% on 18 synthetic hand-labelled text fixtures** (10 relevant, 4 irrelevant, 4 uncertain), evaluated over 30 timing repeats. Median 1.168 ms; p95 1.436 ms per 18-segment batch. These fixtures were written alongside the rules; they are not an independent held-out dataset.
Text-benchmark sampled parent RSS: 144.12 MB. Audio and text API calls/charge: 0 / $0. RAM is sampled every 10 ms and includes libraries/model resident in the parent process. CPU is parent process time; utilization may exceed 100% when multiple threads run. Child-process resources, hardware/electricity and model downloads are excluded.

## Material limitations and remaining submission work

- No real handwriting dataset, real legal conversations or Hindi/Hinglish audio was supplied. Hindi/Hinglish text fixtures pass; their ASR accuracy is unmeasured.
- Legal lookup covers only two official-source BNS section titles; full statutory text, amendments and case applicability are not validated. The full-source fetch timed out, so the provenance explicitly says indexed official excerpts were used.
- Entity extraction is conservative and limited to explicit labels/date patterns, with literal evidence. There is no measured entity-level precision/recall on real data, speaker diarization or general semantic legal reasoning.
- No trusted medicine terminology integration is claimed; the lookup interface accepts a versioned dictionary and the demonstration uses a synthetic mini-dictionary.
- No hosted repository/Colab/Kaggle link has been created. Upload the notebook or project and give the reviewer access.
- Complete candidate name, actual personal time spent and any additional assistance in `DISCLOSURE.md`. AI assistance is disclosed as High.

## Reproduction and build record

Run `python -m unittest discover -s tests -v` and `python run_demo.py --audio` from the project directory after setup. JSON results, source inputs and test log are included. `python execute_notebook.py` provides the restricted-host sequential cell validator; normal Jupyter/Colab users can use Run All. Model weights are downloaded on first use and are not bundled.

Observed automated preparation interval: 2026-10-12T09:15:26.932458+00:00 to 2026-10-12T09:32:52.204278+00:00 (**17.4 minutes**), measured from workspace setup to report creation. This is assistant build wall time, not the candidate’s personal effort.

Environment:

```json
{
  "python": "3.12.14",
  "platform": "Linux-6.18.44-x86_64-with-glibc2.39",
  "logical_cpus": 9,
  "opencv": "4.11.0",
  "numpy": "2.3.5",
  "tesseract": "5.3.4"
}
```

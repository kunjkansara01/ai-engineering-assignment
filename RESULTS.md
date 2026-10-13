# Recorded results — AI engineering assignment

Original run: 2026-10-12. These local measurements are retained from the original build; documentation edits do not replace them with new timings.

## Verification

- **51 tests passed**, covering Tesseract extraction, rotation, caching, access checks, topic changes, conflicts and legal references.
- **12/12 notebook code cells passed** in a fresh process using an in-process IPython shell, including setup, OCR, tests and speech inference. Outputs are saved.
- The original host blocked TCP/IPC socket binding. External Jupyter kernels and Colab were not tested.
- Dependencies installed on that host, where Tesseract was already present. A clean virtual-machine installation was not tested.

## Q1 — document processing

Keeping the latest timestamp produced **5 unique records**, 2 duplicates removed and mean quality **0.734**: **ACCEPT 3 / REVIEW 1 / RETAKE 1**.

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

The synthetic printed English page gave **100% (5/5)** field exact match and **0.0%** normalized text WER. One fixture does not establish general or handwriting accuracy.
Uncached median: **215.569 ms** across 5 passes. Cached median: **0.846 ms** across 20 hits. Clear and rotated pages reached local extraction; dark, blurred and blank pages were routed to recapture. API calls: **0**, API charge: **$0** (compute costs excluded).

## Q6 — retrieval and access control

Atlas answers used Atlas sources. Topic changes updated context, missing salary/carryover evidence caused abstention, and conflicting retention values returned both citations. Unauthorized project requests, cross-user session/history access and revoked access were rejected. These checks use trusted in-process identities and curated records; web authentication and PDF ingestion are not implemented.

## Q7 — real speech and transcript benchmarks

Multilingual Whisper tiny ran on CPU int8 with 2 threads, beam size 1 and VAD. Model load: 386.18 ms, excluding download. Both clips are **synthetic English**, 7.62 seconds each. FFmpeg/libflite voice `slt` generated the clean clip; the noisy clip adds Gaussian noise at 15 dB SNR with NumPy seed 2026. No private conversation was used.

| Clip | WER | Inference wall time | Real-time factor | Parent CPU time | Sampled peak parent RSS |
|---|---:|---:|---:|---:|---:|
| legal_clear.wav | 0.00% | 633.92 ms | 0.083 | 1.810 s | 367.43 MB |
| legal_noisy.wav | 17.24% | 705.71 ms | 0.093 | 2.003 s | 371.00 MB |

WER uses lowercased, punctuation-normalized token Levenshtein distance against the reference. The noisy clip mistranscribed dates and evidence descriptions, showing why originals and review matter. “Did not sign” survived both clips.

Transcript classification: **100% on 18 synthetic hand-labelled text fixtures** (10 relevant, 4 irrelevant, 4 uncertain), with 30 timing repeats. Median 1.168 ms; p95 1.436 ms per batch. The fixtures were written alongside the rules, not held out independently.
Text-benchmark sampled parent RSS: 144.12 MB. Audio and text API calls/charge: 0 / $0. RSS is sampled every 10 ms and includes loaded libraries/models. Parent CPU utilization can exceed 100% across threads. Child processes, hardware/electricity and model downloads are excluded.

## Material limitations and remaining submission work

- Real handwriting, legal conversations and Hindi/Hinglish audio were not supplied. Hindi/Hinglish text tests pass; speech accuracy is unmeasured.
- Legal lookup covers two BNS titles from official indexed excerpts. Full-source retrieval timed out; statutory text, amendments and applicability remain unvalidated.
- Entity extraction uses explicit labels/date patterns and literal evidence. Real-data entity precision/recall, diarization and semantic legal reasoning are unmeasured or unimplemented.
- Medicine lookup uses a synthetic dictionary, without a trusted terminology integration.
- The original build created no hosted repository/Colab/Kaggle link. Submission steps and candidate confirmation are in [README.md](README.md#before-submission) and [DISCLOSURE.md](DISCLOSURE.md).

## Reproduction and build record

See [README.md](README.md#quick-start) for setup, tests and notebook execution. JSON results, inputs and the original test log are included. Model weights are downloaded on first use and are not bundled.

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

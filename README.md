# Local-first AI engineering assignment

Seven runnable solutions, a self-contained notebook, test fixtures, actual local OCR and speech inference, and measured results. AI assistance is disclosed in `DISCLOSURE.md`.

## Quick start

Tested on Linux with Python 3.12 and Tesseract 5. Python 3.11–3.12 is recommended. Initial installation and first speech-model download need internet; subsequent inference is local. No API key is required.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-audio.txt
```

Install the Tesseract **system executable**, in addition to the Python wrapper:

- Ubuntu/Colab: `apt-get update && apt-get install -y tesseract-ocr tesseract-ocr-eng`
- macOS with Homebrew: `brew install tesseract`
- Windows: install Tesseract using the options in the [official installation documentation](https://tesseract-ocr.github.io/tessdoc/Installation.html), and put its executable on PATH.

From this directory:

```bash
python -m unittest discover -s tests -v
python run_demo.py --audio
```

For a faster run without downloading speech weights: `python run_demo.py`. This explicitly reports audio as `NOT_RUN`; it is not an ASR benchmark. The full command runs both clean and noisy audio through a real multilingual Whisper tiny model on CPU. First model download is roughly 75 MB; install/model-download time is excluded from inference metrics but model loading is reported separately. `av==16.0.1` is pinned because PyAV 19 removed an argument used by faster-whisper 1.2.1.

Open `Assignment_Solutions.ipynb` in Jupyter or upload it to Google Colab. The notebook includes the source, tests, local legal index and audio as an embedded reproducible bundle. Run all cells in order. It creates an `assignment_workspace` directory. The notebook's setup installs missing dependencies and installs Tesseract automatically only on Linux when running as root (such as Colab). Local Windows/macOS users must install the system executable first. The final audio cells run by default and download the model if needed. There are no interactive inputs.

## Files and requirements coverage

| Part | Implementation | Demonstration / output |
|---|---|---|
| Q1 | `solution/records.py: process_documents` | `outputs/q1.json` |
| Q2 | `solution/vision.py: check_quality` | Five PNG fixtures and `outputs/q2.json` |
| Q3 | `solution/records.py: review_ocr, approve_correction, lookup_medicine` | `outputs/q3.json`; correction and dictionary tests |
| Q4 | `solution/records.py: extract_record, validate_record` | Valid, incomplete, unreadable and swapped examples in `outputs/q4.json` |
| Q5 | `solution/vision.py: OCRPipeline` | Actual Tesseract extraction, rotation fallback, cache, recapture; `outputs/q5.json` |
| Q6 | `solution/chatbot.py` | Independent project sources under `data/knowledge`; `outputs/q6.json`; failure cases in `docs/DESIGN.md` |
| Q7 | `solution/legal.py` | Local audio + transcript processing, source-backed legal lookup; `outputs/q7.json`, `outputs/audio_benchmark.json` |
| Testing | `tests/test_solution.py` | `outputs/test_results.txt` |
| Results | `RESULTS.md` | Small-fixture accuracy, timing, CPU/RAM, actual ASR WER and limits |
| Explanation | `docs/DESIGN.md` | Assumptions, component design, trade-offs and production improvements |
| Disclosure | `DISCLOSURE.md` | AI tools, usage level, observed build duration and candidate fields |

## Use your own data

### OCR

```python
import cv2
from solution.vision import OCRPipeline
result = OCRPipeline().process(cv2.imread('document.png'), project='my-project')
print(result)
```

English printed text is supported in this prototype. Rotation fallback covers 90-degree increments. Difficult handwriting, severe shadows, perspective distortion and low-confidence text need review; this submission does not claim reliable handwriting recognition.

### Project chatbot

```python
from solution.chatbot import KnowledgeBot, Principal
bot = KnowledgeBot()
user = Principal('alice')  # supplied by trusted application code, never from a user chat message
session = bot.start_session(user, 'atlas')
print(bot.ask(user, session, 'What is the leave allowance?'))
print(bot.ask(user, session, 'How many days is that?'))
print(bot.ask(user, session, 'What is the API rate limit?'))
```

`Principal` is a trusted in-process identity, **not an authentication server**. Do not expose construction of this object or the permission dictionary to remote clients. This is a curated, structured-fact, extractive QA prototype with conservative lexical retrieval, not a general-purpose LLM chatbot. It returns exact source statements. Supply custom records with the same metadata schema to extend the knowledge base.

### Local audio

```bash
python -m solution.legal conversation.wav --model tiny --reference reference.txt --incident-date 2025-08-01
python -m solution.legal hindi.wav --model small --language hi
```

For Hinglish, omit `--language` to allow multilingual detection; code switching accuracy must be evaluated on real recordings. Audio is never sent to an API. Supplying a local model directory makes startup independent of model downloads. Models persist in the standard Hugging Face cache. Do not use the English-only `.en` models for Hindi/Hinglish.

## Submission steps still requiring the candidate

1. Read, run and understand the code, particularly the documented limitations.
2. Fill the candidate name, **actual personal time spent**, and any additional assistance in `DISCLOSURE.md`.
3. Upload the `.ipynb` to Colab/Kaggle and enable reviewer access, or upload this project to a Git repository and share its link. Include the downloadable notebook as required. No hosted repository or notebook URL has been created by this package.
4. Add real document/handwriting and Hindi/Hinglish audio results if the reviewer expects measured performance beyond the supplied synthetic demonstration.

This package is a working minimal prototype, not a production OCR, clinical, or legal decision system. All people, project policies and recordings in the fixtures are synthetic.

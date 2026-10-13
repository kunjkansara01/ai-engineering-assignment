# Local-first AI engineering assignment

Seven solutions covering document processing, local OCR, project QA and local speech processing. See [RESULTS.md](RESULTS.md) for measurements, [docs/DESIGN.md](docs/DESIGN.md) for design and limitations, and [DISCLOSURE.md](DISCLOSURE.md) for AI assistance and candidate details.

## Quick start

The recorded benchmark used Linux, Python 3.12 and Tesseract 5. The notebook targets Python 3.11–3.12. Initial installation and the first speech-model download need internet. Inference runs locally without an API key.

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

Use `python run_demo.py` to skip speech inference; it reports audio as `NOT_RUN`. With `--audio`, multilingual Whisper tiny transcribes the clean and noisy clips on CPU. The first model download is roughly 75 MB. Measurements exclude installation and download time; model loading is timed separately. The `av==16.0.1` pin avoids a PyAV 19 API change affecting faster-whisper 1.2.1.

Open `Assignment_Solutions.ipynb` in Jupyter or upload it to Colab, then run all cells in order. It extracts bundled source, tests, documentation, legal fixtures and audio into `assignment_workspace`. Setup installs missing or mismatched pinned Python packages, including audio dependencies. It installs Tesseract automatically only on Linux as root; install it yourself on Windows/macOS. Audio runs by default and downloads the model if needed. Colab execution has not been verified.

`python build_notebook.py` rebuilds the notebook and embedded files from this repository, clearing saved outputs. `python execute_notebook.py` executes it with an in-process IPython shell and saves outputs; it does not test an external Jupyter kernel. Run the executor from a scratch directory with a copy of the notebook to keep generated files separate from repository results.

## Files and requirements coverage

| Part | Implementation | Demonstration / output |
|---|---|---|
| Q1 | `solution/records.py: process_documents` | `outputs/q1.json` |
| Q2 | `solution/vision.py: check_quality` | Five PNG fixtures and `outputs/q2.json` |
| Q3 | `solution/records.py: review_ocr, approve_correction, lookup_medicine` | `outputs/q3.json`; correction and dictionary tests |
| Q4 | `solution/records.py: extract_record, validate_record` | Valid, incomplete, unreadable and swapped examples in `outputs/q4.json` |
| Q5 | `solution/vision.py: OCRPipeline` | Actual Tesseract extraction, rotation fallback, cache, recapture; `outputs/q5.json` |
| Q6 | `solution/chatbot.py` | Curated facts exported to `data/knowledge`; `outputs/q6.json`; failure cases in `docs/DESIGN.md` |
| Q7 | `solution/legal.py` | Local audio + transcript processing, source-backed legal lookup; `outputs/q7.json`, `outputs/audio_benchmark.json` |
| Testing | `tests/test_solution.py` | `outputs/test_results.txt` |
| Results | `RESULTS.md` | Small-fixture accuracy, timing, CPU/RAM, actual ASR WER and limits |
| Explanation | `docs/DESIGN.md` | Assumptions, component design, trade-offs and production improvements |
| Disclosure | `DISCLOSURE.md` | AI tools, usage level and candidate details |

## Use your own data

### OCR

```python
import cv2
from solution.vision import OCRPipeline
result = OCRPipeline().process(cv2.imread('document.png'), project='my-project')
print(result)
```

OCR uses English printed text and tries rotations in 90-degree increments. Handwriting accuracy is unmeasured. Severe shadows, perspective distortion and low-confidence text need review.

### Project chatbot

```python
from solution.chatbot import KnowledgeBot, Principal
bot = KnowledgeBot()
user = Principal('alice')  # Identity must come from trusted application code.
session = bot.start_session(user, 'atlas')
print(bot.ask(user, session, 'What is the leave allowance?'))
print(bot.ask(user, session, 'How many days is that?'))
print(bot.ask(user, session, 'What is the API rate limit?'))
```

The application must authenticate users before constructing `Principal`. Remote clients must not control this identity or the permission dictionary. The bot quotes curated fact records using lexical matching. To extend its sources, pass records with the same metadata fields; it does not ingest the exported Markdown files.

### Local audio

```bash
python -m solution.legal conversation.wav --model tiny --reference reference.txt --incident-date 2025-08-01
python -m solution.legal hindi.wav --model small --language hi
```

For Hinglish, omit `--language` to allow language detection and use a multilingual model, not an English-only `.en` model. Hindi/Hinglish speech accuracy is unmeasured. Models use the Hugging Face cache; pass a local model directory to avoid downloads at startup.

## Before submission

1. Review and run the code, including the documented limitations.
2. Confirm your name, **actual personal time spent** and assistance in `DISCLOSURE.md`.
3. Add a reviewer-accessible repository or Colab/Kaggle link there and include the downloadable notebook. The recorded build did not create a hosted link.
4. Add real document, handwriting and Hindi/Hinglish audio results if required by the reviewer.

All fixture people, policies and recordings are synthetic. This prototype is not validated for clinical or legal decisions.

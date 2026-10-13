"""Build a portable, self-contained notebook from the reviewable project files."""
import base64
import io
import json
import zipfile
from pathlib import Path
import nbformat as nbf


def build():
    root=Path(__file__).parent
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        paths=[]
        for folder in ['solution','tests','docs']:
            paths.extend(p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
        paths.extend((root/'data').glob('*.json'))
        paths.extend((root/'data/audio').glob('*'))
        paths.extend(root/name for name in ['run_demo.py','requirements.txt','requirements-audio.txt','README.md','DISCLOSURE.md'])
        for p in paths:
            z.write(p,str(p.relative_to(root)))
    bundle=base64.b64encode(buffer.getvalue()).decode()
    cells=[]
    md=lambda s:cells.append(nbf.v4.new_markdown_cell(s))
    code=lambda s:cells.append(nbf.v4.new_code_cell(s))
    md('''# AI engineering assignment — seven working solutions

**Run all cells in order.** Python 3.11–3.12; no API keys. Internet is needed for initial dependencies and the multilingual Whisper tiny model. Source files, tests and synthetic audio are embedded below, so this notebook does not depend on a Git repository URL or an uploaded ZIP.

The work uses ChatGPT/Codex at **High** assistance level. Fill your name, actual personal time spent and any additional assistance in the final disclosure before submission. All fixtures are synthetic. Read the limitations: this is a tested minimal prototype, not validated clinical/medical or legal decision software.

Audio runs by default. To work entirely offline after installing the core dependencies, set `RUN_AUDIO=False` in the last section; the notebook will explicitly report ASR as not run.''')
    md('## 0. Reconstruct the included project')
    code('''import base64, io, zipfile, os, sys
from pathlib import Path
PROJECT = Path.cwd() / "assignment_workspace"
PROJECT.mkdir(exist_ok=True)
BUNDLE = '''+repr(bundle)+'''
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))) as z:
    for member in z.infolist():
        target = (PROJECT / member.filename).resolve()
        if not target.is_relative_to(PROJECT.resolve()):
            raise ValueError("Unsafe archive path")
    z.extractall(PROJECT)
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))
print("Project files ready:", PROJECT)
''')
    md('## 1. Install dependencies and check the OCR executable')
    code('''import subprocess, shutil, importlib.metadata, platform
requirements = [line.strip() for line in Path("requirements-audio.txt").read_text().splitlines() if line.strip() and not line.startswith("-r")]
requirements += [line.strip() for line in Path("requirements.txt").read_text().splitlines() if line.strip()]
missing = []
for item in requirements:
    name, version = item.split("==")
    try:
        installed = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed != version:
        missing.append(item)
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
if not shutil.which("tesseract"):
    if platform.system() == "Linux" and hasattr(os, "geteuid") and os.geteuid() == 0:
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(["apt-get", "install", "-y", "-qq", "tesseract-ocr", "tesseract-ocr-eng"], check=True)
    else:
        raise RuntimeError("Install the Tesseract system executable; see README setup instructions.")
print(subprocess.check_output(["tesseract", "--version"], text=True).splitlines()[0])
''')
    md('''## Q1. Deduplicate and classify

Latest timestamp wins; first occurrence wins tied timestamps. An audit trail explains discarded records. The continuous REVIEW interval is `[0.50, 0.80)`. Expected result: 3 ACCEPT, 1 REVIEW, 1 RETAKE; average 0.734.''')
    code('''import json
from solution.records import *
def show(value):
    print(json.dumps(value, indent=2, ensure_ascii=False))
show(process_documents(DOCUMENTS))''')
    md('''## Q2. OpenCV quality checks

Laplacian variance >=100, brightness 70–250 and percentile contrast >=30 are initial heuristic thresholds. These are not device-calibrated. Sparse pages, texture and noise can mislead the metrics. Test clear, blurred, dark, rotated and blank synthetic images.''')
    code('''from solution.vision import make_samples, check_quality, OCRPipeline
from IPython.display import display, Image
samples = make_samples("data/images")
show({name: check_quality(image) for name, image in samples.items()})
for name in ("clear", "blurred", "dark"):
    print(name)
    display(Image(filename=f"data/images/{name}.png", width=600))''')
    md('''## Q3. Confidence and auditable corrections

HIGH >=0.90; REVIEW [0.60,0.90); LOW <0.60. Original output is immutable evidence. `twlce daily` is only a proposed correction until reviewed. A dictionary fuzzy match is only a suggestion; the illustrative dictionary below is synthetic, not a trusted clinical vocabulary.''')
    code('''reviewed = review_ocr(OCR_RESULTS)
show(reviewed)
show(approve_correction(reviewed[2], "twice daily", "Synthetic example: reviewer checked source", "demo-reviewer"))
show(lookup_medicine("Paracetamo", {"names": ["Paracetamol"], "version": "demo-v1", "source": "synthetic test only"}))''')
    md('''## Q4. JSON extraction and validation

Missing or ambiguous values become null, duplicate labels are flagged, and swapped strength/frequency values are never guessed. A valid syntax result still requires human review. LLM extension: constrained JSON schema → deterministic validation → literal source-span checks → bounded repair or review; see `docs/DESIGN.md`.''')
    code('''examples = {"valid": SAMPLE_TEXT,
    "missing_medicine": SAMPLE_TEXT.replace("Medicine: Paracetamol\\n", ""),
    "unreadable_duration": SAMPLE_TEXT.replace("5 days", "unreadable"),
    "swapped": SAMPLE_TEXT.replace("Strength: 500 mg", "Strength: Twice daily").replace("Frequency: Twice daily", "Frequency: 500 mg")}
show({name: extract_record(text) for name, text in examples.items()})''')
    md('''## Q5. Actual local OCR pipeline

Quality gate → preprocessing → Tesseract LSTM → bounded rotation fallback → schema/evidence validation → JSON or review. Repeated identical images use a project-scoped bounded cache. This demo uses printed English documents; it does not claim tested handwriting accuracy. No API calls are made.''')
    code('''ocr = OCRPipeline()
for name in ("clear", "blurred", "dark", "rotated"):
    result = ocr.process(samples[name])
    show({"case": name, "route": result["route"], "latency_ms": result["latency_ms"],
          "record": result["structured"]["record"] if result["structured"] else None})
show({"cache_hit": ocr.process(samples["clear"])["cache_hit"]})''')
    md('''## Q6. Isolated multi-project chatbot

The application supplies a trusted principal; this is not a remote authentication system. ACL checks precede retrieval. Conversation memory is user/session/project scoped. Answers quote exact sources. The minimal prototype supports curated fact records and three topics, with abstention on unsupported details. Source files are generated by the final benchmark cell.

Failures demonstrated: incorrect cross-project retrieval, stale topic after switching, missing evidence, conflicting sources, and cross-user session access. Fixes are documented and tested.''')
    code('''from solution.chatbot import KnowledgeBot, Principal
bot = KnowledgeBot()
alice = Principal("alice")
session = bot.start_session(alice, "atlas")
for question in ["What is the leave allowance?", "How many days is that?", "What is the API rate limit?", "What is the salary?", "What is log retention?", "Tell me boreal leave"]:
    show({"question": question, "response": bot.ask(alice, session, question)})
try:
    bot.ask(Principal("bob"), session, "leave")
except PermissionError as error:
    print("Cross-user access correctly rejected:", error)''')
    md('''## Q7. Legal transcript processing

Preserve all originals, negations, uncertainty and timestamps. Remove only confidently irrelevant small talk from the focused view. Extract explicit parties/dates and retain facts as unverified statements with evidence. Match statute + section exactly against the versioned two-entry official-source title index. Candidates are for review, never legal applicability decisions. Hindi/Hinglish text is tested; audio below is synthetic English only.''')
    code('''from solution.legal import LegalPipeline, word_error_rate
database = json.loads(Path("data/legal_db.json").read_text())
legal = LegalPipeline(database)
cases = json.loads(Path("data/legal_cases.json").read_text())
result = legal.analyze(cases, incident_date="2025-08-01")
show({"labels": [{"text": s["original_text"], "label": s["label"], "negated": s["negated"]} for s in result["segments"]],
      "explicit_sections": result["explicit_sections"], "candidate_provisions": result["candidate_provisions"]})''')
    md('## Automated tests — real OCR plus validation, retrieval and legal edge cases')
    code('''completed = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], capture_output=True, text=True)
print(completed.stdout + completed.stderr)
assert completed.returncode == 0, "Tests failed"''')
    md('''## Measured benchmarks, including real local speech inference

The next cell loads multilingual Whisper tiny, then measures clean and noisy 7.62-second English synthetic clips. It reports actual WER and resources. A model download failure is a setup failure, never a zero-error result. Set `RUN_AUDIO=False` only when intentionally skipping ASR. All results are small-fixture measurements, not production accuracy estimates.''')
    code('''RUN_AUDIO = True
MODEL = os.environ.get("ASSIGNMENT_MODEL_PATH", "tiny")
from run_demo import run
results = run(audio=RUN_AUDIO, model=MODEL)
if RUN_AUDIO:
    show({"model_load_ms": results["audio_benchmark"]["model_load_ms"], "cases": [
        {"file": row["file"], "wer": row["wer"], "transcript": row["asr"]["transcript"], "resources": row["resources"]}
        for row in results["audio_benchmark"]["cases"]]})''')
    md('''## Documentation, limitations and candidate disclosure

Read the included design document for each pipeline stage's purpose, inputs/outputs, accuracy trade-offs, latency and API cost. The candidate must add their actual personal time and share a reviewer-accessible notebook/repository link. No personal time or external hosting is fabricated.''')
    code('''from IPython.display import Markdown
display(Markdown(Path("docs/DESIGN.md").read_text()))
display(Markdown(Path("DISCLOSURE.md").read_text()))''')
    nb=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.12"}})
    nbf.write(nb,root/'Assignment_Solutions.ipynb')
    print('Notebook created:',len(cells),'cells')


if __name__=='__main__':
    build()

"""Bundle project files and demos into a self-contained notebook."""
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
    md('''# AI engineering assignment — seven solutions

**Run all cells in order.** Python 3.11–3.12; no API keys. Initial dependency installation and the Whisper tiny download need internet. Source, tests, documentation and synthetic audio are embedded below.

ChatGPT/Codex assistance was **High**; see the final disclosure for candidate details. All fixtures are synthetic. This prototype is not validated for clinical or legal decisions.

Audio runs by default. Set `RUN_AUDIO=False` in the benchmark section to skip inference and the model download. Setup still checks all pinned dependencies, including audio packages. Offline execution needs those packages and Tesseract installed, plus local model weights if audio is enabled.''')
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

Keep the latest timestamp per ID and the first occurrence on ties. Discarded duplicates get audit entries. REVIEW covers `[0.50, 0.80)`. Expect 3 ACCEPT, 1 REVIEW, 1 RETAKE and mean quality 0.734.''')
    code('''import json
from solution.records import *
def show(value):
    print(json.dumps(value, indent=2, ensure_ascii=False))
show(process_documents(DOCUMENTS))''')
    md('''## Q2. OpenCV quality checks

Test clear, blurred, dark, rotated and blank fixtures. The gate requires Laplacian variance >=100, brightness 70–250 and percentile contrast >=30. These uncalibrated thresholds can misjudge sparse or noisy pages.''')
    code('''from solution.vision import make_samples, check_quality, OCRPipeline
from IPython.display import display, Image
samples = make_samples("data/images")
show({name: check_quality(image) for name, image in samples.items()})
for name in ("clear", "blurred", "dark"):
    print(name)
    display(Image(filename=f"data/images/{name}.png", width=600))''')
    md('''## Q3. Confidence and auditable corrections

HIGH >=0.90; REVIEW [0.60,0.90); LOW <0.60. Original text is retained. `twlce daily` needs review before correction. Fuzzy dictionary matches are suggestions; the dictionary below is synthetic.''')
    code('''reviewed = review_ocr(OCR_RESULTS)
show(reviewed)
show(approve_correction(reviewed[2], "twice daily", "Synthetic example: reviewer checked source", "demo-reviewer"))
show(lookup_medicine("Paracetamo", {"names": ["Paracetamol"], "version": "demo-v1", "source": "synthetic test only"}))''')
    md('''## Q4. JSON extraction and validation

Missing, empty and duplicate fields become null. Ambiguous text and suspected swaps are retained and flagged. Even syntax-valid records require review. The design document describes a possible LLM extension with schema and source-span checks.''')
    code('''examples = {"valid": SAMPLE_TEXT,
    "missing_medicine": SAMPLE_TEXT.replace("Medicine: Paracetamol\\n", ""),
    "unreadable_duration": SAMPLE_TEXT.replace("5 days", "unreadable"),
    "swapped": SAMPLE_TEXT.replace("Strength: 500 mg", "Strength: Twice daily").replace("Frequency: Twice daily", "Frequency: 500 mg")}
show({name: extract_record(text) for name, text in examples.items()})''')
    md('''## Q5. Local OCR pipeline

Usable images pass through preprocessing, Tesseract, rotation fallback and field validation. Failed quality checks request recapture; failed extraction checks require review. Identical grayscale images use a bounded project cache. The fixture is printed English; handwriting accuracy is unmeasured. No API is called.''')
    code('''ocr = OCRPipeline()
for name in ("clear", "blurred", "dark", "rotated"):
    result = ocr.process(samples[name])
    show({"case": name, "route": result["route"], "latency_ms": result["latency_ms"],
          "record": result["structured"]["record"] if result["structured"] else None})
show({"cache_hit": ocr.process(samples["clear"])["cache_hit"]})''')
    md('''## Q6. Isolated multi-project chatbot

The application supplies identity; session ownership and project permissions are checked before retrieval. Answers quote curated leave, API or retention facts and abstain on unsupported details. The final benchmark cell exports those facts as Markdown.

The examples exercise follow-up context, topic changes, missing evidence, conflicting sources, project-switch requests and cross-user access rejection.''')
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

Keep original text and timestamps, and flag negation and uncertainty. Only full small-talk matches with adequate ASR confidence leave the focused view. Parties and dates retain segment evidence. Act/section references must match the two-entry title index; matches do not establish applicability. Hindi/Hinglish is tested as text only; audio uses synthetic English.''')
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
    md('''## Benchmarks and local speech inference

The next cell transcribes clean and noisy 7.62-second synthetic English clips with multilingual Whisper tiny and reports WER, timing and resources. Download failures stop execution. `RUN_AUDIO=False` skips speech inference. These fixture measurements do not estimate production accuracy.''')
    code('''RUN_AUDIO = True
MODEL = os.environ.get("ASSIGNMENT_MODEL_PATH", "tiny")
from run_demo import run
results = run(audio=RUN_AUDIO, model=MODEL)
if RUN_AUDIO:
    show({"model_load_ms": results["audio_benchmark"]["model_load_ms"], "cases": [
        {"file": row["file"], "wer": row["wer"], "transcript": row["asr"]["transcript"], "resources": row["resources"]}
        for row in results["audio_benchmark"]["cases"]]})''')
    md('''## Documentation, limitations and candidate disclosure

The design document explains validation, access controls, thresholds and limitations. Confirm the candidate details and actual personal time in the disclosure, and add a reviewer-accessible notebook or repository link.''')
    code('''from IPython.display import Markdown
display(Markdown(Path("docs/DESIGN.md").read_text()))
display(Markdown(Path("DISCLOSURE.md").read_text()))''')
    nb=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.12"}})
    nbf.write(nb,root/'Assignment_Solutions.ipynb')
    print('Notebook created:',len(cells),'cells')


if __name__=='__main__':
    build()

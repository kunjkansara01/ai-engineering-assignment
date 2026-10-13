"""Transparent small-fixture benchmarks; no claim of production accuracy."""
import json
import platform
import statistics
import threading
import time
from pathlib import Path
import psutil


def rss_mb():
    """Current parent RSS; proc/self also works in PID-remapped sandboxes."""
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return float(line.split()[1]) / 1024
    except OSError:
        pass
    try:
        return psutil.Process().memory_info().rss / 1024**2
    except (psutil.Error, OSError):
        pass
    return None


def measure(fn):
    stop = threading.Event()
    samples = []
    def sample():
        while not stop.is_set():
            value = rss_mb()
            if value is not None:
                samples.append(value)
            stop.wait(.01)
    cpu0 = time.process_time()
    start = time.perf_counter()
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        result = fn()
    finally:
        elapsed = time.perf_counter()-start
        stop.set()
        thread.join()
    cpu_seconds = time.process_time()-cpu0
    return result, {"wall_ms": round(elapsed*1000, 3), "parent_cpu_seconds": round(cpu_seconds, 4),
                    "parent_cpu_percent_one_core": round(100*cpu_seconds/elapsed, 2),
                    "sampled_peak_parent_rss_mb": round(max(samples), 2) if samples else None,
                    "memory_sampling_interval_ms": 10, "api_cost_usd": 0.}


def benchmark_text_legal(pipeline, cases, repeats=30):
    def run():
        timings = []
        for _ in range(repeats):
            result = pipeline.analyze(cases, incident_date='2025-08-01')
            timings.append(result['latency_ms'])
        expected = [s['expected'] for s in cases]
        actual = [s['label'] for s in result['segments']]
        labels = ['LEGALLY_RELEVANT','IRRELEVANT','UNCERTAIN']
        confusion = {a:{b:sum(e==a and p==b for e,p in zip(expected,actual)) for b in labels} for a in labels}
        return {"fixture_count":len(cases), "repeats":repeats,
                "label_accuracy":sum(e==p for e,p in zip(expected,actual))/len(cases),
                "confusion_matrix":confusion, "p50_ms":statistics.median(timings),
                "p95_ms":sorted(timings)[int(.95*(len(timings)-1))],
                "scope":"Synthetic hand-labelled transcript fixtures; not audio ASR accuracy or an independent evaluation set."}
    result, metrics = measure(run)
    result['resources'] = metrics
    return result


def environment():
    import cv2, numpy, pytesseract
    return {"python":platform.python_version(),"platform":platform.platform(),
            "logical_cpus":psutil.cpu_count(),"opencv":cv2.__version__,"numpy":numpy.__version__,
            "tesseract":str(pytesseract.get_tesseract_version())}

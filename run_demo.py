"""Run from the project directory: python run_demo.py [--audio --model tiny]."""
import argparse
import json
import statistics
from pathlib import Path
from solution.records import *
from solution.vision import *
from solution.chatbot import KnowledgeBot, Principal, demo_documents
from solution.legal import LegalPipeline, transcribe_local, word_error_rate
from solution.benchmark import benchmark_text_legal, measure, environment


def save(name, value):
    Path('outputs').mkdir(exist_ok=True)
    Path('outputs',name).write_text(json.dumps(value,indent=2,ensure_ascii=False))


def run(audio=False, model='tiny'):
    results={}
    results['q1']=process_documents(DOCUMENTS)
    samples=make_samples('data/images')
    results['q2']={name:check_quality(image) for name,image in samples.items()}
    results['q3']=review_ocr(OCR_RESULTS)
    results['q4']={
        'valid':extract_record(SAMPLE_TEXT),
        'missing_medicine':extract_record(SAMPLE_TEXT.replace('Medicine: Paracetamol\n','')),
        'unreadable_duration':extract_record(SAMPLE_TEXT.replace('5 days','unreadable')),
        'swapped':extract_record(SAMPLE_TEXT.replace('Strength: 500 mg','Strength: Twice daily').replace('Frequency: Twice daily','Frequency: 500 mg'))}
    pipeline=OCRPipeline()
    results['q5']={name:pipeline.process(image) for name,image in samples.items()}
    results['q5']['cached_clear']=pipeline.process(samples['clear'])
    cold_times=[]
    for _ in range(5):
        result=OCRPipeline(max_cache=0).process(samples['clear'])
        cold_times.append(result['latency_ms'])
    warm_times=[pipeline.process(samples['clear'])['latency_ms'] for _ in range(20)]
    expected=extract_record(SAMPLE_TEXT)['record']
    actual=results['q5']['clear']['structured']['record']
    results['ocr_benchmark']={
        'dataset':'Synthetic printed English document, same fixture as functional tests; not handwriting evaluation',
        'clear_field_exact_match':sum(expected[f]==actual[f] for f in FIELDS.values())/len(FIELDS),
        'clear_text_wer':word_error_rate(SAMPLE_TEXT,results['q5']['clear']['extraction']['text']),
        'uncached_ms':cold_times,'uncached_p50_ms':statistics.median(cold_times),
        'cached_p50_ms':statistics.median(warm_times),'cached_n':len(warm_times),'api_cost_usd':0.}
    bot=KnowledgeBot()
    principal=Principal('alice')
    sid=bot.start_session(principal,'atlas')
    questions=['What is the annual leave allowance?','How many days is that?','What is the API rate limit?',
               'What is the salary?','What is log retention?','Tell me boreal leave',
               'Can I carry over leave?']
    results['q6']=[{'question':q,'response':bot.ask(principal,sid,q)} for q in questions]
    try:
        bot.ask(Principal('bob'),sid,'leave')
    except PermissionError as e:
        results['q6'].append({'question':'Bob reuses Alice session','error':str(e)})
    Path('data/knowledge').mkdir(exist_ok=True)
    for d in demo_documents():
        path=Path('data/knowledge',d['source'].split('#')[0])
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('# '+d['topic']+'\n\n'+d['text']+'\n')
    db=json.loads(Path('data/legal_db.json').read_text())
    cases=json.loads(Path('data/legal_cases.json').read_text())
    legal=LegalPipeline(db)
    results['q7']=legal.analyze(cases,incident_date='2025-08-01')
    results['legal_text_benchmark']=benchmark_text_legal(legal,cases)
    results['audio_benchmark']={'status':'NOT_RUN','reason':'Run with --audio for actual local ASR; transcript fixtures are not ASR results.'}
    if audio:
        # Load once and reuse across recordings; separate startup from inference.
        import time
        from faster_whisper import WhisperModel
        t0=time.perf_counter()
        speech_model=WhisperModel(model,device='cpu',compute_type='int8',cpu_threads=2)
        load_ms=(time.perf_counter()-t0)*1000
        audio_results=[]
        for filename in ['legal_clear.wav','legal_noisy.wav']:
            asr, metrics=measure(lambda:transcribe_local(Path('data/audio',filename),model,'en',speech_model))
            reference=Path('data/audio/reference.txt').read_text()
            asr['model']='tiny multilingual (CPU int8, two threads)'
            audio_results.append({'file':filename,'asr':asr,'resources':metrics,
                                  'wer':word_error_rate(reference,asr['transcript']),
                                  'legal':legal.analyze(asr['segments'],incident_date='2025-08-01')})
        results['audio_benchmark']={'status':'MEASURED','model_load_ms':load_ms,'cases':audio_results,
                                    'scope':'Two synthetic English speech clips (clean and noisy). No Hindi/Hinglish audio or real conversation accuracy claim.'}
    results['environment']=environment()
    for key,value in results.items():
        save(key+'.json',value)
    print(json.dumps({'q1':results['q1']['counts'],'average_quality':results['q1']['average_quality'],
                      'image_routes':{k:v['route'] for k,v in results['q5'].items()},
                      'ocr_benchmark':results['ocr_benchmark'],
                      'legal_text_benchmark':results['legal_text_benchmark'],
                      'audio_status':results['audio_benchmark']['status']},indent=2))
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--audio',action='store_true')
    parser.add_argument('--model',default='tiny')
    args=parser.parse_args()
    run(args.audio,args.model)

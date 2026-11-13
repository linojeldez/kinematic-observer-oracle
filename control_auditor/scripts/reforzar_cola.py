"""Adelanta los últimos 144 trabajos previstos sin cambiar diseño ni motor.

Los calcula en otro directorio y publica mediante enlace atómico. El ejecutor
principal los reconoce como checkpoints compatibles; nunca comparten temporales.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import time
import ejecutar_respuestas as campaign

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / 'resultados/refuerzo_temporal'

def one(job, hashes):
    campaign.ROOT = SCRATCH
    record_path = SCRATCH / 'resultados/corridas' / campaign.filename(job)
    target = ROOT / 'resultados/corridas' / campaign.filename(job)
    if target.exists():
        old = json.loads(target.read_text())
        assert old['job'] == job and old['source_hashes'] == hashes
        return 'already_completed'
    result = campaign.worker(job, hashes)
    try:
        os.link(record_path, target)
        state = 'published'
    except FileExistsError:
        old = json.loads(target.read_text())
        fresh = json.loads(record_path.read_text())
        assert old['job'] == fresh['job'] and old['source_hashes'] == fresh['source_hashes']
        assert abs(old['result']['R'] - fresh['result']['R']) < 2e-12
        state = 'compatible_overlap'
    record_path.unlink()
    return state

def main():
    jobs, _, _ = campaign.build_jobs('matched')
    jobs = jobs[-144:]
    hashes = {str(p): campaign.sha(p) for p in campaign.ENGINE_FILES}
    (SCRATCH / 'resultados/corridas').mkdir(parents=True, exist_ok=True)
    counts = {}
    started = time.time()
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(one, job, hashes) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            state = future.result()
            counts[state] = counts.get(state, 0) + 1
            if index % 12 == 0:
                print(index, '/', len(jobs), counts, flush=True)
    report = dict(finished=True, jobs=len(jobs), counts=counts, elapsed=time.time()-started,
                  changed_design=False, changed_motor=False)
    (ROOT / 'logs/refuerzo_cola.json').write_text(json.dumps(report, indent=2))
    print(report, flush=True)

if __name__ == '__main__':
    main()

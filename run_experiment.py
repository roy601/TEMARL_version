"""Portable, resumable six-arm comparison plus no-intent controls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

ROOT = Path(__file__).resolve().parent


def fingerprint():
    h = hashlib.sha256()
    for p in sorted(ROOT.rglob('*')):
        rel = p.relative_to(ROOT)
        if p.is_file() and p.suffix in ('.py', '.json') and rel.parts[0] not in ('results', '.venv'):
            h.update(rel.as_posix().encode())
            h.update(p.read_bytes().replace(b'\r\n', b'\n'))
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
    os.replace(tmp, path)


def device_name(requested):
    import torch
    if requested == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Install a CUDA-enabled PyTorch build.')
    return requested


def configure_smoke():
    from mappo import HP
    from pretrain_marl import PRETRAIN
    HP.update(n_envs=2, rollout_len=8, ppo_epochs=1, n_evals=1, val_episodes=8)
    PRETRAIN.update(episodes=8, steps=2, batch=8, val_episodes=8)


def cell_plan(study, seeds):
    import experiment as spec
    from env_marl import SCRIPT_IDS
    folds = [None] if study == 'main' else list(SCRIPT_IDS)
    return [dict(encoder=e, learner=l, seed=s, held_out=fold)
            for fold in folds for e, l in spec.MAIN_ARMS + spec.CONTROL_ARMS for s in seeds]


def cell_name(c):
    return f"{c['encoder']}_{c['learner']}_s{c['seed']}" + (f"_{c['held_out']}" if c['held_out'] else '')


def run(study='main', smoke=False, device='auto', seeds=None, only=None, output=None):
    import torch
    import numpy as np
    import _frozen
    import experiment as spec
    from env_marl import EnvConfig, SCRIPT_IDS
    from encoders_marl import arm_specs
    from mappo import train, summary, SEED_TEST, HP
    from pretrain_marl import pretrain, PRETRAIN
    if smoke:
        configure_smoke()
    device = device_name(device)
    seeds = list(seeds if seeds is not None else ([0] if smoke else
                 (spec.SEEDS if study == 'main' else spec.LOSO_SEEDS)))
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 or s >= 1000 for s in seeds):
        raise ValueError('Use unique seeds in 0..999 to keep episode streams disjoint')
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
    out = Path(output) if output else ROOT / 'results' / (('smoke_' if smoke else '') + study)
    out.mkdir(parents=True, exist_ok=True)
    if (out / '.running').exists():
        raise RuntimeError(f'{out} has a .running lock; check for an active run before removing it')
    cfg = EnvConfig(**json.loads((ROOT / 'env_config.json').read_text())['config'])
    manifest = dict(fingerprint=fingerprint(), smoke=smoke, study=study, seeds=seeds,
                    budget=16 if smoke else spec.ENV_STEPS, n_test=8 if smoke else spec.N_TEST,
                    env_config=cfg.to_dict(), encoders=arm_specs(), ppo=HP, pretrain=PRETRAIN,
                    device=device, selection=spec.SELECTION,
                    expected_cells=cell_plan(study, seeds))
    mp = out / 'manifest.json'
    if mp.exists() and json.loads(mp.read_text()) != manifest:
        raise RuntimeError('Existing run has different code/config/seeds/device; use a new output folder')
    write_json(mp, manifest)
    runtime = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                   device=device, gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None)
    lock = out / '.running'
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    try:
        for c in manifest['expected_cells']:
            if only and c['encoder'] not in only:
                continue
            name = cell_name(c)
            dest, checkpoint = out / 'cells' / (name + '.json'), out / 'checkpoints' / (name + '.pt')
            if dest.exists():
                prior = json.loads(dest.read_text())
                if (prior['fingerprint'] != manifest['fingerprint'] or not checkpoint.exists()
                        or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != prior['checkpoint_sha256']):
                    raise RuntimeError(f'Invalid/stale completed cell: {name}')
                print('Already complete:', name, flush=True)
                continue
            pool = [s for s in SCRIPT_IDS if s != c['held_out']]
            test_pool = [c['held_out']] if c['held_out'] else pool
            test_seed = SEED_TEST + c['seed'] + (10000 * (1 + SCRIPT_IDS.index(c['held_out'])) if c['held_out'] else 0)
            print('Starting:', name, flush=True)
            start = time.time()
            pre = pretrain(c['encoder'], c['seed'], pool, device, verbose=True)
            learned = train(pre['encoder'], cfg, pool, c['seed'], manifest['budget'],
                            centralised=c['learner'] == 'MAPPO', device=device,
                            test_pool=test_pool, n_test=manifest['n_test'], test_seed=test_seed,
                            verbose=True)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            tmp = checkpoint.with_suffix('.pt.tmp')
            torch.save(dict(actor=learned['actor'].state_dict(),
                            encoder=None if pre['encoder'] is None else pre['encoder'].state_dict(),
                            cell=c, fingerprint=manifest['fingerprint'], config=cfg.to_dict(),
                            encoder_specs=arm_specs()), tmp)
            os.replace(tmp, checkpoint)
            rec = dict(cell=c, fingerprint=manifest['fingerprint'], smoke=smoke, runtime=runtime,
                       checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                       seconds=time.time()-start, env_steps=learned['env_steps'],
                       best_update=learned['best_update'], val_best=learned['val_best'],
                       curve=learned['curve'], pretrain=pre['metrics'],
                       actor_params=learned['actor_params'], critic_params=learned['critic_params'],
                       test=learned['test_rows'], test_summary=summary(learned['test_rows']),
                       test_h0=learned['test_rows_h0'], test_h0_summary=summary(learned['test_rows_h0']))
            write_json(dest, rec)
            print('Completed:', name, rec['test_summary'], flush=True)
    finally:
        lock.unlink(missing_ok=True)
    from report_results import generate
    generate(out)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', choices=['main', 'loso'], default='main')
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--seeds', type=int, nargs='+')
    ap.add_argument('--only', nargs='+', choices=['Transformer', 'GRU', 'LSTM', 'NoHistory'])
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    if args.threads < 1:
        ap.error('--threads must be positive')
    import torch
    torch.set_num_threads(args.threads)
    run(args.study, args.smoke, args.device, args.seeds, args.only, args.output)

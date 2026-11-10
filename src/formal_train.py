# -*- coding: utf-8 -*-
"""standalone formal-experiment trainer (Linux/Windows both OK).

Why this file exists: the original pipeline (reviewed5250_three15 / abc15) is a
chain of "increment on the previous experiment" scripts with hard asserts on the
old 15-epoch run, plus msvcrt / ctypes.windll / creationflags in the queue path.
On a fresh cloud box none of that exists, so this trainer reproduces exactly the
same training loop (AdamW + backbone lr_scale .5, EMA .9999/2000w, epoch-wise
linear decay, clip 10, fliplr .5) without any of those dependencies.

It writes the same artefacts the old pipeline did:
    runs/<tag>/weights/{last.pth,best.pth}
    runs/<tag>/results.csv
    runs/<tag>/configuration.json
    runs/<tag>/evaluation.json          (after --eval-only / auto post-train eval)
and is safely resumable: re-run the same command, it continues from last.pth.

Usage:
    python 03_train.py --model pico    --epochs 120 --patience 35 --batch 16
    python 03_train.py --model pico_ab --epochs 120 --patience 35 --batch 16
    python 03_train.py --model yolov8n --epochs 120 --patience 35 --batch 16
    python 03_train.py --model pico_ab --eval-only      # only re-evaluate best
    python 03_train.py --model pico --limit 40 --epochs 1   # plumbing smoke test
"""
import argparse, csv, json, os, random, sys, time
from pathlib import Path

WORK = Path(os.environ.get('GUIYU_ROOT', '/home/waas/GUIYU')).resolve()
DATA = WORK / 'exports/reviewed5250_20261027_v1'
TAG = os.environ.get('FORMAL_TAG', 'formal50_20261030_v1')

prov = json.loads((WORK / 'runtime/external/deimv2_source_provenance.json').read_text(encoding='utf-8'))
DFINE = WORK / 'runtime/external/D-FINE-956d1709314c'
for p in [str(WORK), str(WORK / 'tools'), str(Path(prov['source_directory'])), str(DFINE)]:
    if Path(p).is_dir():
        sys.path.insert(0, p)
os.environ.update(YOLO_CONFIG_DIR=str(WORK / 'runtime/ultralytics'), TORCH_HOME=str(WORK / 'runtime/torch'),
                  HF_HOME=str(WORK / 'runtime/huggingface'), HF_HUB_OFFLINE='1', PYTHONUNBUFFERED='1')
import types
sys.modules.setdefault('tensorboard.compat.notf', types.ModuleType('tensorboard.compat.notf'))

import wall20_models_v1 as wm
import short20_deim_core as core
wm.SPECS['pico_b_mask'] = ('rgbd_mask', False, 'mask')

def build_architecture(key, pretrained=True):
    if key == 'pico_ab_cspatial_control':
        import models_spatial
        return models_spatial.build_model(key, pretrained)
    return wm.build_model(key, pretrained)


def run_dir(key):
    return WORK / 'runs' / f'{TAG}_{key}_seed42'


def build_datasets(key, batch, limit=0):
    """Same PairedDataset wiring as wall20.context(), minus the Windows bits."""
    import yaml
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import ops  # noqa: F401  (import side effect parity)
    e = core.load_module('eval_helpers', WORK / 'tools/evaluate_paired_baselines.py')
    _, _, meta = build_architecture(key, pretrained=False)
    mode = 'rgb' if meta['input_channels'] == 3 else 'rgbd_mask'
    data = yaml.safe_load((DATA / f'{mode}.yaml').read_text(encoding='utf-8'))
    args = get_cfg(overrides={'imgsz': 960, 'fliplr': .5, 'flipud': 0., 'fraction': 1., 'task': 'detect'})
    ds = []
    for split in ['train', 'val']:
        d = e.paired.PairedDataset(img_path=str(Path(data['path']) / data[split]), imgsz=960, batch_size=batch,
                                   augment=split == 'train', hyp=args, rect=False, cache=False, single_cls=False,
                                   stride=32, pad=0., prefix=f'{key} {split}: ', task='detect',
                                   classes=None, data=data, fraction=1.)
        if split == 'train' and limit:
            n = min(limit, len(d))
            d.im_files = d.im_files[:n]
            if getattr(d, 'labels', None) is not None:
                d.labels = d.labels[:n]
        ds.append(d)
    return e, data, ds[0], ds[1], meta


def validate(key, net, cfg, e, data, valset, dest):
    """Same ReviewValidator flow as core.validate(), without its 400/1065 asserts."""
    import numpy as np, torch
    from torch.utils.data import DataLoader
    loader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0, collate_fn=valset.collate_fn)
    v = core.new_validator(e, data, dest, 'cuda')
    net.eval(); post = cfg.postprocessor.cuda().eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = core.move_batch(batch, 'cuda')
            preds = core.decode(net(batch['img']), post)
            v.update_metrics(preds, batch)
            if (i + 1) % 100 == 0:
                print('validation', i + 1, len(valset), flush=True)
    return core.finish_metrics(v)


def train_one(key, a):
    import numpy as np, torch
    from torch.utils.data import DataLoader
    from engine.optim.ema import ModelEMA

    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = False
    dest = run_dir(key); (dest / 'weights').mkdir(parents=True, exist_ok=True)
    last, best_p, hist = dest / 'weights/last.pth', dest / 'weights/best.pth', dest / 'results.csv'

    net, cfg, meta = build_architecture(key, pretrained=True)
    net = net.cuda(); crit = cfg.criterion.cuda()
    bb = [p for n, p in net.named_parameters() if n.startswith('backbone.') and p.requires_grad]
    rest = [p for n, p in net.named_parameters() if not n.startswith('backbone.') and p.requires_grad]
    opt = torch.optim.AdamW([{'params': bb, 'lr_scale': .5}, {'params': rest, 'lr_scale': 1.}],
                            lr=a.lr0, betas=(.9, .999), weight_decay=1e-4)
    ema = ModelEMA(net, decay=.9999, warmups=2000)
    start, best, best_ep, elapsed_before = 0, -1., -1, 0.
    if last.exists():
        st = torch.load(last, map_location='cpu', weights_only=False)
        net.load_state_dict(st['model']); opt.load_state_dict(st['optimizer']); ema.load_state_dict(st['ema'])
        start = st['epoch'] + 1; best = st['best_fitness']; best_ep = st['best_epoch']; elapsed_before = st['elapsed']
        random.setstate(st['python_rng']); np.random.set_state(st['numpy_rng'])
        torch.set_rng_state(st['torch_rng']); torch.cuda.set_rng_state_all(st['cuda_rng'])
        del st
        if hist.exists():
            rows = list(csv.DictReader(hist.open(encoding='utf-8-sig')))
            rows = [r for r in rows if int(r['epoch']) <= start]
            with hist.open('w', encoding='utf-8', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['epoch','time','train/total_loss','metrics/mAP50(B)','metrics/mAP50-95(B)','metrics/count_MAE_conf025','lr/pg0']); w.writeheader(); w.writerows(rows)
        print(f'RESUME from epoch {start + 1}', flush=True)

    cfgout = {'model': key, 'tag': TAG, 'epochs': a.epochs, 'patience': a.patience, 'batch': a.batch,
              'accum': a.accum, 'lr0': a.lr0, 'lrf': a.lrf, 'warmup_epochs': a.warmup, 'imgsz': 960,
              'seed': a.seed, 'optimizer': 'AdamW', 'weight_decay': 1e-4, 'clip_grad_norm': 10.,
              'amp': False, 'fliplr': .5, 'parameters': meta['parameters'],
              'input_channels': meta['input_channels'], 'started_at': time.strftime('%F %T')}
    (dest / 'configuration.json').write_text(json.dumps(cfgout, ensure_ascii=False, indent=2), encoding='utf-8')
    if start >= a.epochs:
        print('already finished'); return

    e, data, trainset, valset, _ = build_datasets(key, a.batch, a.limit)
    loader = DataLoader(trainset, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                        collate_fn=trainset.collate_fn, drop_last=False)
    cols = ['epoch', 'time', 'train/total_loss', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
            'metrics/count_MAE_conf025', 'lr/pg0']
    if not hist.exists():
        with hist.open('w', encoding='utf-8', newline='') as f:
            csv.DictWriter(f, fieldnames=cols).writeheader()

    t_session = time.perf_counter()
    for epoch in range(start, a.epochs):
        net.train(); crit.train(); opt.zero_grad(set_to_none=True)
        t_ep = time.perf_counter(); tot = 0.
        n = len(loader)
        for i, batch in enumerate(loader):
            lr = a.lr0 * ((1 - epoch / a.epochs) * (1 - a.lrf) + a.lrf)
            warm = min(1., (epoch + (i + 1) / n) / a.warmup)
            for g in opt.param_groups:
                g['lr'] = lr * warm * g['lr_scale']
            batch = core.move_batch(batch, 'cuda'); tg = core.targets_from_batch(batch, 'cuda')
            out = net(batch['img'], tg)
            loss = sum(crit(out, tg, epoch=epoch).values())
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite loss')
            group = min(a.accum, n - (i // a.accum) * a.accum)
            (loss / group).backward(); tot += float(loss.detach())
            if (i + 1) % a.accum == 0 or i + 1 == n:
                gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 10.)
                if not torch.isfinite(gn):
                    raise RuntimeError('non-finite grad')
                opt.step(); opt.zero_grad(set_to_none=True); ema.update(net)
            if i == 0 or (i + 1) % 50 == 0:
                el = time.perf_counter() - t_ep
                print(json.dumps({'epoch': epoch + 1, 'batch': i + 1, 'of': n,
                                  'loss': round(tot / (i + 1), 4),
                                  'eta_epoch_min': round(el / (i + 1) * n / 60, 1)}), flush=True)

        val = validate(key, ema.module, cfg, e, data, valset, dest / 'epoch_validation')
        m = val['metrics']; fit = m['metrics/mAP50-95(B)']
        improved = fit > best
        if improved:
            best, best_ep = fit, epoch
        el = elapsed_before + time.perf_counter() - t_session
        row = {'epoch': epoch + 1, 'time': round(el, 2), 'train/total_loss': tot / n,
               **{k: m[k] for k in cols if k.startswith('metrics/')}, 'lr/pg0': opt.param_groups[0]['lr']}
        with hist.open('a', encoding='utf-8', newline='') as f:
            csv.DictWriter(f, fieldnames=cols).writerow(row)
        state = {'model': net.state_dict(), 'ema': ema.state_dict(), 'optimizer': opt.state_dict(),
                 'epoch': epoch, 'best_fitness': best, 'best_epoch': best_ep, 'elapsed': el,
                 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}
        torch.save(state, last.with_suffix('.tmp'))
        last.with_suffix('.tmp').replace(last)
        if improved:
            torch.save(state, best_p.with_suffix('.tmp'))
            best_p.with_suffix('.tmp').replace(best_p)
        print('EPOCH_COMPLETE', json.dumps(row), flush=True)
        if epoch - best_ep >= a.patience:
            print(f'EARLY_STOP at epoch {epoch + 1} (best {best_ep + 1})', flush=True)
            break


def evaluate(key, a):
    """dump per-frame predictions so P/R/F1 can be recomputed at any threshold."""
    import torch
    dest = run_dir(key); wp = dest / 'weights/best.pth'
    if not wp.exists():
        print('no best.pth, skip eval'); return
    net, cfg, meta = build_architecture(key, pretrained=False)
    net.load_state_dict(torch.load(wp, map_location='cpu', weights_only=False)['ema']['module'])
    net.cuda().eval()
    e, data, _, valset, _ = build_datasets(key, 1)
    res = validate(key, net, cfg, e, data, valset, dest / 'final_validation')
    idx = json.loads((DATA / 'paired_index.json').read_text(encoding='utf-8-sig'))
    vboxes = sum(r['box_count'] for r in idx['records'] if r['split'] == 'val')
    print(f"val frames={len(res['rows'])} boxes={sum(r['reference_count'] for r in res['rows'])} expected={vboxes}")
    (dest / 'evaluation.json').write_text(json.dumps({
        'model': key, 'images': len(res['rows']),
        'reference_boxes': sum(r['reference_count'] for r in res['rows']),
        'metrics': res['metrics'], 'records': res['rows'], 'parameters': meta['parameters'],
        'tag': TAG, 'test_used': False,
    }, ensure_ascii=False), encoding='utf-8')
    print('wrote', dest / 'evaluation.json')


def train_yolo(a):
    from ultralytics import YOLO
    dest = run_dir('yolov8n'); dest.mkdir(parents=True, exist_ok=True)
    w = WORK / 'runtime/weights/yolov8n.pt'
    last = dest / 'weights/last.pt'
    m = YOLO(str(last) if last.exists() else str(w))
    m.train(data=str(DATA / 'rgb.yaml'), epochs=a.epochs, patience=a.patience, imgsz=960, batch=a.batch,
            device='0', workers=a.workers, optimizer='AdamW', lr0=a.lr0, lrf=a.lrf, momentum=.9, nbs=a.batch*a.accum,
            weight_decay=1e-4, warmup_epochs=a.warmup, amp=False, deterministic=True, seed=a.seed,
            project=str(WORK / 'runs'), name=dest.name, exist_ok=True, resume=last.exists(),
            mosaic=0., mixup=0., copy_paste=0., hsv_h=0., hsv_s=0., hsv_v=0.,
            degrees=0., translate=0., scale=0., shear=0., perspective=0., fliplr=.5, flipud=0.,
            multi_scale=False, close_mosaic=0, plots=False, val=True, iou=.5, conf=.001, max_det=300)
    best = dest / 'weights/best.pt'
    if best.exists():
        YOLO(str(best)).val(data=str(DATA / 'rgb.yaml'), imgsz=960, batch=1, device='0',
                            conf=.001, iou=.5, max_det=300, plots=False, save_json=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True,
                    choices=['pico', 'pico_a', 'pico_b_mask', 'pico_a_mask', 'pico_ab_cspatial_control', 'yolov8n'])
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--patience', type=int, default=50)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--accum', type=int, default=4)
    ap.add_argument('--lr0', type=float, default=2e-4)
    ap.add_argument('--lrf', type=float, default=.05)
    ap.add_argument('--warmup', type=float, default=2.)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0, help='only first N train images (smoke test)')
    ap.add_argument('--eval-only', action='store_true')
    a = ap.parse_args()
    print('WORK =', WORK, '| data =', DATA.exists(), '| model =', a.model, flush=True)
    if not DATA.exists():
        sys.exit('DATA missing: run 04_fix_paths.py after extracting the tar')
    if a.model == 'yolov8n':
        if not a.eval_only:
            train_yolo(a)
        return
    if a.eval_only:
        evaluate(a.model, a); return
    train_one(a.model, a)
    evaluate(a.model, a)


if __name__ == '__main__':
    main()





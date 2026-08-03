#!/usr/bin/env python3
"""
compute_cost.py  ->  parameter count, FLOPs, latency, throughput and peak memory.

asks for parameter count, FLOPs, peak GPU memory, training time,
inference time per breast and per patient, and throughput, for TomoMamba and
each baseline. analyze_checkpoint.py measures only MambaCenterNet, because it
cannot build the models in centernet_baselines.py. This does all four.

METHOD, DELIBERATELY IDENTICAL TO THE PUBLISHED NUMBER
    FLOPs are counted with thop as 2 x MACs, exactly as analyze_checkpoint.py
    does, so the TomoMamba row must reproduce the 460.9 GFLOPs/volume already
    in the reported table. If it does not, something differs and the whole table
    is suspect. That check is printed.

RUN IT WHEN THE GPU IS IDLE
    Latency and peak memory are meaningless under contention. If a training
    job is running, these numbers will be wrong. The script warns if it sees
    memory already allocated.

DERIVED QUANTITIES
    A breast contributes 2 views (CC + MLO) = 2 volumes.
    A patient contributes 2 breasts = 4 volumes.
    Throughput is volumes per second at batch size 1.

USAGE
    source ~/tomomamba/bin/activate
    cd ~/DBT/MambaCenterNet_v5.2
    python compute_cost.py --out compute_cost.json
"""

import json
import time
import argparse

import torch


def measure(model, vol, n_warmup, n_iter, device):
    """Returns (ms_per_volume, peak_mb, gflops, params)."""
    model.eval().to(device)
    params = sum(p.numel() for p in model.parameters())

    gflops = None
    try:
        from thop import profile
        # thop mutates the module, so profile a copy of the forward pass only
        macs, _ = profile(model, inputs=({'volume': vol},), verbose=False)
        gflops = float(2 * macs / 1e9)          # FLOPs ~= 2 x MACs
    except Exception as e:
        print(f'      [warn] FLOPs unavailable: {type(e).__name__}: {e}')

    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for _ in range(n_warmup):
            model({'volume': vol})
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            model({'volume': vol})
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    ms = (t1 - t0) / n_iter * 1000.0
    peak = (torch.cuda.max_memory_allocated() / 1024 ** 2
            if device.type == 'cuda' else float('nan'))
    return ms, peak, gflops, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spatial_size', type=int, default=384)
    ap.add_argument('--n_slices', type=int, default=15)
    ap.add_argument('--n_warmup', type=int, default=10)
    ap.add_argument('--n_iter', type=int, default=50)
    ap.add_argument('--dropout', type=float, default=0.7)
    ap.add_argument('--out', default='compute_cost.json')
    args = ap.parse_args()

    from centernet_models import MambaCenterNet
    from centernet_baselines import create_baseline

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        used = torch.cuda.memory_allocated() / 1024 ** 2
        free_b, total_b = torch.cuda.mem_get_info()
        busy = (total_b - free_b) / 1024 ** 2
        if busy > 500:
            print(f'  [WARNING] {busy:.0f} MB already in use on this GPU.')
            print(f'  [WARNING] Another job is running. Latency and peak-memory')
            print(f'  [WARNING] numbers will be WRONG. Wait for the GPU to idle.\n')

    vol = torch.randn(1, args.n_slices, args.spatial_size,
                      args.spatial_size, device=device)

    builders = [
        ('ResNet-18 classifier',      lambda: create_baseline(1, num_classes=2, dropout=args.dropout, spatial_size=args.spatial_size)),
        ('ResNet-18 + CenterNet',     lambda: create_baseline(2, num_classes=2, dropout=args.dropout, spatial_size=args.spatial_size)),
        ('ResNet-18 + BiGRU + CN',    lambda: create_baseline(3, num_classes=2, dropout=args.dropout, spatial_size=args.spatial_size)),
        ('ResNet-18 + Transformer + CN', lambda: create_baseline(4, num_classes=2, dropout=args.dropout, spatial_size=args.spatial_size)),
        ('TomoMamba',                 lambda: MambaCenterNet(num_classes=2, dropout=args.dropout, use_mamba=True, spatial_size=args.spatial_size, backbone='resnet18')),
    ]

    print('=' * 82)
    print(f'  COMPUTE COST  |  input 1 x {args.n_slices} x {args.spatial_size}'
          f'^2  |  {args.n_iter} timed iterations after {args.n_warmup} warmup')
    print('=' * 82)

    rows = {}
    for name, build in builders:
        print(f'\n  {name}')
        try:
            model = build()
        except Exception as e:
            print(f'      [skip] could not build: {type(e).__name__}: {e}')
            continue
        try:
            ms, peak, gflops, params = measure(model, vol, args.n_warmup,
                                               args.n_iter, device)
        except Exception as e:
            print(f'      [skip] measurement failed: {type(e).__name__}: {e}')
            del model
            torch.cuda.empty_cache()
            continue

        rows[name] = {
            'params': params,
            'gflops_per_volume': gflops,
            'ms_per_volume': ms,
            'ms_per_breast': ms * 2,          # CC + MLO
            'ms_per_patient': ms * 4,         # 2 breasts
            'volumes_per_second': 1000.0 / ms if ms > 0 else None,
            'peak_mb': peak,
        }
        g = 'n/a' if gflops is None else f'{gflops:.1f}'
        print(f'      params {params:,}  |  FLOPs {g} G/vol  |  '
              f'{ms:.1f} ms/vol  |  peak {peak:.0f} MB')
        print(f'      per breast {ms*2:.1f} ms  |  per patient {ms*4:.1f} ms  '
              f'|  throughput {1000.0/ms:.2f} vol/s')

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print('\n' + '=' * 82)
    print(f'  {"model":<26}{"params":>12}{"GFLOPs":>10}{"ms/vol":>9}'
          f'{"ms/patient":>12}{"vol/s":>8}{"peak MB":>10}')
    for name, r in rows.items():
        g = '  n/a' if r['gflops_per_volume'] is None else f'{r["gflops_per_volume"]:.1f}'
        print(f'  {name:<26}{r["params"]:>12,}{g:>10}{r["ms_per_volume"]:>9.1f}'
              f'{r["ms_per_patient"]:>12.1f}{r["volumes_per_second"]:>8.2f}'
              f'{r["peak_mb"]:>10.0f}')

    # sanity check against the published figure
    tm = rows.get('TomoMamba')
    if tm and tm['gflops_per_volume']:
        d = abs(tm['gflops_per_volume'] - 460.9)
        verdict = 'MATCHES the published 460.9' if d < 5 else \
                  f'DIFFERS from the published 460.9 by {d:.1f} -- investigate'
        print(f'\n  TomoMamba FLOPs check: {tm["gflops_per_volume"]:.1f} G/vol, '
              f'{verdict}')

    json.dump({'input_shape': [1, args.n_slices, args.spatial_size,
                               args.spatial_size],
               'n_iter': args.n_iter, 'device': str(device),
               'models': rows}, open(args.out, 'w'), indent=2)
    print(f'\n  written: {args.out}')
    print('\n  Note: training times are in each run\'s log, not measured here.')


if __name__ == '__main__':
    main()

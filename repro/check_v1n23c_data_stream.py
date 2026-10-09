"""Read first two real batches with production 8-worker protocol; no optimizer."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'LFMN'))


def main():
    import torch
    from option import args
    from data import Data
    if args.n_threads != 8 or args.data_range != '1-800/801-900':
        raise ValueError('Production stream audit requires 8 workers and full range')
    torch.manual_seed(args.seed)
    loader = Data(args).loader_train
    iterator = iter(loader)
    rows = []
    try:
        for _ in range(2):
            lr, hr, _ = next(iterator)
            rows.append([hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest() for x in (lr,hr)])
    finally:
        # Explicitly release workers before running next independent group.
        del iterator
        del loader
    output = Path(args.experiment_root)/'data_stream'/f'{args.save}.json'
    output.resolve().relative_to((ROOT/'experiment/all_runs').resolve())
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(rows),encoding='utf-8')
    print('DATA_STREAM '+json.dumps(rows),flush=True)


if __name__ == '__main__':
    main()

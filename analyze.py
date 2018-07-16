from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn import linear_model
from collections import namedtuple


def main(args):
    out_dir = args.out

    log = pd.read_pickle(out_dir / 'log.pkl')
    with (out_dir / 'proc.pkl').open('rb') as fp:
        all_proc = pickle.load(fp)

    # relative change in slab size
    diff = np.diff(np.array(log['size']))
    positive_growth = diff > 0

    model = linear_model.LinearRegression()

    Result = namedtuple("Result", ['proc', 'r2'])
    results = []

    for pid, proc in all_proc.items():
        running = np.array(log.pids.apply(lambda pids: pid in pids))
        running = running[1:]
        running = running.reshape(-1, 1)

        model.fit(running, positive_growth)
        r2 = model.score(running, positive_growth)

        results.append(Result(proc=proc, r2=r2))

    results = sorted(results, key=lambda res: res.r2)

    for result in results:
        cmdline = " ".join(result.proc['cmdline'])
        print(f"{result.r2:.03f} - {result.proc['name']} - {cmdline}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    main(args)

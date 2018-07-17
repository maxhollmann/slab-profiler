from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn import linear_model
from collections import namedtuple
import matplotlib.pyplot as plt
import re


def main(args):
    recording_dir = args.recording
    out_dir = recording_dir / 'analysis'
    out_dir.mkdir(exist_ok=True)

    log = pd.read_pickle(recording_dir / 'log.pkl')
    with (recording_dir / 'proc.pkl').open('rb') as fp:
        all_proc = pickle.load(fp)

    # relative change in slab size
    diff = np.diff(np.array(log['size']))
    diff = diff / 1024**2 # convert to MB
    positive_growth = diff > 0

    model = linear_model.LinearRegression(fit_intercept=True)

    Result = namedtuple("Result", ['proc', 'r2', 'coef'])
    results = []

    invalid_for_filename = re.compile(r"[^\w.]")

    for pid, proc in all_proc.items():
        running = np.array(log.pids.apply(lambda pids: pid in pids))
        running = running[1:]
        if not 0.2 < np.mean(running) < 0.8:
            continue

        running = running[:, np.newaxis]

        model.fit(running, diff)
        r2 = model.score(running, diff)
        coef = model.coef_[0]

        if coef < np.max(diff) / 3:
            continue

        results.append(Result(proc=proc, r2=r2, coef=coef))

        # plot
        fig, ax_size = plt.subplots()
        ax_size.plot(diff)
        ax_running = ax_size.twinx()
        ax_running.plot(running, color='red')

        plot_name = f"{r2:.03f}_{coef:5.03f}_{pid}_{proc['name']}.png"
        plot_name = invalid_for_filename.sub('-', plot_name)
        fig.savefig(out_dir / plot_name)
        plt.close()


    results = sorted(results, key=lambda res: res.r2)

    for result in results:
        cmdline = " ".join(result.proc['cmdline'])
        print(f"{result.r2} - {result.coef:.03f} - {result.proc['name']} - {cmdline}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--recording', required=True, type=Path)
    args = parser.parse_args()
    main(args)

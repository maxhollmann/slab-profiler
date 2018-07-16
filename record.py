from pathlib import Path
import re
import pandas as pd
import numpy as np
import time
import psutil
import pickle


def slabinfo():
    path = Path("/proc/slabinfo")
    with path.open('r') as fp:
        content = fp.read()
    lines = content.split('\n')

    spaces = re.compile(r"\s+")
    lines = [spaces.split(line) for line in lines]

    del lines[0] # remove header line
    lines = [line for line in lines if line != ['']]

    names = lines.pop(0)
    del names[0] # remove leading '#'

    # remove '<' and '>' around names
    not_alphanum = re.compile(r"\W")
    names = [not_alphanum.sub('', name) for name in names]
    names = np.array(names)

    columns = np.array(lines).T

    columns = columns[names != '']
    names = names[names != '']

    data = pd.DataFrame(dict(zip(names, columns)))
    return data

def slab_size(name):
    slab = slabinfo()
    row = slab.loc[slab.name == name]
    num_objs = int(row.num_objs.values[0])
    objsize = int(row.objsize.values[0])
    return num_objs * objsize

def running_processes():
    processes = {}
    for proc in psutil.process_iter():
        try:
            pinfo = proc.as_dict(['name', 'username', 'pid', 'cmdline'])
        except psutil.NoSuchProcess:
            pass
        else:
            processes[pinfo['pid']] = pinfo
    return processes


def main(args):
    slab_name = args.slab
    out_dir = args.out
    out_dir.mkdir(exist_ok=True, parents=True)

    log = pd.DataFrame({'time': [], 'size': [], 'pids': []})
    all_proc = running_processes()

    start_t = time.time()

    while True:
        t = time.time()
        size = slab_size(slab_name)

        proc = running_processes()
        pids = np.array(list(proc.keys()))

        log = log.append(
            {'time': t, 'size': size, 'pids': pids},
            ignore_index=True)
        log.to_pickle(out_dir / 'log.pkl')

        all_proc = {**all_proc, **proc}
        with (out_dir / 'proc.pkl').open('wb') as fp:
            pickle.dump(all_proc, fp)

        print(f"Size of {slab_name} after {int((t - start_t) / 60)} min: {size / (1024**2):.02f} MB", end="\r")
        time.sleep(60)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--slab', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    main(args)

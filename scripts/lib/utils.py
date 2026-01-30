import os
import subprocess
import pandas as pd
import numpy as np

CSV_PATH = "../data/runtime_data.csv"


def collect_runtime_data(filepath):
    Ns = 2 ** np.arange(7, 12)
    times = []

    for N in Ns:
        result = subprocess.run(
            [filepath, str(N)], capture_output=True, text=True, check=True
        )
        output = result.stdout.strip()
        time_str = output.split(":")[-1].strip().split(" ")[0]
        print(f"N={N}, Time={time_str} seconds")
        times.append(float(time_str))

    return Ns, np.array(times)


def upsert_runtime_data(implementation, Ns, times):
    row_dict = {"Implementation": implementation, **dict(zip(Ns.astype(str), times))}
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        df = df.set_index("Implementation")
        df.loc[row_dict["Implementation"]] = row_dict
        df.reset_index("Implementation", inplace=True)
    else:
        df = pd.DataFrame([row_dict])
    df.to_csv(CSV_PATH, index=False)

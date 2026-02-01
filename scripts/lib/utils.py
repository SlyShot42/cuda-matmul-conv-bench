import os
import subprocess
import pandas as pd
import numpy as np
from .preprocess import png_to_pgm

CSV_PATH = "../data/runtime_data.csv"
CONV_CSV_PATH = "../data/conv_runtime_data.csv"


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


def collect_conv_runtime_data(exe_path, img_path):
    Ms = 2 ** np.arange(9, 12)
    edge_times = []
    sharpen_times = []
    sharpen7_times = []

    for M in Ms:
        png_to_pgm(img_path, "../data/processed_input/temp_input.pgm", M)
        result = subprocess.run(
            [exe_path, "../data/processed_input/temp_input.pgm", "../data/raw_output/"],
            capture_output=True,
            text=True,
            check=True,
        )
        output = result.stdout.strip()
        outputs = output.splitlines()
        edge_time_str = outputs[0].split(":")[-1].strip().split(" ")[0]
        sharpen_time = outputs[1].split(":")[-1].strip().split(" ")[0]
        sharpen7_time = outputs[2].split(":")[-1].strip().split(" ")[0]
        print(f"M={M}, Edge Time={edge_time_str} seconds")
        print(f"M={M}, Sharpen Time={sharpen_time} seconds")
        print(f"M={M}, Sharpen7 Time={sharpen7_time} seconds")
        edge_times.append(float(edge_time_str))
        sharpen_times.append(float(sharpen_time))
        sharpen7_times.append(float(sharpen7_time))

    return Ms, np.array(edge_times), np.array(sharpen_times), np.array(sharpen7_times)


def upsert_conv_runtime_data(
    implementation,
    image,
    Ms,
    edge_times,
    sharpen_times,
    sharpen7_times,
):
    rows = []

    for kernel, times in [
        ("edge", edge_times),
        ("sharpen", sharpen_times),
        ("sharpen7", sharpen7_times),
    ]:
        row = {
            "Implementation": implementation,
            "Image": image,
            "Kernel": kernel,
        }
        for M, t in zip(Ms, times):
            row[str(M)] = t
        rows.append(row)

    new_df = pd.DataFrame(rows)

    if os.path.exists(CONV_CSV_PATH):
        df = pd.read_csv(CONV_CSV_PATH)
        df = df[
            ~(
                (df["Implementation"] == implementation)
                & (df["Image"] == image)
                & (df["Kernel"].isin(["edge", "sharpen", "sharpen7"]))
            )
        ]
        df = pd.concat([df, new_df], ignore_index=True)
    else:
        df = new_df

    df.to_csv(CONV_CSV_PATH, index=False)

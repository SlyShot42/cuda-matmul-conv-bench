import ctypes
import time
import numpy as np
from PIL import Image
from scripts.lib.preprocess import png_to_pgm
from scripts.lib.utils import upsert_conv_runtime_data

CONV_CSV_PATH = "../data/conv_runtime_data.csv"

lib = ctypes.cdll.LoadLibrary("../src/bin/libconv.so")

lib.gpu_convolution.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_int,
]

edge_detection = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.int32)


sharpen5 = np.array([
    [ 0,  0, -1,  0,  0],
    [ 0, -1, -2, -1,  0],
    [-1, -2, 25, -2, -1],
    [ 0, -1, -2, -1,  0],
    [ 0,  0, -1,  0,  0]
], dtype=np.int32)


sharpen7 = np.array([
    [ 0,  0,  0, -1,  0,  0,  0],
    [ 0,  0, -1, -2, -1,  0,  0],
    [ 0, -1, -2, -3, -2, -1,  0],
    [-1, -2, -3, 49, -3, -2, -1],
    [ 0, -1, -2, -3, -2, -1,  0],
    [ 0,  0, -1, -2, -1,  0,  0],
    [ 0,  0,  0, -1,  0,  0,  0]
], dtype=np.int32)

def run_conv_cpu(A, k, M, N):
    B = np.zeros((M, M), dtype=np.int32)
    
    start = time.time()
    lib.gpu_convolution(A.ravel(), k.ravel(), B.ravel(), M, N)
    end = time.time()
    
    return end - start

def collect_conv_runtime_data(img_path):
    Ms = 2 ** np.arange(9, 12)
    edge_times = []
    sharpen_times = []
    sharpen7_times = []

    for M in Ms:
        png_to_pgm(img_path, "../data/processed_input/temp_input.pgm", M)
        img = Image.open("../data/processed_input/temp_input.pgm")
        A = np.array(img, dtype=np.uint32)
        edge_time = run_conv_cpu(A, edge_detection, M, 3)
        sharpen_time = run_conv_cpu(A, sharpen5, M, 5)
        sharpen7_time = run_conv_cpu(A, sharpen7, M, 7)
        print(f"M={M}, Edge Time={edge_time} seconds")
        print(f"M={M}, Sharpen Time={sharpen_time} seconds")
        print(f"M={M}, Sharpen7 Time={sharpen7_time} seconds")
        edge_times.append(float(edge_time))
        sharpen_times.append(float(sharpen_time))
        sharpen7_times.append(float(sharpen7_time))

    return Ms, np.array(edge_times), np.array(sharpen_times), np.array(sharpen7_times)

Ms, edge_times, sharpen_times, sharpen7_times = collect_conv_runtime_data(
    "../data/raw_input/boat.png"
)
upsert_conv_runtime_data(
    "GPU (python CUDA)",
    "boat.png",
    Ms,
    edge_times,
    sharpen_times,
    sharpen7_times,
)

Ms, edge_times, sharpen_times, sharpen7_times = collect_conv_runtime_data(
    "../data/raw_input/bumblebee.png"
)
upsert_conv_runtime_data(
    "GPU (python CUDA)",
    "bumblebee.png",
    Ms,
    edge_times,
    sharpen_times,
    sharpen7_times,
)

Ms, edge_times, sharpen_times, sharpen7_times = collect_conv_runtime_data(
    "../data/raw_input/hut.png"
)
upsert_conv_runtime_data(
    "GPU (python CUDA)",
    "hut.png",
    Ms,
    edge_times,
    sharpen_times,
    sharpen7_times,
)

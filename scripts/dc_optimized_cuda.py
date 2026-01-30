from lib.utils import upsert_runtime_data
from lib.utils import collect_runtime_data


Ns, times = collect_runtime_data("../src/matrix_gpu_optimized")
upsert_runtime_data("Optimized CUDA", Ns, times)

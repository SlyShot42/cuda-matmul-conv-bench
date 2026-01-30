from scripts.utils import upsert_runtime_data
from scripts.utils import collect_runtime_data


Ns, times = collect_runtime_data("../src/matrix_gpu")
upsert_runtime_data("Naive CUDA", Ns, times)

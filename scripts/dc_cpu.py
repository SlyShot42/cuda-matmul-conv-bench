from lib.utils import upsert_runtime_data
from lib.utils import collect_runtime_data


Ns, times = collect_runtime_data("../src/bin/matrix_cpu")
upsert_runtime_data("CPU (C)", Ns, times)

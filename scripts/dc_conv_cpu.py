from lib.utils import collect_conv_runtime_data
from lib.utils import upsert_conv_runtime_data

Ms, edge_times, sharpen_times, sharpen7_times = collect_conv_runtime_data(
    "../src/bin/conv_cpu", "../data/raw_input/boat.png"
)
upsert_conv_runtime_data(
    "CPU (C)",
    "boat.png",
    Ms,
    edge_times,
    sharpen_times,
    sharpen7_times,
)

Ms, edge_times, sharpen_times, sharpen7_times = collect_conv_runtime_data(
    "../src/bin/conv_cpu", "../data/raw_input/bumblebee.png"
)
upsert_conv_runtime_data(
    "CPU (C)",
    "bumblebee.png",
    Ms,
    edge_times,
    sharpen_times,
    sharpen7_times,
)

Ms, edge_times, sharpen_times, sharpen7_times = collect_conv_runtime_data(
    "../src/bin/conv_cpu", "../data/raw_input/hut.png"
)
upsert_conv_runtime_data(
    "CPU (C)",
    "hut.png",
    Ms,
    edge_times,
    sharpen_times,
    sharpen7_times,
)

import numpy as np
from PIL import Image


def bin_to_png_minmax(bin_path, out_png_path, M, dtype=np.int32):
    B = np.fromfile(bin_path, dtype=dtype, count=M * M).reshape((M, M))

    bmin = B.min()
    bmax = B.max()

    img_u8 = ((B - bmin) * 255.0 / (bmax - bmin)).astype(np.uint8)

    Image.fromarray(img_u8, mode="L").save(out_png_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python3 postprocess.py <input.bin> <output.png> <M>")
        sys.exit(1)

    input_bin = sys.argv[1]
    output_png = sys.argv[2]
    M = int(sys.argv[3])

    bin_to_png_minmax(input_bin, output_png, M)

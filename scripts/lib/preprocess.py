from PIL import Image
import sys


def png_to_pgm(input_png, output_pgm, M):
    img = Image.open(input_png).convert("L")
    img = img.resize((M, M), Image.BILINEAR)
    img.save(output_pgm)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 preprocess.py <input.png> <output.pgm> <M>")
        sys.exit(1)

    input_png = sys.argv[1]
    output_pgm = sys.argv[2]
    M = int(sys.argv[3])

    png_to_pgm(input_png, output_pgm, M)

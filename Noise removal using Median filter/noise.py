# ============================================================
#  CHANGE YOUR IMAGE PATH HERE
# ============================================================
INPUT_IMAGE  = "Screenshot-1110.png"  # <-- your input image
OUTPUT_IMAGE = "Medianfilter_output.png"  # <-- output result
# ============================================================

from PIL import Image


def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


def median_filter(input_path, output_path):
    # ── Load image ─────────────────────────────────────────
    img = Image.open(input_path).convert("L")  # convert to grayscale
    numcols, numrows = img.size
    print(f"Image loaded: {numcols} columns x {numrows} rows")

    pixels = list(img.getdata())

    # ── Load pixels into padded 2D array (zero borders) ───
    array = [[0] * (numcols + 2) for _ in range(numrows + 2)]
    idx = 0
    for row in range(1, numrows + 1):
        for col in range(1, numcols + 1):
            array[row][col] = pixels[idx]
            idx += 1

    # ── Apply 3x3 median filter ────────────────────────────
    result = [[0] * (numcols + 2) for _ in range(numrows + 2)]
    for row in range(1, numrows + 1):
        for col in range(1, numcols + 1):
            window = [
                array[row-1][col-1], array[row-1][col], array[row-1][col+1],
                array[row  ][col-1], array[row  ][col], array[row  ][col+1],
                array[row+1][col-1], array[row+1][col], array[row+1][col+1],
            ]
            insertion_sort(window)
            result[row][col] = window[4]  # median value

    # ── Save output image ──────────────────────────────────
    out_img = Image.new("L", (numcols, numrows))
    out_pixels = []
    for row in range(1, numrows + 1):
        for col in range(1, numcols + 1):
            out_pixels.append(result[row][col])

    out_img.putdata(out_pixels)
    out_img.save(output_path)
    print(f"Done! Output saved to: {output_path}")


if __name__ == "__main__":
    median_filter(INPUT_IMAGE, OUTPUT_IMAGE)
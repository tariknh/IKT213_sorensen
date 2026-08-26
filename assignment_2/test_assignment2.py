"""Tests for the assignment 2 image-operation functions.

Run with:  python3 -m pytest assignment_2/ -v

`test.py` executes a script body at module level (loading the image, calling
every function, writing files). Importing it normally would run all of that,
so instead we parse the source and execute *only* the imports and function
definitions. That keeps these tests about the functions themselves and leaves
test.py untouched.
"""

import ast
import os
import types

import cv2
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE_PATH = os.path.join(HERE, "test.py")
IMAGE_CANDIDATES = ("iris.png", "iris-1.jpg", "iris.jpg")


def _load_functions():
    """Exec only the imports and defs from test.py, skipping its script body."""
    with open(MODULE_PATH) as f:
        source = f.read()

    tree = ast.parse(source, MODULE_PATH)
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))
    ]

    module = types.ModuleType("assignment2_under_test")
    module.__file__ = MODULE_PATH
    exec(compile(tree, MODULE_PATH, "exec"), module.__dict__)
    return module


solution = _load_functions()


def func(name):
    """Fetch a required function by the exact name the assignment specifies."""
    fn = getattr(solution, name, None)
    if fn is None:
        defined = sorted(
            k for k, v in vars(solution).items() if isinstance(v, types.FunctionType)
        )
        pytest.fail(
            f"Assignment requires a function named '{name}'. "
            f"Functions defined in test.py: {defined}"
        )
    return fn


# --- Readable image comparisons ----------------------------------------------
#
# pytest's default diff on a 600x800x3 array prints thousands of numbers and
# tells you nothing. These helpers reduce a mismatch to: what shape came back,
# how many pixels differ, how far off they are, and the first few offenders.


def _summarise(actual, expected, label):
    a = np.asarray(actual)
    e = np.asarray(expected)
    lines = []

    if a.shape != e.shape:
        lines.append(f"  shape:            {a.shape}")
        lines.append(f"  expected shape:   {e.shape}   <-- MISMATCH")
        return lines

    lines.append(f"  shape:            {a.shape}  (matches {label})")

    differing = a != e
    per_pixel = differing.any(axis=2) if differing.ndim == 3 else differing
    n = int(per_pixel.sum())
    total = per_pixel.size
    max_delta = int(np.abs(a.astype(np.int32) - e.astype(np.int32)).max())

    lines.append(f"  pixels differing: {n:,} of {total:,}  ({100 * n / total:.1f}%)")
    lines.append(f"  largest delta:    {max_delta}")

    ys, xs = np.nonzero(per_pixel)
    for i in range(min(3, len(ys))):
        y, x = int(ys[i]), int(xs[i])
        lines.append(
            f"    row {y:>4}, col {x:>4}:  got {np.atleast_1d(a[y, x]).tolist()}"
            f"   expected {np.atleast_1d(e[y, x]).tolist()}"
        )
    if len(ys) > 3:
        lines.append(f"    ... and {len(ys) - 3:,} more")
    return lines


def assert_image_equals(actual, expected, what, hint=None):
    """Compare against a single expected image, reporting a compact diff."""
    if actual is None:
        pytest.fail(f"{what}\n  got: None (the function returned nothing)", pytrace=False)

    a = np.asarray(actual)
    e = np.asarray(expected)
    if a.shape == e.shape and np.array_equal(a, e):
        return

    report = [what] + _summarise(a, e, "expected")
    if hint:
        report += ["", f"  hint: {hint}"]
    pytest.fail("\n".join(report), pytrace=False)


def assert_image_matches_one_of(actual, candidates, what, hint=None):
    """Compare against several acceptable answers, e.g. wrap vs clip.

    `candidates` maps a human-readable name to the expected array. On failure
    the report is written against whichever candidate came closest, so you can
    see which interpretation your code was nearest to.
    """
    if actual is None:
        pytest.fail(f"{what}\n  got: None (the function returned nothing)", pytrace=False)

    a = np.asarray(actual)
    for expected in candidates.values():
        e = np.asarray(expected)
        if a.shape == e.shape and np.array_equal(a, e):
            return

    def distance(expected):
        e = np.asarray(expected)
        if a.shape != e.shape:
            return float("inf")
        return float(np.abs(a.astype(np.int32) - e.astype(np.int32)).mean())

    name, closest = min(candidates.items(), key=lambda kv: distance(kv[1]))

    report = [what, f"  accepted answers: {', '.join(candidates)}", f"  closest was '{name}':"]
    report += _summarise(a, closest, f"'{name}'")
    if hint:
        report += ["", f"  hint: {hint}"]
    pytest.fail("\n".join(report), pytrace=False)


@pytest.fixture(scope="module")
def image():
    for name in IMAGE_CANDIDATES:
        path = os.path.join(HERE, name)
        if os.path.exists(path):
            img = cv2.imread(path)
            assert img is not None, f"cv2.imread returned None for {path}"
            return img
    pytest.fail(f"No iris image found in {HERE}; looked for {IMAGE_CANDIDATES}")


# --- Loading the image -------------------------------------------------------


def test_iris_image_loads_as_a_colour_image(image):
    assert image.ndim == 3
    assert image.shape[2] == 3
    assert image.dtype == np.uint8


# --- Padding -----------------------------------------------------------------


def test_padding_adds_border_width_on_all_four_sides(image):
    h, w = image.shape[:2]
    out = func("padding")(image, 100)
    assert out.shape[:2] == (h + 200, w + 200)
    assert out.shape[2] == 3


def test_padding_border_reflects_the_edges(image):
    """The border must mirror the image, not be a constant fill.

    Accepts either cv2.BORDER_REFLECT (edge row duplicated) or
    cv2.BORDER_REFLECT_101 (edge row not duplicated).
    """
    bw = 100
    out = func("padding")(image, bw)
    row_above_top_edge = out[bw - 1, bw : bw + image.shape[1]]

    assert_image_matches_one_of(
        row_above_top_edge,
        {
            "BORDER_REFLECT (edge row duplicated)": image[0],
            "BORDER_REFLECT_101 (edge row not duplicated)": image[1],
        },
        "padding(image, 100): the row just above the top edge is not a reflection",
        hint="you are using cv2.BORDER_CONSTANT, which fills the border with a "
        "flat colour. Pass cv2.BORDER_REFLECT instead.",
    )


def test_padding_does_not_modify_the_original(image):
    before = image.copy()
    func("padding")(image, 100)
    assert np.array_equal(image, before)


# --- Cropping ----------------------------------------------------------------


def test_crop_returns_the_requested_region(image):
    h, w = image.shape[:2]
    x_0, x_1 = 200, w - 130
    y_0, y_1 = 200, h - 130

    out = func("crop")(image, x_0, x_1, y_0, y_1)

    assert out.shape[:2] == (h - 330, w - 330)
    assert_image_equals(
        out,
        image[y_0:y_1, x_0:x_1],
        f"crop(image, {x_0}, {x_1}, {y_0}, {y_1}) returned the wrong region",
    )


def test_crop_argument_order_is_x_first_then_y(image):
    """Guards against swapping the x and y pairs: a deliberately non-square
    region must come back with the height/width the arguments imply."""
    out = func("crop")(image, 10, 110, 20, 320)
    assert out.shape[:2] == (300, 100)


# --- Resize ------------------------------------------------------------------


def test_resize_to_200x200(image):
    out = func("resize")(image, 200, 200)
    assert out.shape[:2] == (200, 200)
    assert out.shape[2] == 3


def test_resize_treats_first_argument_as_width(image):
    out = func("resize")(image, 320, 150)
    assert out.shape[:2] == (150, 320), "resize(image, width, height) -> shape (height, width, 3)"


# --- Manual copy -------------------------------------------------------------


def test_copy_fills_the_empty_array_with_the_image_pixels(image):
    height, width = image.shape[:2]
    empty = np.zeros((height, width, 3), dtype=np.uint8)

    out = func("copy")(image, empty)

    assert out is not None, "copy() must return the filled array"
    assert np.array_equal(out, image)


def test_copy_writes_into_the_array_it_was_given(image):
    height, width = image.shape[:2]
    empty = np.zeros((height, width, 3), dtype=np.uint8)

    func("copy")(image, empty)

    assert np.array_equal(empty, image), (
        "The pixels should be written into emptyPictureArray, not into a new array"
    )


def test_copy_is_independent_of_the_source(image):
    height, width = image.shape[:2]
    empty = np.zeros((height, width, 3), dtype=np.uint8)

    out = func("copy")(image, empty)
    out[0, 0] = [0, 0, 0] if image[0, 0].any() else [255, 255, 255]

    assert not np.array_equal(out[0, 0], image[0, 0]), (
        "copy() returned a view of the original; changing it changed the source"
    )


# --- Grayscale ---------------------------------------------------------------


def test_grayscale_returns_a_single_channel_image(image):
    out = func("grayscale")(image)
    assert out.ndim == 2, "A grayscale image has one channel, so shape is (height, width)"
    assert out.shape == image.shape[:2]
    assert out.dtype == np.uint8


def test_grayscale_matches_opencv_conversion(image):
    assert_image_equals(
        func("grayscale")(image),
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        "grayscale(image) does not match cv2.COLOR_BGR2GRAY",
    )


# --- HSV ---------------------------------------------------------------------


def test_hsv_returns_a_three_channel_image_of_the_same_size(image):
    out = func("hsv")(image)
    assert out.shape == image.shape
    assert out.dtype == np.uint8


def test_hsv_matches_opencv_conversion(image):
    assert_image_equals(
        func("hsv")(image),
        cv2.cvtColor(image, cv2.COLOR_BGR2HSV),
        "hsv(image) does not match cv2.COLOR_BGR2HSV",
    )


def test_hsv_is_not_the_untouched_bgr_image(image):
    assert not np.array_equal(func("hsv")(image), image)


# --- Hue shift ---------------------------------------------------------------


def test_hue_shifted_takes_image_empty_array_and_hue(image):
    height, width = image.shape[:2]
    empty = np.zeros((height, width, 3), dtype=np.uint8)

    out = func("hue_shifted")(image, empty, 50)

    assert out is not None, "hue_shifted() must return the shifted image"
    assert out.shape == image.shape


def test_hue_shifted_by_50_wraps_or_clips_every_colour_value(image):
    """Values above 255 must be handled deliberately: either wrap around
    (uint8 overflow) or saturate at 255. Both are accepted, silently
    producing something else is not."""
    height, width = image.shape[:2]
    empty = np.zeros((height, width, 3), dtype=np.uint8)

    out = func("hue_shifted")(image, empty, 50)

    assert_image_matches_one_of(
        out,
        {
            "wraparound (v + 50) % 256": ((image.astype(np.int16) + 50) % 256).astype(np.uint8),
            "saturation min(v + 50, 255)": np.clip(
                image.astype(np.int16) + 50, 0, 255
            ).astype(np.uint8),
        },
        "hue_shifted(image, emptyPictureArray, 50) did not shift every colour value by 50",
        hint="the assignment asks you to shift the BGR colour values themselves, "
        "not the hue channel of an HSV conversion.",
    )


def test_hue_shifted_by_zero_is_unchanged(image):
    height, width = image.shape[:2]
    empty = np.zeros((height, width, 3), dtype=np.uint8)

    assert_image_equals(
        func("hue_shifted")(image, empty, 0),
        image,
        "hue_shifted(image, emptyPictureArray, 0) should return the image unchanged",
    )


def test_hue_shifted_handles_a_negative_shift(image):
    height, width = image.shape[:2]
    empty = np.zeros((height, width, 3), dtype=np.uint8)

    assert_image_matches_one_of(
        func("hue_shifted")(image, empty, -50),
        {
            "wraparound (v - 50) % 256": ((image.astype(np.int16) - 50) % 256).astype(np.uint8),
            "clamp max(v - 50, 0)": np.clip(image.astype(np.int16) - 50, 0, 255).astype(np.uint8),
        },
        "hue_shifted(image, emptyPictureArray, -50) did not handle values dropping below 0",
    )


# --- Smoothing ---------------------------------------------------------------


def test_smoothing_keeps_the_image_size(image):
    out = func("smoothing")(image)
    assert out.shape == image.shape
    assert out.dtype == np.uint8


def test_smoothing_uses_a_15x15_gaussian_with_the_default_border(image):
    # sigmaX=0 lets OpenCV derive sigma from the kernel size; borderType is
    # left unset so it uses BORDER_DEFAULT.
    assert_image_equals(
        func("smoothing")(image),
        cv2.GaussianBlur(image, (15, 15), 0),
        "smoothing(image) does not match cv2.GaussianBlur(image, (15, 15), 0)",
    )


def test_smoothing_actually_blurs(image):
    """A blurred image has less high-frequency detail than the original."""
    out = func("smoothing")(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

    assert cv2.Laplacian(blurred_gray, cv2.CV_64F).var() < cv2.Laplacian(gray, cv2.CV_64F).var()


# --- Rotation ----------------------------------------------------------------


def test_rotation_90_is_clockwise(image):
    assert_image_equals(
        func("rotation")(image, 90),
        cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
        "rotation(image, 90) is not a clockwise 90 degree rotation",
        hint="rotating by 90 swaps width and height, so the result must not be "
        "cropped back into the original frame. Also note a positive angle in "
        "cv2.getRotationMatrix2D rotates COUNTER-clockwise. cv2.rotate("
        "image, cv2.ROTATE_90_CLOCKWISE) does both correctly.",
    )


def test_rotation_180(image):
    assert_image_equals(
        func("rotation")(image, 180),
        cv2.rotate(image, cv2.ROTATE_180),
        "rotation(image, 180) is not a 180 degree rotation",
        hint="rotating about (w // 2, h // 2) is half a pixel off on even "
        "dimensions, which leaves a black first row and column. "
        "cv2.rotate(image, cv2.ROTATE_180) is exact.",
    )


def test_rotation_does_not_modify_the_original(image):
    before = image.copy()
    func("rotation")(image, 90)
    assert np.array_equal(image, before)

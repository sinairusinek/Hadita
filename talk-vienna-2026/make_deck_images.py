"""Downscale the figure set for embedding in the .pptx.

The site's figures are at print resolution; a slide panel is ~6.5in wide, so
~1800px on the long edge is already beyond what any projector resolves. This
keeps the deck small enough to upload to Drive in one piece, with no visible
loss on screen. The site keeps the full-resolution originals.
"""
from pathlib import Path

from PIL import Image

HERE = Path(__file__).parent
SRC = HERE / "img"
DST = HERE / "img" / "deck"
LONG_EDGE = 1800


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    total = 0
    for p in sorted(list(SRC.glob("*.jpg")) + list(SRC.glob("*.png"))):
        im = Image.open(p)
        w, h = im.size
        scale = min(1.0, LONG_EDGE / max(w, h))
        if scale < 1.0:
            im = im.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
        out = DST / f"{p.stem}.jpg"
        # Flatten transparency onto the deck's paper colour before JPEG.
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (244, 241, 234))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        im.convert("RGB").save(out, quality=86, optimize=True, progressive=True)
        total += out.stat().st_size
        print(f"  {out.name:38} {im.size[0]}x{im.size[1]}  {out.stat().st_size/1e6:.2f}MB")
    print(f"total {total/1e6:.1f} MB")


if __name__ == "__main__":
    main()

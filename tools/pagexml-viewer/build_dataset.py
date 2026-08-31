#!/usr/bin/env python3
"""Wire a flat PAGE-XML corpus into the minimal PageXML viewer.

Builds <dataset>/ with a mets.json manifest and symlinks back to the corpus,
so no image data is duplicated. Run from anywhere:

    python3 tools/pagexml-viewer/build_dataset.py \
        --src "Transkribus upload/final3" --name Hadita_final3

Then serve the viewer directory and open it:

    python3 -m http.server 8777 --directory tools/pagexml-viewer
"""
import argparse, glob, json, os, re, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_EXTS = (".jpeg", ".jpg", ".png", ".tif", ".tiff")


def page_no(path):
    m = re.search(r'_(\d+)\.', os.path.basename(path))
    return int(m.group(1)) if m else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="corpus dir with *.xml + images")
    ap.add_argument("--name", required=True, help="dataset folder name")
    ap.add_argument("--title", default=None)
    ap.add_argument("--description", default="")
    ap.add_argument("--copy", action="store_true",
                    help="copy files instead of symlinking (for deploying)")
    a = ap.parse_args()

    src = os.path.abspath(a.src)
    if not os.path.isdir(src):
        sys.exit(f"no such corpus dir: {src}")

    dst = os.path.join(HERE, a.name)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(os.path.join(dst, "page"))

    def place(s, d):
        if a.copy:
            shutil.copy2(s, d)
        else:
            os.symlink(os.path.relpath(s, os.path.dirname(d)), d)

    entries, missing = [], []
    for x in sorted(glob.glob(os.path.join(src, "*.xml")), key=page_no):
        base = os.path.splitext(os.path.basename(x))[0]
        img = next((base + e for e in IMG_EXTS
                    if os.path.exists(os.path.join(src, base + e))), None)
        if not img:
            missing.append(base)
            continue
        place(x, os.path.join(dst, "page", base + ".xml"))
        place(os.path.join(src, img), os.path.join(dst, img))
        entries.append({"image": img, "pagexml": f"page/{base}.xml"})

    with open(os.path.join(dst, "mets.json"), "w") as fh:
        json.dump(entries, fh, indent=1)

    # Merge into manuscripts.json rather than clobbering other datasets.
    man_path = os.path.join(HERE, "manuscripts.json")
    try:
        with open(man_path) as fh:
            manuscripts = json.load(fh)
    except Exception:
        manuscripts = []
    manuscripts = [m for m in manuscripts if m.get("manuscriptFolder") != a.name]
    manuscripts.append({
        "manuscriptFolder": a.name,
        "title": a.title or a.name,
        "sourceUrl": "https://github.com/sinairusinek/Hadita",
        "description": a.description,
    })
    with open(man_path, "w") as fh:
        json.dump(manuscripts, fh, indent=1)

    print(f"{a.name}: {len(entries)} pages"
          + (f", {len(missing)} xml without image: {missing[:5]}" if missing else ""))


if __name__ == "__main__":
    main()

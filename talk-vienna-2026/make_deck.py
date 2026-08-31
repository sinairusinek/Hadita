"""Build the editable .pptx deck for the Vienna 2026 talk.

Import the output into Google Slides (File > Import slides), edit and comment
there, then export back as .pptx so the edits can be read.

Every slide carries: a headline, a figure where there is one, a few key points,
and the fuller web-page prose in the speaker-notes field, ready to trim or speak
from. Layout is deliberately plain — boxes, not templates — so that editing in
Google Slides does not fight a design.

    ../.venv/bin/python make_deck.py
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt
from PIL import Image

HERE = Path(__file__).parent
# Deck-sized copies (see make_deck_images.py): the site's originals are print
# resolution and make the .pptx too large to upload in one piece.
IMG = HERE / "img" / "deck"
OUT = HERE / "Vienna2026_Hadita.pptx"

# 16:9, the aspect the venue will project.
W, H = Inches(13.333), Inches(7.5)

INK = RGBColor(0x26, 0x46, 0x53)
MUTED = RGBColor(0x8D, 0x99, 0xAE)
BAD = RGBColor(0xD1, 0x49, 0x5B)
GOOD = RGBColor(0x2A, 0x9D, 0x8F)
PAPER = RGBColor(0xF4, 0xF1, 0xEA)
FONT = "Helvetica Neue"

M = Inches(0.62)          # page margin
TITLE_T = Inches(0.46)


def _txbox(slide, l, t, w, h):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    return tf


def _run(p, text, size, *, bold=False, color=INK, italic=False):
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color
    r.font.name = FONT
    return r


def _bg(slide):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = PAPER


def add_slide(prs, *, kicker=None, title, points=(), image=None, notes="",
              caption=None, image_side="right", pull=None, pull_colour=GOOD):
    """One content slide. `points` are short bullets; `notes` is the prose."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])   # blank
    _bg(slide)

    has_img = image is not None
    text_w = Inches(5.3) if has_img else Inches(11.0)

    # ---- headline block
    tf = _txbox(slide, M, TITLE_T, text_w, Inches(1.5))
    if kicker:
        p = tf.paragraphs[0]
        _run(p, kicker.upper(), 11, bold=True, color=MUTED)
        p.space_after = Pt(6)
        p = tf.add_paragraph()
    else:
        p = tf.paragraphs[0]
    _run(p, title, 30 if has_img else 34, bold=True)
    p.line_spacing = 0.94

    # ---- bullets
    top = TITLE_T + Inches(1.62)
    if points:
        tf = _txbox(slide, M, top, text_w, Inches(3.4))
        for i, pt_ in enumerate(points):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            _run(p, "— ", 15, color=MUTED)
            # **bold** spans inside a bullet
            for j, seg in enumerate(pt_.split("**")):
                if seg:
                    _run(p, seg, 15, bold=(j % 2 == 1))
            p.space_after = Pt(9)
            p.line_spacing = 1.16
        top = top + Inches(0.42) * len(points) + Inches(0.3)

    # ---- pull quote
    if pull:
        tf = _txbox(slide, M, top, text_w, Inches(1.2))
        p = tf.paragraphs[0]
        _run(p, pull, 17, bold=True, color=pull_colour)
        p.line_spacing = 1.15

    # ---- figure
    if has_img:
        path = IMG / image
        iw, ih = Image.open(path).size
        box_l, box_t = Inches(6.25), Inches(0.62)
        box_w, box_h = Inches(6.45), Inches(5.9)
        scale = min(box_w / iw, box_h / ih)
        w_, h_ = int(iw * scale), int(ih * scale)
        left = int(box_l + (box_w - w_) / 2)
        topi = int(box_t + (box_h - h_) / 2)
        slide.shapes.add_picture(str(path), left, topi, w_, h_)
        if caption:
            tf = _txbox(slide, box_l, Emu(topi + h_) + Inches(0.10), box_w, Inches(0.8))
            p = tf.paragraphs[0]
            _run(p, caption, 10, color=MUTED)
            p.line_spacing = 1.2

    # ---- speaker notes
    if notes:
        slide.notes_slide.notes_text_frame.text = notes.strip()
    return slide


def add_title_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(slide)
    # PNG, not the site's webp: python-pptx cannot embed WEBP.
    slide.shapes.add_picture(str(IMG / "banner_trees.jpg"),
                             Inches(3.4), Inches(4.55), Inches(6.5), None)
    tf = _txbox(slide, M, Inches(1.5), Inches(11.4), Inches(3.0))
    p = tf.paragraphs[0]
    _run(p, "OCR/HTR FOR UNDER-RESOURCED LANGUAGES REDUX · VIENNA", 11,
         bold=True, color=MUTED)
    p.space_after = Pt(16)
    p = tf.add_paragraph()
    _run(p, "Winding Workflows: Editors, Artificial Agents\nand Specialist ATR Models",
         40, bold=True)
    p.line_spacing = 1.02
    p.space_after = Pt(16)
    p = tf.add_paragraph()
    _run(p, "Two registers from Mandate Palestine, 1930s–40s. One solved, one not.", 17,
         color=RGBColor(0x4A, 0x5C, 0x66))
    p.space_after = Pt(10)
    p = tf.add_paragraph()
    _run(p, "Sinai Rusinek · Haifa University, Open University DH-Dev · "
            "Austrian Academy of Sciences, 10 September 2026", 13, color=MUTED)
    slide.notes_slide.notes_text_frame.text = (
        "Session 2: Workflows & Systems, 10:50–12:10 (~20 min + Q&A).\n\n"
        "Framing note: the through-line is that no stage of this work belongs to one "
        "kind of reader — the title's three actors each fail at something the others "
        "catch.")
    return slide


def add_section_slide(prs, kicker, title, notes=""):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(slide)
    tf = _txbox(slide, M, Inches(2.9), Inches(11.4), Inches(2.0))
    p = tf.paragraphs[0]
    _run(p, kicker.upper(), 12, bold=True, color=MUTED)
    p.space_after = Pt(12)
    p = tf.add_paragraph()
    _run(p, title, 36, bold=True)
    p.line_spacing = 1.0
    if notes:
        slide.notes_slide.notes_text_frame.text = notes.strip()
    return slide


def build() -> None:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    add_title_slide(prs)

    add_slide(
        prs, kicker="the place", title="Al-Haditha",
        image="fig04_tel_hadid.jpg",
        caption="Tel Hadid today, and the register that records who held which parcel.",
        points=[
            "The village near Tel Hadid — Lydda, Ramla, Jaffa. Ottoman origins, destroyed 1948.",
            "The Hadid Expedition (Alon & Koch, ISF 1316/22) reconstructs it from "
            "archaeology, oral history, Ottoman and Mandate records.",
            "**102 pages · 19 columns · 3,069 rows · 58,122 cells**",
            "Handwritten Arabic under printed English headers. Shelfmark TAX 1-85.",
        ],
        notes="The property-tax register is one of the few surviving documents of who "
              "held which parcel. Entries reference Tax Distribution Lists, 1938–1945.\n\n"
              "This is the 'why it matters' slide — the register is evidence for a "
              "village that no longer exists.")

    add_slide(
        prs, kicker="two phases", title="The register that did work",
        points=[
            "Haifa Government Hospital, 1930–1948. **33 notebooks · 2,548 images · "
            "29,879 admissions.**",
            "Phase 1, Transkribus: 10 notebooks, ~6,000 records, 95 human training "
            "pages, CER 4.30% → 3.51%.",
            "Phase 2, Gemini: 33 notebooks, 29,879 records, **0 training pages**, "
            "structured TSV in one pass.",
            "Not a pure upgrade — the slow phase built the knowledge the fast phase runs on.",
        ],
        notes="CAVEAT TO STATE ALOUD: no head-to-head accuracy was ever computed. "
              "The comparison is scope and architecture, not CER.\n\n"
              "This slide sets up the contrast. The audience should be thinking "
              "'so apply phase 2 to the other register' — and the next slide shows why "
              "that fails.")

    add_slide(
        prs, kicker="slide 6 · why this one is hard", title="Two registers, side by side",
        image="fig06_two_registers.jpg",
        caption="Left — Haifa: every row filled, rules crisp, page flat, serials 99–109 "
                "unbroken (names redacted). Right — Hadita: five filled rows then a "
                "scatter, ruling almost invisible, page bowing into the gutter.",
        points=[
            "Same technology. One yields, one does not.",
            "The difference is not the script — it is the **genre**: a sparse, "
            "numeric, ruled table.",
        ],
        notes="This is the single most important image in the talk. Let it sit.\n\n"
              "Left: nb01_p003. Right: Hadita page 9.\n\n"
              "Point at the bottom third of the right-hand page: that is where every "
              "later problem lives.")

    add_slide(
        prs, kicker="slide 6 · why this one is hard", title="Four failure modes",
        image="fig06b_detail_strip.jpg",
        caption="Missing left border · clerk-drawn lines · ditto marks · nil dashes.",
        points=[
            "**Photographed bound pages** — column boundaries deviate −49px to +57px "
            "down a single page.",
            "**The rules are faint**, and on a sparse page nothing else marks a row.",
            "**The left border line is missing** — paper edge and binding shadow read "
            "as printed rules.",
            "**The clerk drew lines too.** Morphology cannot tell those from printed ruling.",
        ],
        notes="PHYSICAL problems on this slide; content and semantics on the next.\n\n"
              "The narrow money columns on the right are the next worst after the "
              "left border.")

    add_slide(
        prs, kicker="slide 6 · why this one is hard",
        title="Content and semantics",
        points=[
            "**32% of cells carry ink** (18,718 of 58,122).",
            "Cells hold **1–4 glyphs**, mostly Eastern Arabic numerals. No lexicon, "
            "no redundancy: every stroke is load-bearing.",
            "**Two digit systems coexist**, both authentic — ٠-٩ in most columns, "
            "0-9 in New_Serial_No and T.D.L. references. Not normalised.",
            "**Ditto marks and nil dashes carry meaning.** 97 reference cells hold text "
            "with essentially zero ink.",
            "**A row is not always an entry** — sub-totals, carry-forwards, tax-year "
            "breakdowns.",
        ],
        notes="The ditto/nil point matters twice over: once here, and again at the ink "
              "gate, where 'meaning with no ink' is exactly the class the gate costs us.\n\n"
              "On page 9 the serials stop after row 5 while dated entries continue below.")

    add_slide(
        prs, kicker="slide 7 · the cast", title="Who is actually reading?",
        image="fig07_the_cast.jpg",
        caption="The pipeline, coloured by who does each stage rather than by what it computes.",
        points=[
            "Specialist ATR models · VLMs · rule-based code · editors.",
            "The title was too short to name them all — and “agents” was already doing "
            "double duty for two of them.",
        ],
        notes="MAY FOLD INTO SLIDE 8 (your note in SLIDES.md).\n\n"
              "The argument: no stage belongs to one kind of reader. The colouring is "
              "the point — this is the talk's through-line in one picture.\n\n"
              "Two players not yet named at this stage: the community, and the chat "
              "that relayed what the community knew and experimented alongside me "
              "toward the pipeline.")

    add_section_slide(
        prs, "part two", "Getting the geometry right",
        notes="Optional divider. Delete if the talk runs long.")

    add_slide(
        prs, kicker="slide 8 · the lossy convenience",
        title="Dewarping destroyed the pages",
        image="fig08_dewarp_damage.jpg",
        caption="Page 71: 382px — 4.2 row pitches — written and discarded.",
        points=[
            "The pipeline straightened the **image**. The canvas ended at "
            "last_row_center + ½ pitch.",
            "Content below the last **detected** row was **cut, not smeared** — "
            "a missed row was silently discarded.",
            "**98 of 98 pages damaged.** 25 lost a whole written row; 92 had the "
            "bottom-cut geometry.",
        ],
        pull="Fix: warp the coordinates, not the image.",
        notes="Recovered 35 rows. Structural validation 37/98 → 95/98.\n\n"
              "Upload the undamaged page; push the cell polygons through the inverse "
              "remap. PAGE XML polygons can bow — the format could always express this.\n\n"
              "THE LESSON: the lossy step was CONVENIENCE. Flattening the image made "
              "every later stage simpler, and paid for it by destroying data invisibly. "
              "Nothing errored. The corpus just quietly lost rows.")

    add_slide(
        prs, kicker="slide 9 · columns", title="One miss is the whole problem",
        image="fig09b_one_miss.jpg",
        caption="Left: 19 rules found. Right: one rule dropped — every column index to "
                "its right shifts, and the page ends at 17 instead of 18.",
        points=[
            "Detecting 19 rules one by one cannot work: faint, crossed by handwriting, "
            "false verticals from the binding, and the page bows.",
            "Text is placed into cells by column **index** — a slightly misdrawn line is "
            "cosmetic, a dropped one mislabels every cell to its right.",
        ],
        pull_colour=BAD,
        notes="This is the slide that justifies the template approach. The audience has "
              "to feel that near-miss geometry is not 'nearly right' — it is wrong in a "
              "way that silently corrupts the data.")

    add_slide(
        prs, kicker="slide 9 · columns",
        title="Stop detecting nineteen lines; fit one known form",
        image="fig09a_header_band.jpg",
        caption="The header band — the cleanest column anchors on the page, because the "
                "scribe never wrote there.",
        points=[
            "The printed form is constant, so the 19 relative widths are a **template**.",
            "The header band is print-only and carries all 18 interior rules as short "
            "clean strokes.",
            "Fit the template to those, with two free parameters: **right edge and scale**.",
            "**The left edge is derived, never detected** — whatever lies left of the "
            "Serial/Date rule is the Serial column.",
        ],
        pull="“Stop detecting 19 lines; fit ONE known form.”",
        notes="Deriving the left edge is what removes the binding shadow from the "
              "problem entirely.\n\n"
              "RESIDUAL CLASS: pages where the scribe wrote across the printed columns. "
              "No line detector can fix that — it needs content-level flagging and RA "
              "judgment.\n\n"
              "The full-page version of this figure is fig09_column_template.jpg if you "
              "want to show the template carried down the page.")

    add_slide(
        prs, kicker="slide 10 · rows", title="Where the same move failed",
        image="fig10a_morphology.jpg",
        caption="Adaptive threshold + long horizontal opening — the standard "
                "table-extraction recipe. This is all it returns.",
        points=[
            "Columns worked by ignoring the writing and fitting the printed form. "
            "The obvious next step was to do it again for rows.",
            "It did not survive contact with the scribe.",
            "The horizontal rules are printed **1–3% darker than the paper**, and dashed. "
            "The faint dashes do not binarise; what survives is handwriting.",
        ],
        notes="Note the irony worth saying aloud: this is the SAME recipe that works "
              "on the verticals. The method is not wrong in general — it is wrong here.")

    add_slide(
        prs, kicker="slide 10 · rows", title="A median across strips does find them",
        image="fig10b_strip_median.jpg",
        caption="30 individual strips (faint) and their median. On the densest page: "
                "33 of 33 rules, within 6px of a perfectly regular ladder.",
        points=[
            "Cut the page into **30 vertical strips**; take each strip's darkness profile.",
            "Take the **median across strips**. A printed rule crosses every strip and "
            "survives; a written line touches a few and is suppressed.",
        ],
        pull="The signal was always there — it needed a statistic robust to ink, "
             "not a stronger filter.",
        notes="This is the methodological heart of the geometry half of the talk. "
              "Worth dwelling on: the failure was not sensitivity, it was the wrong "
              "estimator.")

    add_slide(
        prs, kicker="slide 10 · rows", title="And then the scribe",
        image="fig10c_rules_not_boundaries.jpg",
        caption="Top: printed rules used as row boundaries — every line cuts through the "
                "writing. Bottom: boundaries midway between.",
        points=[
            "Building the grid on those rules gave 34 rows on every preview page — "
            "the printed form's own row count, right for the first time.",
            "On visual review, every boundary cut straight through a line of handwriting.",
            "Measured: the written lines sit within **1–7px** of the printed rules, "
            "on a form whose rules are **93px** apart.",
        ],
        pull="The scribe writes ON the rule, not between the rules. "
             "A rule is not the edge of a row — it is the middle of one.",
        notes="Reading the ruling as a set of boundaries put every boundary half a row "
              "out of place: geometrically flawless, and useless for reading.")

    add_slide(
        prs, kicker="slide 10 · rows", title="Two sources, two jobs",
        points=[
            "The grid is built on **Kraken baselines** — a generic segmenter, no Arabic, "
            "no notion of a table — which finds where the writing is. Boundaries go "
            "midway between.",
            "The printed rules stay in, demoted to one job: they **measure the spacing**, "
            "which they do better than handwriting can, because a printed form is "
            "regular and a scribe is not.",
            "Where a gap is wide enough for two rows, a row is inserted: "
            "**blank rows are found by arithmetic, not by vision.**",
        ],
        pull="The printed form knows the spacing. Only the ink knows where the rows "
             "actually fall.",
        notes="AND THE MEASUREMENT WAS THE REAL FAILURE. The bad grid was scored by "
              "asking what fraction of the writing fell inside its rows — writing "
              "measured against lines derived from that same writing. It read 0.97 "
              "while sitting half a row off.\n\n"
              "NEVER SCORE GEOMETRY WITH A METRIC DRAWN FROM THE SOURCE IT CAME FROM.\n\n"
              "This is probably the most transferable lesson in the talk. Consider "
              "giving it its own slide.")

    add_slide(
        prs, kicker="slide 11 · ink-gating", title="PyLaia has no “empty” prediction",
        image="fig11_ink_histogram.jpg",
        caption="Measured on the six proxy-GT pages with the shipped gate's own masks. "
                "Cells with text: median 270px. Blank cells: median 0.",
        points=[
            "Our XML placed a TextLine in all 58,122 cells — so on page 11 PyLaia "
            "returned text in **416 of 532 blank cells**.",
            "The fix is not a model: measure ink per cell, emit a TextLine only where "
            "there is some. **Cutoff at 6px.**",
            "**39,404 of 58,122 lines dropped. Inventions: zero.**",
        ],
        notes="NOTE: the cutoff is 6px, not the 12 in the older draft — "
              "gate_textlines.py lowered it because user-labelled cells sat at 9px.\n\n"
              "COST: 16.5% of remaining disagreements are now cells the model left "
              "empty that the reference fills — the near-invisible ones. A pink-pencil "
              "second chance on the green channel recovers most of that class "
              "(the pigment absorbs green, so strokes darken G relative to nearby paper).")

    add_section_slide(
        prs, "part three", "Who can actually read the cells?",
        notes="Optional divider. Delete if the talk runs long.")

    add_slide(
        prs, kicker="slide 12 · who can read the cells?",
        title="Nobody reads them well",
        image="fig12_agreement_funnel.jpg",
        caption="What the pipeline hands the editor. Cross-family agreement auto-accepts "
                "about a third of inked cells at 0.88% substantive error.",
        points=[
            "**hadid02** (PyLaia fine-tuned on this hand, 6 GT pages): 45.8% perfect "
            "cells, ~1,084 RA keystrokes/page.",
            "**Gemini 3**: 56–70% perfect cells, 720 keystrokes/page.",
            "**Baseer/NAKBA** (competition winner, CER 0.079 on Palestinian "
            "manuscripts): **1.8× more** keystrokes.",
        ],
        notes="CAVEATS: hadid02 overfits because there are only 6 pages of GT; "
              "Baseer is a single run on isolated cell crops.\n\n"
              "The agreement result is the constructive finding: a same-model rerun "
              "agrees twice as often and is wrong 17% of the time. Correlated errors "
              "are not evidence. Cross-family disagreement is.")

    add_slide(
        prs, kicker="slide 13 · why the specialists lose",
        title="Genre, not script",
        image="fig13_baseer_days.jpg",
        caption="Baseer's hallucinations in blank cells: السبت · الخميس · الاربعاء — "
                "days of the week, in a tax ledger. Every cell shown holds under 20px of ink.",
        points=[
            "**Baseer**: base model trained on 500k pairs with **zero handwriting**, and "
            "tables with >25% empty cells **explicitly discarded** — our exact "
            "distribution. Fine-tuned on memoir **prose**.",
            "**Every Arabic Kraken model** is trained on prose **lines**. CTC needs "
            "horizontal context that a 2-glyph cell denies it.",
            "Digits are a rounding error in every training set — so ٢/٣ and ٤/٦ are "
            "exactly the classes no model had pressure to separate.",
        ],
        notes="This is the slide that generalises beyond this project: the specialist "
              "models lose not because the script is under-resourced but because the "
              "GENRE is. Sparse numeric tables are excluded from the training data by "
              "construction.\n\n"
              "The day-name hallucination usually gets a laugh. Let it.")

    add_slide(
        prs, kicker="slides 14–15 · the counting problem",
        title="The model returns a flat list; PAGE XML needs (row, column)",
        image="fig14_counting_problem.jpg",
        caption="Page 9: dense entries at the top, then scattered faint ones, then ruled "
                "blanks. Nothing tells the model how many rows it has passed.",
        points=[
            "No guarantee the model's row N is the grid's row N.",
            "On a sparse page it cannot tell a blank ruled row from a faint written one, "
            "so it drops rows — and everything below shifts. **Page 9: 21 of 24 rows.**",
            "Everything that gave the model **less** context failed: banded crops, "
            "column strips (Block_No 71.4% → 0.0%), cell crops.",
            "Even feeding it perfectly correct geometry did not help.",
        ],
        notes="The column-strip result is worth stating: giving the model a cleaner, "
              "more focused crop made it dramatically WORSE. That is counter-intuitive "
              "and worth the audience's attention.")

    add_slide(
        prs, kicker="slide 15 · two fixes, different in kind",
        title="Delete the task",
        image="fig15_set_of_marks.jpg",
        caption="Set-of-marks: the indices come from the grid we already computed. "
                "The model is no longer counting anything.",
        points=[
            "**The measurement was wrong.** The scorer assumed a constant row offset, so "
            "one dropped row charged every row below it as a misread. Needleman–Wunsch "
            "alignment recovered **20–28% — with zero API calls.**",
            "**Then: delete the task.** Print the row index into the margin, from "
            "geometry we already had, and ask the model to key each row to the number.",
            "**Missed rows 23 → 1–4.** ~$0.011/page.",
        ],
        pull="Not a better model. No counting task.",
        notes="MEASURED NOISE: ±1 row/page, so ±2–3 over six pages. The improvement is "
              "~6–8 sd. '0 missed on 5 of 6 pages' is the best run, not the expected "
              "value — say this aloud.\n\n"
              "Also worth saying: bigger models and more thinking made this WORSE, not "
              "better. The fix was structural, not a matter of capability.")

    add_slide(
        prs, kicker="slide 16 · where I would like help", title="٣ → ٢",
        image="fig16_three_to_two.jpg",
        caption="The scribe's ٣ at reading size. Recomputed against the proxy GT: "
                "98 instances across the six pages.",
        points=[
            "Top substitution for **Gemini** (135), **Baseer** (67), and **Claude** alike.",
            "Three model families, three architectures, one scribe's hand.",
        ],
        pull="Not model-fixable. This is where I would like help.",
        pull_colour=BAD,
        notes="Close on this. It is the honest ending: the remaining error is in the "
              "source, not the system.\n\n"
              "The 135 figure is Gemini's own count over a different comparison set; "
              "98 is the recount on the six proxy-GT pages. Either is defensible — "
              "just be consistent about which you quote.\n\n"
              "Invite the room to try reading them.")

    add_slide(
        prs, kicker="open questions", title="What is still unsettled",
        points=[
            "Sparse-row skipping is improved but not solved; pages 9 and 10 remain the "
            "hard cases.",
            "Framing / through-line not yet fixed.",
            "Yiddish vocalization (YiDraCor) case not included — no room without cutting "
            "something load-bearing.",
            "Pending: row-strip variant of the Baseer test, to separate model deficit "
            "from crop deficit.",
            "Pending: rerun of gen2_sc_clean on the rebuilt corpus for a fair-footing "
            "number.",
        ],
        notes="Keep or cut depending on time. If the Q&A is likely to be strong, this "
              "slide seeds it well.")

    prs.save(OUT)
    print(f"wrote {OUT.name}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides, "
          f"{OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    build()

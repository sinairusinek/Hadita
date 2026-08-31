# Accurate Grid Line Detection in Scanned Historical Tabular Documents

Historical table extraction is challenging when the printed gridlines are very faint and overlaid by handwriting.  General-purpose table tools (Tabula, pdfplumber, etc.) only handle clean, machine-generated tables – they explicitly “do not work as well” on *scanned, handwritten text*【21†L121-L124】【21†L162-L165】.  In fact, a 2024 survey noted “all the free tools struggle with handwritten analysis”【24†L1-L4】.  Modern deep-learning table detectors (CascadeTabNet, TableFormer, etc.) excel on typed documents【18†L246-L254】 but often fail on degraded historical forms.  One review reports that CascadeTabNet “demonstrates excellence” on ICDAR benchmarks but its performance “is less effective for the complex requirements of historical documents”【18†L268-L277】.  In short, off-the-shelf table OCR methods are insufficient. We need a specialized approach for this 1930s tax register’s fixed but faint grid.  

## Existing Tools & Research Approaches  
- **Table segmentation tools:**  The **Taulu** Python package is designed for historical tables【6†L41-L49】. It requires annotating a representative table header and then automatically finds the intersections of horizontal/vertical rulings to map out every cell【12†L392-L396】.  Taulu’s *GridDetector* uses custom line-shaped kernels and dilation to connect broken lines, then detects rule intersections【12†L427-L435】. This is open source and directly targets precisely the problem of segmenting cells in photographed or scanned tables.  (In Taulu’s documentation: “the algorithm identifies the header’s location…then scans the image to find intersections of the rules (borders) and segments the image into cells accordingly”【12†L392-L396】.)  
- **Other general tools:**  Most free tools (Tabula, Camelot, pdfplumber, PaddleOCR, DocTR, etc.) are geared toward digital-born PDFs or OCR text layers. For example, pdfplumber “does an exceptional job at extracting lines, intersections, cells, and tables” from clean PDFs【21†L129-L138】, but it “does not work as well” on scanned handwritten forms【21†L162-L165】. PaddleOCR is good for multilingual text and images, but even it “struggles to analyze handwritten text”【21†L198-L200】. None of these provide pixel-exact ruling positions on noisy scans.  
- **Historical table research:** Academic work has proposed specialized methods. One paper uses a fixed-table *template* and matches detected separators to it via graph matching【64†L1-L4】.  For a consistent form (like a fixed 34×26 grid), a template lets you enforce the known number of rows/columns even if line spacing varies【64†L1-L4】.  Other studies treat table structure recognition as a deep-learning task (e.g. CascadeTabNet, DiT, TableFormer【18†L246-L254】) or use semi-supervised learning on historical table datasets【18†L237-L242】.  But these generally assume large annotated datasets or deal with whole-cell extraction, not the pixel-level faint-line problem.  

## Image-Processing Techniques for Line Extraction  
Given the faintness of the printed lines, a proven approach is to use **morphological filtering plus line detection**:  

- **Adaptive Binarization:** First apply a robust threshold (e.g. Sauvola or adaptive mean).  The Taulu docs suggest tuning the threshold parameter (`sauvola_k`) so as to remove most noise without erasing the thin printed lines【12†L455-L459】. This creates a binary image where the lines, though faint, should still appear as thin dark strokes.  

- **Edge Detection & Closing:** Detect edges (e.g. with Canny or Sobel).  Historical-table pipelines often start with Canny and then *close* gaps: “Many images contain dotted lines for some of the table lines, so we pass the image through a closing morphological operation using a straight-line filter kernel”【56†L575-L581】. This dilation+erosion will connect broken line fragments caused by handwriting or printing artifacts.  

- **Line-Enhancing Filtering:** Apply morphological **opening** (erosion then dilation) with a long horizontal kernel (and separately a vertical kernel).  The intent is to “maximize the effect of the cell lines and minimize the effect of the handwritten text”【56†L575-L581】. In practice, use a structuring element shaped like a horizontal line (e.g. 1×50 pixels) to preserve long horizontal runs, and likewise a vertical line for vertical rules. The OpenCV tutorial on line detection confirms that carefully chosen linear kernels can extract straight lines【51†L42-L50】. After this step, each real table line will appear as a relatively long continuous blob, while most handwriting strokes (which are shorter or non-linear) will be largely removed.  

- **Dilate & Hough Transform:** After opening, dilate the result slightly so each line appears as one thick band. Then run a Hough Line Transform (or Probabilistic Hough) to detect the straight horizontal and vertical lines.  This approach was explicitly used in a handwritten census-form segmentation: “the lines are now fairly well defined…we then perform the Hough line filter algorithm on the image to detect all the horizontal and vertical lines”【56†L575-L581】.  (Equivalently, one could find connected components and take their bounding boxes as line candidates.)  

- **Kernel Matching / Filters:** Another tactic is to convolve the image with line-like kernels.  For example, convolving with a wide horizontal line (or using a matched filter) will yield high responses where horizontal rulings lie. Similarly, a vertical line filter can highlight column borders.  This is conceptually similar to the morphological steps above.  

In summary, a classic pipeline is: binarize (Sauvola), Canny edges, close with linear kernel, open horizontally/vertically, dilate, then Hough or connected-component detection. OpenCV’s documentation even demonstrates this idea: “apply morphology (dilation/erosion) with custom kernels to extract straight lines on the horizontal and vertical axes”【51†L42-L50】.  

## Distinguishing Printed Lines from Handwriting  
A key difficulty is separating the *faint printed grid* from overlying scribbles. Strategies include:  

- **Line Regularity:** Printed lines are globally straight and often uniform thickness. In contrast, handwriting produces irregular blobs and strokes.  Thus one can classify detected segments by length/straightness. For example, after filtering there should be many candidate line segments; keep only the ones that span most of the page width (for rows) or height (for columns).  

- **Run-Length Smoothing (RLSA):** Apply RLSA along each row to connect gaps, forming *pseudo-lines*.  Belaïd et al. used RLSA to group connected text into pseudo-lines and then classified each as handwritten vs. printed with an SVM【28†L53-L62】. They reported ~90% accuracy on complex documents.  Similarly, one could smooth long horizontal runs (e.g. connect all black pixels within 5px vertically) and then label runs: those that are nearly horizontal lines (constant y, uniform thickness) are likely printed, whereas clustered blobs with jagged edges are ink.  

- **Pixel Neighborhood Features:** In a learned approach, a CNN or U-Net could be trained (or fine-tuned) to segment the image into “line” vs “non-line” pixels. Indeed, there is research on CNN-based ruled-line removal in handwritten docs (e.g. Catalpa’s ruled-line-removal project【28†L53-L62】).  Although such models require training data, they can be effective at picking up very faint lines while ignoring text.  

- **Template Matching / Priors:** If the form layout is fixed (e.g. exactly 34 rows and known column splits), one can enforce that.  For instance, match the strongest 34 horizontal detections to the expected row positions (skipping spurious ones). Couasnon et al. used a fixed-table template: the form’s separators are matched to a template graph to allow slight variations in spacing【64†L1-L4】. In practice, you could sort candidate lines by y-coordinate and pick the 34 most regularly spaced ones.  

Each of these ideas (morphology, RLSA, machine learning, template constraints) helps filter out handwriting. In practice a **hybrid pipeline** is best: use morphology/Hough to get line candidates, then prune using priors or simple classifiers.  

## Proposed Pipeline

Based on the above, a robust workflow might be:

1. **Preprocess:** If not already done, crop out any black scanner border and ensure the image is well-deskewed (as you have). Optionally apply a mild blur to reduce speckle.  
2. **Adaptive Binarization:** Convert to grayscale and apply Sauvola or adaptive threshold. Tune parameters so printed lines (even if 1–2 pixels thick and light) remain.  
3. **Edge/Line Filtering:** Compute a Canny edge map (to emphasize edges) and perform a *closing* with a linear kernel (e.g. 30–50px horizontal) to join broken segments【56†L575-L581】.  
4. **Morphological Opening:** Separately apply horizontal and vertical opening: erosion followed by dilation with a long 1×N or N×1 kernel. Combine the results (OR)【56†L575-L581】. This isolates the maximal straight runs.  
5. **Dilate and Detect Lines:** Dilate the result slightly to thicken lines, then run a Hough transform (or `HoughLinesP`) on the binary image to extract exact line equations. Filter these by orientation (keep ≈0° or 90°) and by length.  
6. **Select True Grid Lines:** From the Hough output, pick the 34 horizontal lines (for rows) and 26 vertical lines (for columns) that best fit the expected grid.  For example, sort horizontal lines by y-coordinate and choose 34 evenly covering the range.  If using Taulu, it automatically does a corner match given a header template【12†L394-L396】.  
7. **Refinement:** If lines are broken, adjust endpoints by interpolation or using intersections. Optionally re-project lines back onto the original image to correct for any small skew left.  

At each stage, one can visualize intermediate masks (morphology result, Hough lines) to tune kernel sizes.  Many of these steps can be done in OpenCV or skimage. The key is to emphasize straight, long structures and suppress local ink noise. 

## Validation Metrics and Testing

To measure accuracy, define ground truth and metrics:  

- **Pixel Accuracy / Deviation:** Since you need sub-5px precision, compute the average vertical (for horizontal lines) or horizontal (for vertical lines) displacement between detected and true line positions.  You can also use RMSE of line coordinates. A successful method should have <5px mean error.  
- **Line Detection F1:** Treat each true line as a ground-truth object. Compute how many detected lines match (within a tolerance) and how many false positives occur.  Precision, recall, and F1-score give a sense of reliability. (In table segmentation studies, a weighted F1 score is often used【18†L237-L242】.)  
- **Cell IoU:** Since you’ll crop cells, measure the overlap between the predicted cell boxes and true cells.  The TEDS (Table Edit Distance Similarity) metric or cell-by-cell IoU could be used, but a simple recall of correctly identified cells is informative.  

For testing, select a **representative sample** of pages (e.g. 20–50) covering variations in ink density, handwriting style, and paper aging.  Manually annotate the printed lines (or even full cell polygons) on these pages.  Run the pipeline and compute the above metrics.  Iterate on parameters (kernel sizes, thresholds) to maximize F1 and minimize pixel error.  As an initial goal, ensure the crop tool produces clean cell images with virtually no rim artifacts; any systematic offset over 5px can be tuned out. 

In summary, combining classical image processing (adaptive binarization + morphology) with domain knowledge (fixed table template, run-length smoothing) is likely the most reliable approach.  Tools like Taulu offer a starting point【6†L41-L49】【12†L392-L396】, and the literature provides pipelines (Canny + close/open + Hough) that have worked on analogous historical forms【56†L575-L581】【51†L42-L50】.  By tuning these methods to the TR/39 form and validating on labeled pages, you can achieve the required pixel-accurate grid line detection.

**Sources:** Prior work on table segmentation and line removal【6†L41-L49】【12†L392-L396】【56†L575-L581】, open-source table tools reviews【21†L121-L124】【21†L162-L165】【24†L1-L4】, and historical document analysis research【18†L268-L277】【28†L53-L62】【64†L1-L4】.
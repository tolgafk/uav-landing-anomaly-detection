from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    FrameBreak,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"


TITLE = (
    "Self-Supervised Memory-Bank Anomaly Detection for UAV Landing Pads: "
    "Whole-Pad versus Grid-Hybrid Representations under Synthetic Occlusions"
)

ABSTRACT = (
    "Autonomous landing requires rejection of pads affected by previously unseen objects or surface "
    "changes. This paper compares two nominal-only, self-supervised anomaly detectors for UAV landing "
    "areas. Zero-margin UAP/UAI regions are encoded by a ResNet-101 trained with a SimCLR objective, "
    "retained in a nominal memory bank, and scored by one minus maximum cosine similarity. Wholepad SSL "
    "embeds the complete region; Grid-Hybrid SSL splits it into a 2 x 2 grid and uses the maximum tile "
    "anomaly. Evaluation uses 630 clean test pads and 630 paired synthetic obstructions generated with "
    "randomized geometry, 22-65% coverage, contrast-aware color, texture, contours, and shadows. "
    "Wholepad SSL achieves 0.9236 AUROC, 0.8583 F1, and 0.8563 accuracy. Grid-Hybrid SSL obtains 0.7555 "
    "AUROC and, at its stored threshold, 0.1667 F1 because recall falls to 0.0921. Threshold analysis "
    "improves the grid result but does not close the AUROC gap. The findings support whole-pad "
    "representation for the tested obstruction family while identifying calibration, feature "
    "dimensionality, and synthetic-to-real transfer as central limitations."
)

KEYWORDS = (
    "UAV; safe landing; anomaly detection; self-supervised learning; SimCLR; "
    "memory bank; synthetic occlusion"
)


SECTIONS = [
    (
        "1. Introduction",
        [
            (
                "Reliable autonomous landing requires more than locating a designated pad. A vehicle "
                "must also reject a pad whose appearance has changed because of an unexpected object, "
                "surface damage, or partial occlusion. Supervised detectors such as YOLO are effective "
                "for known object categories, but their closed-set formulation does not directly cover "
                "every hazard that can occur after deployment [4], [8]. This motivates a nominal-only "
                "formulation: learn the visual distribution of safe landing pads and flag inputs that "
                "depart from that distribution."
            ),
            (
                "This work studies that formulation with two self-supervised representations. "
                "<i>Wholepad SSL</i> embeds the complete pad region of interest (ROI). "
                "<i>Grid-Hybrid SSL</i> divides the same ROI into a 2 x 2 grid, embeds each tile, and "
                "uses the most anomalous tile as the pad score. Both methods use contrastive "
                "representation learning, a memory bank of nominal embeddings, and nearest-neighbor "
                "cosine similarity. Their evaluation is intentionally focused on synthetically "
                "generated pad obstructions, because the available test split contains too few "
                "consistently labeled, real anomalous pads for a controlled comparison."
            ),
            (
                "The contributions are threefold. First, a reproducible synthetic evaluation procedure "
                "creates geometrically and photometrically diverse pad occlusions while retaining "
                "paired clean controls. Second, global and tiled representations are compared under the "
                "same ROI and sample protocol. Third, the analysis separates threshold-free ranking "
                "quality from threshold-dependent operational behavior, exposing a large calibration "
                "sensitivity in the tiled model. On 630 clean and 630 paired synthetic samples, "
                "Wholepad SSL reaches 0.924 AUROC and 0.858 F1, whereas Grid-Hybrid SSL reaches 0.756 "
                "AUROC and 0.167 F1 at its stored operating threshold."
            ),
        ],
    ),
    (
        "2. Related Work",
        [
            (
                "Contrastive self-supervision learns invariances by bringing two augmented views of an "
                "image together while separating other instances. SimCLR operationalizes this idea "
                "with the normalized temperature-scaled cross-entropy (NT-Xent) loss [1]. Contrastive "
                "objectives have also been adapted to novelty detection, where transformations and "
                "representation geometry help distinguish nominal data from shifted inputs [7]. "
                "Recent surveys place such methods within a broader family of deep anomaly detectors "
                "that learn a description of normality rather than a closed set of abnormal classes "
                "[8], [12]."
            ),
            (
                "Memory-bank methods retain nominal embeddings and score a query by its distance from "
                "nearby references. PatchCore demonstrates the value of patch representations and "
                "compact nominal memories for industrial anomaly detection [2]; PaDiM models "
                "patch-level feature distributions [5]; and deep pyramid correspondences show that "
                "sub-image matching can reveal local anomalies [6]. PANDA similarly adapts deep "
                "features for anomaly detection [9]. The present study adopts nearest-neighbor scoring "
                "but compares two ROI granularities for aerial landing pads."
            ),
            (
                "Visual safe-landing systems have long combined onboard vision with geometric or "
                "learned decision rules [10]. More recent work studies safe landing-zone mapping in "
                "dynamic scenes [11], while DroneSSL applies self-supervision to multimodal anomaly "
                "detection in drone systems [13]. Our scope is narrower: controlled synthetic "
                "obstruction generation and appearance-based anomaly detection inside labeled "
                "landing-pad ROIs."
            ),
        ],
    ),
    (
        "3. Method",
        [
            (
                "<b>3.1 ROI construction and nominal learning.</b> The source collection is the "
                "TEKNOFEST Aviation Artificial Intelligence data set, which contains more than 10,000 "
                "aerial images with YOLO-format annotations [14]. Only the UAP and UAI landing-area "
                "classes are used to extract ROIs. Bounding boxes are converted from normalized "
                "coordinates to pixels and cropped with zero margin so that background texture outside "
                "the pad does not dominate the representation. ROIs smaller than 32 pixels in either "
                "dimension are discarded. All retained inputs are resized to 224 x 224 pixels."
            ),
            (
                "A ResNet-101 encoder [3] is optimized on nominal pad crops with SimCLR. For a pad x, "
                "two stochastic augmentations form a positive pair. For normalized projected "
                "embeddings z_i and z_j, the NT-Xent objective is given in Eq. (1)."
            ),
            "FORMULA_NTXENT",
            (
                "where sim is cosine similarity and T = 0.2. Both implementations train for 20 epochs "
                "with AdamW, batch size 8, learning rate 3 x 10^-4, and random seed 42. The active "
                "experiment uses no supervised anomaly penalty; synthetic anomalies are introduced "
                "only for evaluation."
            ),
            (
                "<b>3.2 Whole-pad and grid-hybrid representations.</b> Wholepad SSL preserves the "
                "complete zero-margin crop. The memory bank contains 5,968 L2-normalized, "
                "2,048-dimensional encoder outputs. Given query embedding z and nominal bank B, "
                "the whole-pad anomaly score is defined in Eq. (2)."
            ),
            "FORMULA_WHOLE",
            (
                "Grid-Hybrid SSL divides each ROI into four fixed quadrants. Each tile is independently "
                "resized and encoded in the 128-dimensional projection space. The resulting bank "
                "contains 23,872 tile embeddings, exactly four times the whole-pad bank count. "
                "A query tile is compared with the pooled bank of all nominal tile positions, and "
                "the maximum tile anomaly becomes the pad score in Eq. (3)."
            ),
            "FORMULA_GRID",
            (
                "A pad is labeled <font name='Times-Bold'>UNSAFE</font> when A(x) is at least the "
                "threshold tau and <font name='Times-Bold'>SAFE</font> otherwise. Wholepad uses a "
                "Youden-J threshold of 0.01419 because no prior threshold was available. Grid-Hybrid "
                "uses the stored deployment threshold 0.1400. AUROC is reported independently of "
                "these operating points."
            ),
            (
                "<b>3.3 Synthetic obstruction generation.</b> Each accepted clean test ROI produces a "
                "paired anomalous copy. A random target coverage between 22% and 65% of the ROI is "
                "sampled, followed by one of four masks: rotated rectangle, ellipse, band, or irregular "
                "eight-vertex polygon. Position, aspect ratio, and orientation are randomized. The fill "
                "color is selected from high-contrast candidates relative to the ROI median. Diagonal "
                "stripes, a contrasting contour, and an offset darkened shadow add internal texture "
                "and boundary cues."
            ),
            (
                "The test scan covers 671 images and 650 annotated pad candidates. Twenty candidates "
                "are removed by the minimum-size rule, leaving 630 clean ROIs. Their paired synthetic "
                "copies yield a balanced set of 1,260 samples. Because each synthetic image is derived "
                "from its clean counterpart, the design controls pad identity but does not constitute "
                "an independent sample of real hazards."
            ),
        ],
    ),
]


RESULT_PARAGRAPHS = [
    (
        "Table 1 shows a 0.1681 absolute AUROC advantage for Wholepad SSL. Its operating point is "
        "also balanced: 548 of 630 synthetic anomalies are detected while 531 of 630 clean pads "
        "remain accepted. Grid-Hybrid produces only eight false alarms and consequently high "
        "precision, but misses 572 synthetic hazards. For a landing-safety gate, this false-negative "
        "concentration is more consequential than the low false-positive count."
    ),
    (
        "The ROC curves in Figs. 3 and 4 confirm that the conclusion is not solely a consequence of threshold "
        "selection. Wholepad ranks paired anomalies above clean pads more reliably across the full "
        "range of false-positive rates. Score histograms in Figs. 5 and 6 further show a clearer distribution "
        "shift for the global representation; Grid-Hybrid retains substantial overlap below its high "
        "stored threshold."
    ),
    (
        "Grid-Hybrid is highly sensitive to calibration. Replacing its stored threshold with the "
        "Youden threshold, 0.03994, raises recall from 0.0921 to 0.6571 and F1 from 0.1667 to 0.6832, "
        "with specificity 0.7333. The best-F1 threshold on this evaluation set, 0.02190, yields 0.8603 "
        "recall and 0.7217 F1 but only 0.4762 specificity. These are diagnostic, post-hoc operating "
        "points rather than unbiased deployment estimates. Wholepad is more stable: its best-F1 "
        "threshold of 0.01166 produces 0.9111 recall and 0.8606 F1, close to its Youden-selected result."
    ),
]


DISCUSSION = [
    (
        "The global representation appears better matched to this synthetic protocol. The occlusions "
        "cover up to 65% of a pad and deliberately introduce strong contrast, structured texture, "
        "contours, and shadows. Such changes alter the overall pad representation and are readily "
        "exposed by a nearest-neighbor comparison. The tiled model was designed to prevent small "
        "hazards from being diluted, yet its active threshold suppresses most positive decisions."
    ),
    (
        "Representation granularity is not the only difference between the systems. Wholepad scoring "
        "uses 2,048-dimensional encoder features, whereas Grid-Hybrid uses 128-dimensional projector "
        "features; the latter also compares every quadrant against a position-pooled bank. A "
        "normal-looking tile from any quadrant can therefore provide a strong match, and the "
        "lower-dimensional projection may discard details useful for anomaly separation. Consequently, "
        "the experiment supports the complete Wholepad pipeline but does not isolate a causal advantage "
        "for global cropping alone. A controlled ablation should keep feature space, bank sampling, "
        "augmentation policy, and threshold calibration fixed."
    ),
    (
        "The synthetic benchmark has four further limitations. First, procedural obstructions are "
        "conspicuous and do not cover all real hazards, including transparent materials, low-contrast "
        "debris, specular surfaces, or shadows without physical obstacles. Second, each anomalous "
        "sample shares its background with a clean sample, so confidence intervals must respect pairing. "
        "Third, the threshold fallback and post-hoc alternatives are computed on the evaluated scores; "
        "a separate calibration split is required for deployment. Fourth, the current results measure "
        "discrimination, not latency, memory transfer, or end-to-end landing risk."
    ),
    (
        "Despite these constraints, the paired protocol is useful for controlled development. It "
        "exposes the effect of an obstruction without changing viewpoint or pad identity and reveals "
        "whether a model's score responds in the intended direction. Future work should validate on "
        "real, independently annotated pad hazards; calibrate thresholds on a held-out split; compare "
        "position-specific and coreset memory banks; and report resource use on the target UAV computer."
    ),
]


CONCLUSION = (
    "This study evaluated two self-supervised, memory-bank anomaly detectors for UAV landing pads "
    "using reproducible synthetic occlusions. Under a balanced 1,260-sample paired test, Wholepad SSL "
    "achieved 0.9236 AUROC and balanced precision-recall behavior, while Grid-Hybrid SSL reached "
    "0.7555 AUROC and suffered severe false negatives at its stored threshold. The results favor the "
    "complete Wholepad pipeline for the tested obstruction family and show that threshold calibration "
    "can materially change the apparent safety profile of the tiled model. Real-hazard validation and "
    "controlled representation ablations remain necessary before flight deployment."
)


REFERENCES = [
    "[1] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, \"A simple framework for contrastive learning of visual representations,\" Proc. ICML, pp. 1597-1607, 2020.",
    "[2] K. Roth, L. Pemula, J. Zepeda, B. Schölkopf, T. Brox, and P. Gehler, \"Towards total recall in industrial anomaly detection,\" Proc. IEEE/CVF CVPR, pp. 14318-14328, 2022.",
    "[3] K. He, X. Zhang, S. Ren, and J. Sun, \"Deep residual learning for image recognition,\" Proc. IEEE CVPR, pp. 770-778, 2016.",
    "[4] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, \"You only look once: Unified, real-time object detection,\" Proc. IEEE CVPR, pp. 779-788, 2016.",
    "[5] T. Defard, A. Setkov, A. Loesch, and R. Audigier, \"PaDiM: A patch distribution modeling framework for anomaly detection and localization,\" ICPR Workshops, pp. 475-489, 2021.",
    "[6] N. Cohen and Y. Hoshen, \"Sub-image anomaly detection with deep pyramid correspondences,\" Proc. ICML, pp. 2058-2067, 2020.",
    "[7] J. Tack, S. Mo, J. Jeong, and J. Shin, \"CSI: Novelty detection via contrastive learning on distributionally shifted instances,\" NeurIPS, vol. 33, pp. 11839-11852, 2020.",
    "[8] L. Ruff et al., \"A unifying review of deep and shallow anomaly detection,\" Proc. IEEE, vol. 109, no. 5, pp. 756-795, 2021.",
    "[9] T. Reiss, N. Cohen, L. Bergman, and Y. Hoshen, \"PANDA: Adapting pretrained features for anomaly detection and segmentation,\" Proc. IEEE/CVF CVPR, pp. 2806-2814, 2021.",
    "[10] S. Yang, S. A. Scherer, and A. Zell, \"An onboard visual autonomous landing system for quadrotors in GPS-denied environments,\" Proc. IEEE ICRA, pp. 2736-2741, 2013.",
    "[11] S. Abdollahzadeh, M. S. Allili, A. Boulmerka, and J.-F. Lapointe, \"A vision-based framework for safe landing zone mapping of UAVs in dynamic environments,\" IEEE Open J. Comput. Soc., 2026, doi: 10.1109/OJCS.2026.3663268.",
    "[12] H. Hojjati, T. K. K. Ho, and N. Armanfard, \"Self-supervised anomaly detection in computer vision and beyond: A survey and outlook,\" arXiv:2205.05173, 2022.",
    "[13] J. Akram, A. Anaissi, W. Othman, A. Alabdulatif, and A. Akram, \"DroneSSL: Self-supervised multimodal anomaly detection in Internet of Drone Things,\" IEEE Trans. Consum. Electron., vol. 70, no. 1, pp. 4287-4298, 2024.",
    "[14] TEKNOFEST, \"Havacılıkta Yapay Zeka Yarışması 2026 data set,\" 2026. https://drive.google.com/drive/folders/18_VqLBbyTubVSWAXG_CgmuJWGCx0mcBd",
]


def _font_setup() -> None:
    candidates = [
        (
            Path("C:/Windows/Fonts/times.ttf"),
            Path("C:/Windows/Fonts/timesbd.ttf"),
            Path("C:/Windows/Fonts/timesi.ttf"),
        ),
        (
            Path("C:/Windows/Fonts/calibri.ttf"),
            Path("C:/Windows/Fonts/calibrib.ttf"),
            Path("C:/Windows/Fonts/calibrii.ttf"),
        ),
    ]
    for regular, bold, italic in candidates:
        if regular.exists() and bold.exists() and italic.exists():
            pdfmetrics.registerFont(TTFont("PaperSerif", str(regular)))
            pdfmetrics.registerFont(TTFont("PaperSerif-Bold", str(bold)))
            pdfmetrics.registerFont(TTFont("PaperSerif-Italic", str(italic)))
            pdfmetrics.registerFontFamily(
                "PaperSerif",
                normal="PaperSerif",
                bold="PaperSerif-Bold",
                italic="PaperSerif-Italic",
                boldItalic="PaperSerif-Bold",
            )
            return
    raise RuntimeError("No suitable Windows serif font was found.")


def _styles(variant: str):
    base = getSampleStyleSheet()
    if variant == "ieee":
        body_size, leading, title_size = 9.7, 11.1, 17.5
        section_size, subsection_size = 10.5, 9.8
    else:
        body_size, leading, title_size = 9.3, 10.7, 16.0
        section_size, subsection_size = 10.3, 9.5

    body = ParagraphStyle(
        "Body",
        parent=base["BodyText"],
        fontName="PaperSerif",
        fontSize=body_size,
        leading=leading,
        alignment=TA_JUSTIFY,
        spaceAfter=4.5,
        allowWidows=0,
        allowOrphans=0,
    )
    return {
        "title": ParagraphStyle(
            "Title",
            parent=body,
            fontName="PaperSerif-Bold",
            fontSize=title_size,
            leading=title_size + 1.8,
            alignment=TA_CENTER,
            spaceAfter=6,
        ),
        "author": ParagraphStyle(
            "Author",
            parent=body,
            fontSize=10.0,
            leading=12,
            alignment=TA_CENTER,
            spaceAfter=5,
        ),
        "abstract": ParagraphStyle(
            "Abstract",
            parent=body,
            fontSize=8.3,
            leading=9.3,
            leftIndent=0.18 * inch,
            rightIndent=0.18 * inch,
            spaceAfter=3,
        ),
        "keywords": ParagraphStyle(
            "Keywords",
            parent=body,
            fontSize=8.1,
            leading=9,
            leftIndent=0.18 * inch,
            rightIndent=0.18 * inch,
            spaceAfter=1,
        ),
        "section": ParagraphStyle(
            "Section",
            parent=body,
            fontName="PaperSerif-Bold",
            fontSize=section_size,
            leading=section_size + 1.5,
            alignment=TA_LEFT,
            spaceBefore=5,
            spaceAfter=3,
            keepWithNext=True,
        ),
        "subsection": ParagraphStyle(
            "Subsection",
            parent=body,
            fontName="PaperSerif-Bold",
            fontSize=subsection_size,
            leading=subsection_size + 1,
            spaceBefore=3,
            spaceAfter=2,
            keepWithNext=True,
        ),
        "body": body,
        "caption": ParagraphStyle(
            "Caption",
            parent=body,
            fontSize=7.8,
            leading=8.8,
            alignment=TA_CENTER,
            spaceBefore=2,
            spaceAfter=5,
        ),
        "reference": ParagraphStyle(
            "Reference",
            parent=body,
            fontSize=8.0 if variant == "ieee" else 7.8,
            leading=9.1 if variant == "ieee" else 8.9,
            leftIndent=0.16 * inch,
            firstLineIndent=-0.16 * inch,
            spaceAfter=2.5,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=body,
            fontSize=7.8,
            leading=8.8,
        ),
    }


class PaperDoc(BaseDocTemplate):
    def __init__(self, filename: str, variant: str):
        self.variant = variant
        self.styles = _styles(variant)
        super().__init__(
            filename,
            pagesize=letter,
            leftMargin=0.68 * inch if variant == "ieee" else 0.70 * inch,
            rightMargin=0.68 * inch if variant == "ieee" else 0.70 * inch,
            topMargin=0.52 * inch,
            bottomMargin=0.52 * inch,
            title=TITLE,
            author="Tolga Ferhan Küçük",
            subject="Self-supervised anomaly detection for UAV landing pads",
        )
        self._create_templates()

    def _create_templates(self):
        page_w, page_h = letter
        left = self.leftMargin
        right = self.rightMargin
        bottom = self.bottomMargin
        top = self.topMargin
        usable_w = page_w - left - right
        usable_h = page_h - top - bottom
        gap = 0.20 * inch if self.variant == "ieee" else 0.22 * inch
        col_w = (usable_w - gap) / 2

        title_h = 3.55 * inch if self.variant == "ieee" else 3.42 * inch
        title_frame = Frame(
            left,
            page_h - top - title_h,
            usable_w,
            title_h,
            id="title",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=4,
        )
        body_h_first = usable_h - title_h
        first_left = Frame(
            left,
            bottom,
            col_w,
            body_h_first,
            id="first-left",
            leftPadding=0,
            rightPadding=0,
            topPadding=2,
            bottomPadding=0,
        )
        first_right = Frame(
            left + col_w + gap,
            bottom,
            col_w,
            body_h_first,
            id="first-right",
            leftPadding=0,
            rightPadding=0,
            topPadding=2,
            bottomPadding=0,
        )
        col_left = Frame(
            left,
            bottom,
            col_w,
            usable_h,
            id="left",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        col_right = Frame(
            left + col_w + gap,
            bottom,
            col_w,
            usable_h,
            id="right",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )

        wide_h = 7.30 * inch
        wide = Frame(
            left,
            page_h - top - wide_h,
            usable_w,
            wide_h,
            id="wide",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=4,
        )
        results_h = usable_h - wide_h
        results_left = Frame(
            left,
            bottom,
            col_w,
            results_h,
            id="results-left",
            leftPadding=0,
            rightPadding=0,
            topPadding=2,
            bottomPadding=0,
        )
        results_right = Frame(
            left + col_w + gap,
            bottom,
            col_w,
            results_h,
            id="results-right",
            leftPadding=0,
            rightPadding=0,
            topPadding=2,
            bottomPadding=0,
        )

        self.addPageTemplates(
            [
                PageTemplate(
                    "First",
                    [title_frame, first_left, first_right],
                    onPage=self._decorate,
                    autoNextPageTemplate="Later",
                ),
                PageTemplate("Later", [col_left, col_right], onPage=self._decorate),
                PageTemplate(
                    "Workflow",
                    [wide, results_left, results_right],
                    onPage=self._decorate,
                    autoNextPageTemplate="Later",
                ),
            ]
        )

    def _decorate(self, canvas, doc):
        canvas.saveState()
        page_w, page_h = letter
        if self.variant == "ieee":
            header = "Author Preprint - UAV Landing-Pad Anomaly Detection"
            canvas.setFont("PaperSerif-Italic", 7.2)
        else:
            header = "Preprint, September 2026, Kocaeli, Türkiye"
            canvas.setFont("PaperSerif", 7.2)
        canvas.setFillColor(colors.HexColor("#4a4a4a"))
        canvas.drawString(self.leftMargin, page_h - 0.30 * inch, header)
        canvas.drawRightString(page_w - self.rightMargin, page_h - 0.30 * inch, f"{doc.page}")
        canvas.setStrokeColor(colors.HexColor("#b5b5b5"))
        canvas.setLineWidth(0.35)
        canvas.line(
            self.leftMargin,
            page_h - 0.36 * inch,
            page_w - self.rightMargin,
            page_h - 0.36 * inch,
        )
        canvas.restoreState()


def _title_flowables(styles, variant: str):
    venue = (
        "IEEE-style conference manuscript"
        if variant == "ieee"
        else "ACM SIGCONF-style manuscript"
    )
    return [
        Paragraph(TITLE, styles["title"]),
        Paragraph(
            "<b>Tolga Ferhan Küçük</b><br/>"
            "Department of Computer Engineering, Kocaeli University, Kocaeli, Türkiye<br/>"
            "kucuktolgaferhan@gmail.com &nbsp;&nbsp; https://github.com/tolgafk<br/>"
            f"<font size='7.5'>{venue}</font>",
            styles["author"],
        ),
        Paragraph(f"<b>Abstract-</b> {ABSTRACT}", styles["abstract"]),
        Paragraph(f"<b>Keywords-</b> {KEYWORDS}", styles["keywords"]),
        FrameBreak(),
    ]


def _main_table(styles):
    data = [
        [
            Paragraph("<b>Method</b>", styles["small"]),
            Paragraph("<b>AUROC</b>", styles["small"]),
            Paragraph("<b>Prec.</b>", styles["small"]),
            Paragraph("<b>Recall</b>", styles["small"]),
            Paragraph("<b>F1</b>", styles["small"]),
            Paragraph("<b>Acc.</b>", styles["small"]),
        ],
        ["Wholepad", "0.9236", "0.8470", "0.8698", "0.8583", "0.8563"],
        ["Grid-Hybrid", "0.7555", "0.8788", "0.0921", "0.1667", "0.5397"],
    ]
    widths = [0.82 * inch] + [0.48 * inch] * 5
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "PaperSerif"),
                ("FONTNAME", (0, 0), (-1, 0), "PaperSerif-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.6),
                ("LEADING", (0, 0), (-1, -1), 8.5),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
            ]
        )
    )
    return table


def _single_figure(name: str, caption: str, styles):
    figure = Image(str(FIGURES / name), width=2.90 * inch, height=1.90 * inch)
    return KeepTogether(
        [
            Table([[figure]], colWidths=[3.0 * inch], hAlign="CENTER"),
            Paragraph(caption, styles["caption"]),
        ]
    )


def _formula_image(name: str, width: float):
    from PIL import Image as PILImage

    path = FIGURES / name
    with PILImage.open(path) as im:
        aspect = im.height / im.width
    image = Image(str(path), width=width, height=width * aspect)
    holder = Table([[image]], colWidths=[3.0 * inch], hAlign="CENTER")
    holder.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return holder


def _workflow_figure(name: str, caption: str, styles):
    from PIL import Image as PILImage

    path = FIGURES / name
    width = 6.35 * inch
    with PILImage.open(path) as im:
        height = width * im.height / im.width
    figure = Image(str(path), width=width, height=height)
    return KeepTogether(
        [
            Table([[figure]], colWidths=[6.5 * inch], hAlign="CENTER"),
            Paragraph(caption, styles["caption"]),
        ]
    )


def _figure_pair(left_name: str, right_name: str, caption: str, styles):
    left = Image(str(FIGURES / left_name), width=3.1 * inch, height=2.05 * inch)
    right = Image(str(FIGURES / right_name), width=3.1 * inch, height=2.05 * inch)
    pair = Table([[left, right]], colWidths=[3.35 * inch, 3.35 * inch], hAlign="CENTER")
    pair.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return KeepTogether([pair, Paragraph(caption, styles["caption"])])


def build(path: Path, variant: str) -> None:
    doc = PaperDoc(str(path), variant)
    s = doc.styles
    story = []
    story.extend(_title_flowables(s, variant))

    for heading, paragraphs in SECTIONS[:2]:
        story.append(Paragraph(heading, s["section"]))
        for para in paragraphs:
            story.append(Paragraph(para, s["body"]))

    story.extend([NextPageTemplate("Workflow"), PageBreak()])
    story.append(Paragraph("3. Method", s["section"]))
    story.append(
        _workflow_figure(
            "wholepad_workflow.png",
            "<b>Fig. 1.</b> Wholepad SSL workflow reproduced directly from the thesis: "
            "ROI selection, self-supervised training, frozen encoding, memory-bank construction, "
            "and anomaly scoring.",
            s,
        )
    )
    story.append(
        _workflow_figure(
            "grid_workflow.png",
            "<b>Fig. 2.</b> Grid-Hybrid SSL workflow reproduced directly from the thesis. "
            "The pad ROI is divided into a 2 x 2 grid before representation learning and scoring.",
            s,
        )
    )
    story.append(FrameBreak())
    for para in SECTIONS[2][1]:
        if para == "FORMULA_NTXENT":
            story.append(_formula_image("ntxent_formula.png", 3.05 * inch))
        elif para == "FORMULA_WHOLE":
            story.append(_formula_image("whole_score_formula.png", 2.85 * inch))
        elif para == "FORMULA_GRID":
            story.append(_formula_image("grid_score_formula.png", 3.05 * inch))
        else:
            story.append(Paragraph(para, s["body"]))

    story.append(Paragraph("4. Results", s["section"]))
    story.append(
        Paragraph(
            "<b>Table 1.</b> Performance at the stored or fallback operating threshold. "
            "Positive denotes UNSAFE.",
            s["caption"],
        )
    )
    story.append(_main_table(s))
    story.append(Paragraph(RESULT_PARAGRAPHS[0], s["body"]))
    story.append(Paragraph(RESULT_PARAGRAPHS[1], s["body"]))
    story.append(Paragraph(RESULT_PARAGRAPHS[2], s["body"]))
    story.append(
        _single_figure(
            "wholepad_roc.png",
            "<b>Fig. 3.</b> Wholepad SSL ROC curve (630 clean and 630 synthetic samples).",
            s,
        )
    )
    story.append(
        _single_figure(
            "grid_roc.png",
            "<b>Fig. 4.</b> Grid-Hybrid SSL ROC curve on the same evaluation set.",
            s,
        )
    )
    story.append(
        _single_figure(
            "wholepad_hist.png",
            "<b>Fig. 5.</b> Wholepad SSL anomaly-score distribution and threshold.",
            s,
        )
    )
    story.append(
        _single_figure(
            "grid_hist.png",
            "<b>Fig. 6.</b> Grid-Hybrid SSL anomaly-score distribution and threshold.",
            s,
        )
    )

    story.append(Paragraph("5. Discussion", s["section"]))
    for para in DISCUSSION:
        story.append(Paragraph(para, s["body"]))
    story.append(Paragraph("6. Conclusion", s["section"]))
    story.append(Paragraph(CONCLUSION, s["body"]))
    story.append(Paragraph("References", s["section"]))
    for ref in REFERENCES:
        story.append(Paragraph(ref, s["reference"]))

    doc.build(story)


def main() -> None:
    _font_setup()
    build(HERE / "ieee_paper.pdf", "ieee")
    build(HERE / "acm_paper.pdf", "acm")


if __name__ == "__main__":
    main()

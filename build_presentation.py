"""
build_presentation.py
Generates a 12-slide widescreen PowerPoint presentation for the
Handwritten Sudoku Reader and Solver project.
"""

import sys
import shutil
from pathlib import Path
import pptx
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

ROOT_DIR = Path(__file__).resolve().parent

# Color Palette Constants
COLOR_BG_DARK     = RGBColor(15, 23, 42)     # Deep Slate 900
COLOR_BG_CARD     = RGBColor(30, 41, 59)     # Slate 800
COLOR_BG_LIGHT    = RGBColor(248, 250, 252)  # Slate 50
COLOR_CARD_LIGHT  = RGBColor(255, 255, 255)  # Pure White
COLOR_BORDER      = RGBColor(226, 232, 240)  # Slate 200
COLOR_BORDER_DARK = RGBColor(51, 65, 85)     # Slate 700

COLOR_PRIMARY     = RGBColor(37, 99, 235)    # Royal Indigo Blue #2563EB
COLOR_SECONDARY   = RGBColor(16, 185, 129)   # Emerald Green #10B981
COLOR_AMBER       = RGBColor(245, 158, 11)   # Warm Amber #F59E0B
COLOR_RED         = RGBColor(239, 68, 68)    # Coral Red #EF4444

COLOR_TEXT_MAIN   = RGBColor(15, 23, 42)     # Slate 900
COLOR_TEXT_MUTED  = RGBColor(71, 85, 105)    # Slate 600
COLOR_TEXT_WHITE  = RGBColor(255, 255, 255)
COLOR_TEXT_SLATE  = RGBColor(203, 213, 225)  # Slate 300

FONT_HEADING = "Segoe UI"
FONT_BODY    = "Segoe UI"


def create_deck():
    prs = pptx.Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def set_slide_background(slide, color):
        bg = slide.background
        fill = bg.fill
        fill.solid()
        fill.fore_color.rgb = color

    def add_card(slide, left, top, width, height, bg_color=COLOR_CARD_LIGHT, border_color=COLOR_BORDER):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        if border_color:
            shape.line.color.rgb = border_color
            shape.line.width = Pt(1.5)
        else:
            shape.line.fill.background()
        return shape

    def add_header(slide, title, category="PROJECT OVERVIEW", dark_mode=False):
        # Category Pill
        pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.45), Inches(2.8), Inches(0.35))
        pill.fill.solid()
        pill.fill.fore_color.rgb = RGBColor(30, 41, 59) if dark_mode else RGBColor(239, 246, 255)
        pill.line.color.rgb = RGBColor(51, 65, 85) if dark_mode else RGBColor(191, 219, 254)
        pill.line.width = Pt(1)
        tf = pill.text_frame
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.text = category.upper()
        p.font.name = FONT_HEADING
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = RGBColor(148, 163, 184) if dark_mode else COLOR_PRIMARY
        p.alignment = PP_ALIGN.CENTER

        # Title Text
        tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.85), Inches(11.7), Inches(0.7))
        tf_t = tb.text_frame
        tf_t.word_wrap = True
        p_t = tf_t.paragraphs[0]
        p_t.text = title
        p_t.font.name = FONT_HEADING
        p_t.font.size = Pt(22)
        p_t.font.bold = True
        p_t.font.color.rgb = COLOR_TEXT_WHITE if dark_mode else COLOR_TEXT_MAIN

    def add_speaker_notes(slide, notes):
        notes_slide = slide.notes_slide
        tf = notes_slide.notes_text_frame
        tf.text = notes.strip()

    # =========================================================================
    # SLIDE 1: Title Slide (Dark Theme)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1, COLOR_BG_DARK)

    # Accent Top Bar
    top_bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.12))
    top_bar.fill.solid()
    top_bar.fill.fore_color.rgb = COLOR_PRIMARY
    top_bar.line.fill.background()

    # Badge
    b1 = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.3), Inches(4.2), Inches(0.45))
    b1.fill.solid()
    b1.fill.fore_color.rgb = RGBColor(30, 41, 59)
    b1.line.color.rgb = COLOR_PRIMARY
    b1.line.width = Pt(1.5)
    p = b1.text_frame.paragraphs[0]
    p.text = "INTELLIGENT SYSTEMS & AI ARCHITECTURE"
    p.font.name = FONT_HEADING
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = RGBColor(147, 197, 253)
    p.alignment = PP_ALIGN.CENTER

    # Main Title
    t1 = s1.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(11.3), Inches(1.8))
    tf1 = t1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "Handwritten Sudoku Reader & Solver"
    p1.font.name = FONT_HEADING
    p1.font.size = Pt(40)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_TEXT_WHITE

    p1_sub = tf1.add_paragraph()
    p1_sub.text = "Bridging Deep Learning Perception, Informed Search, and Constraint Satisfaction"
    p1_sub.font.name = FONT_HEADING
    p1_sub.font.size = Pt(20)
    p1_sub.font.color.rgb = RGBColor(148, 163, 184)

    # 3 Pillar Cards
    pillars = [
        ("Layer 1: Perception", "DigitCNN (99.07% Accuracy)", "Classifies handwritten digits 1-9 with calibrated softmax probabilities (ECE=0.0041).", COLOR_PRIMARY),
        ("Layer 2: Repair Search", "A* & UCS with Conflict Heuristic", "Explores minimal log-probability repairs to resolve contradictory clues before solving.", COLOR_AMBER),
        ("Layer 3: Constraint Engine", "AC-3 + Forward Checking (FC)", "Enforces 27 Sudoku constraints with MRV & LCV heuristics to solve boards in <15 ms.", COLOR_SECONDARY),
    ]

    for i, (p_title, p_sub, p_desc, p_accent) in enumerate(pillars):
        c_left = Inches(1.0 + i * 3.85)
        c = add_card(s1, c_left, Inches(4.0), Inches(3.6), Inches(2.2), bg_color=COLOR_BG_CARD, border_color=COLOR_BORDER_DARK)
        
        # Pill inside card
        card_bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, c_left + Inches(0.2), Inches(4.2), Inches(1.2), Inches(0.06))
        card_bar.fill.solid()
        card_bar.fill.fore_color.rgb = p_accent
        card_bar.line.fill.background()

        tb = s1.shapes.add_textbox(c_left + Inches(0.2), Inches(4.35), Inches(3.2), Inches(1.7))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = p_title
        p.font.name = FONT_HEADING
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = p_accent

        p_s = tf.add_paragraph()
        p_s.text = p_sub
        p_s.font.name = FONT_HEADING
        p_s.font.size = Pt(11)
        p_s.font.bold = True
        p_s.font.color.rgb = COLOR_TEXT_WHITE

        p_d = tf.add_paragraph()
        p_d.text = p_desc
        p_d.font.name = FONT_BODY
        p_d.font.size = Pt(10)
        p_d.font.color.rgb = COLOR_TEXT_SLATE

    # Footer
    tb_foot = s1.shapes.add_textbox(Inches(1.0), Inches(6.5), Inches(11.3), Inches(0.5))
    p_foot = tb_foot.text_frame.paragraphs[0]
    p_foot.text = "Author: Kareem Elebiary   |   Hardware: Trained on NVIDIA RTX 5060 Ti (CUDA 13.2)   |   PyTorch & AC-3 Engine"
    p_foot.font.name = FONT_BODY
    p_foot.font.size = Pt(11)
    p_foot.font.color.rgb = RGBColor(148, 163, 184)

    add_speaker_notes(s1, """
Good morning everyone. Today I am presenting our project: The Handwritten Sudoku Reader and Solver.
In the real world, AI applications cannot rely on clean, perfect inputs. When a user snaps a photo of a newspaper or handwritten Sudoku with their smartphone, handwriting is ambiguous. A classic vision pipeline recognizes the digits greedily and passes them to a solver. If even a single digit is misrecognized—for instance, an open '7' misclassified as a '1'—the puzzle becomes mathematically unsolvable, causing a complete system crash.
Our architecture bridges neural network perception with symbolic constraint satisfaction by introducing an informed search layer that treats recognition confidence as a cost function, repairing ambiguous clues in milliseconds.
""")

    # =========================================================================
    # SLIDE 2: Executive Summary & The Core Challenge
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2, COLOR_BG_LIGHT)
    add_header(s2, "The Real-World Dilemma: Why Standard Solvers Fail on Real Images", "PROBLEM DEFINITION")

    # Left Column: The Problem Card
    c_prob = add_card(s2, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.1))
    tb_p = s2.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(5.2), Inches(4.7))
    tf_p = tb_p.text_frame
    tf_p.word_wrap = True

    p = tf_p.paragraphs[0]
    p.text = "The Disconnect in Traditional Pipelines"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = COLOR_RED

    points_p = [
        ("Greedy OCR Assumption", "Conventional systems take argmax(P(d|x)) for each cell independently, ignoring spatial constraints across rows, columns, and 3x3 boxes."),
        ("Catastrophic Propagation", "A single incorrect digit (e.g. duplicate 5 in row 2) introduces a contradiction. Standard CSP solvers return 'UNSATISFIABLE' (0% success)."),
        ("Combinatorial Impossibility", "Brute-force checking all 9^81 digit possibilities is impossible (O(9^N)). We need principled, minimal-cost repair search."),
    ]
    for h, b in points_p:
        p_h = tf_p.add_paragraph()
        p_h.text = f"- {h}: "
        p_h.font.bold = True
        p_h.font.size = Pt(12)
        p_h.font.color.rgb = COLOR_TEXT_MAIN
        p_b = tf_p.add_paragraph()
        p_b.text = f"  {b}"
        p_b.font.size = Pt(11)
        p_b.font.color.rgb = COLOR_TEXT_MUTED

    # Right Column: The Solution Card
    c_sol = add_card(s2, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.1))
    tb_s = s2.shapes.add_textbox(Inches(7.0), Inches(1.9), Inches(5.3), Inches(4.7))
    tf_s = tb_s.text_frame
    tf_s.word_wrap = True

    p = tf_s.paragraphs[0]
    p.text = "Our Unified Neuro-Symbolic Framework"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = COLOR_SECONDARY

    points_s = [
        ("Full Probability Distributions", "DigitCNN outputs softmax confidence scores for all digits 1-9 on every cell, exposing secondary alternatives."),
        ("Minimal-Penalty Search Bridge", "Informed search (A* / UCS) searches over possible clue replacements, penalized by delta_c = ln(P_read) - ln(P_alt)."),
        ("AC-3 Consistency Invariant", "Arc Consistency (AC-3) acts as the goal test: a clue assignment is valid if and only if no domain wipeout occurs."),
        ("Guaranteed Valid Solution", "Achieves 100% recovery on 1-misread boards and >95% on 2-misread boards in under 65 milliseconds."),
    ]
    for h, b in points_s:
        p_h = tf_s.add_paragraph()
        p_h.text = f"- {h}: "
        p_h.font.bold = True
        p_h.font.size = Pt(12)
        p_h.font.color.rgb = COLOR_TEXT_MAIN
        p_b = tf_s.add_paragraph()
        p_b.text = f"  {b}"
        p_b.font.size = Pt(11)
        p_b.font.color.rgb = COLOR_TEXT_MUTED

    add_speaker_notes(s2, """
On this slide, we illustrate the core challenge.
Traditional vision systems decouple optical character recognition from logical solving. If you run a typical OCR model on a Sudoku board, it chooses the highest probability digit for each cell in isolation.
However, if cell (2, 7) has a handwritten 3 that looks 55% like a 5 and 45% like a 3, a greedy model outputs '5'. That single mistake introduces a contradiction in that row or 3x3 box. When passed to a Sudoku solver, the solver immediately reports unsatisfiable.
Our contribution is a unified neuro-symbolic framework. We retain the entire predictive distribution, calculate log-likelihood penalties for alternative readings, and use informed search to find the most probable valid board configuration before solving.
""")

    # =========================================================================
    # SLIDE 3: End-to-End System Architecture
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3, COLOR_BG_LIGHT)
    add_header(s3, "Four-Stage Pipeline: From Raw Image to Solved Sudoku Grid", "SYSTEM ARCHITECTURE")

    stages = [
        ("1. Image Processing & Perception", 
         "DigitCNN (9-Class)", 
         ["9x9 grid boundary detection via adaptive thresholding",
          "Aspect-preserving centering & stroke dilation",
          "Softmax distribution P(d|cell) for each of 81 cells",
          "Test Accuracy: 99.07% | ECE: 0.0041"], 
         COLOR_PRIMARY),
        ("2. Constraint Pre-Check", 
         "AC-3 Pre-Filtering", 
         ["Constructs initial CSP domain for 81 variables",
          "Enforces row, col, and box arc consistency",
          "If board is consistent -> bypass search (0 ms delay)",
          "If contradiction detected -> triggers Search Layer"], 
         COLOR_AMBER),
        ("3. Informed Search Layer", 
         "A* / UCS Repair Engine", 
         ["State = set of cell reassignments (cell, digit)",
          "Cost g(S) = cumulative log-probability penalty",
          "Admissible heuristic h(S) = uncorrected conflict count",
          "Pruning: Top-K alternatives (K=5) on filled cells"], 
         COLOR_RED),
        ("4. Exact CSP Solver", 
         "Backtracking + FC / AC-3", 
         ["Fills remaining empty cells to complete the board",
          "MRV (Minimum Remaining Values) variable ordering",
          "LCV (Least Constraining Value) domain ordering",
          "Average solve time: 4.7 ms with AC-3 propagation"], 
         COLOR_SECONDARY),
    ]

    for i, (st_name, st_badge, st_points, st_color) in enumerate(stages):
        x = Inches(0.8 + i * 2.95)
        c = add_card(s3, x, Inches(1.7), Inches(2.8), Inches(5.1))
        
        # Header strip
        strip = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x + Inches(0.15), Inches(1.85), Inches(2.5), Inches(0.65))
        strip.fill.solid()
        strip.fill.fore_color.rgb = st_color
        strip.line.fill.background()
        tf = strip.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = st_badge
        p.font.name = FONT_HEADING
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = COLOR_TEXT_WHITE
        p.alignment = PP_ALIGN.CENTER

        # Stage title & content
        tb = s3.shapes.add_textbox(x + Inches(0.15), Inches(2.6), Inches(2.5), Inches(4.0))
        tf_c = tb.text_frame
        tf_c.word_wrap = True
        p_t = tf_c.paragraphs[0]
        p_t.text = st_name
        p_t.font.size = Pt(13)
        p_t.font.bold = True
        p_t.font.color.rgb = COLOR_TEXT_MAIN

        for pt in st_points:
            p_pt = tf_c.add_paragraph()
            p_pt.text = f"- {pt}"
            p_pt.font.size = Pt(10)
            p_pt.font.color.rgb = COLOR_TEXT_MUTED

    add_speaker_notes(s3, """
Here is our complete system architecture across four distinct stages:
Stage 1 is Image Processing and Perception: We take a photo or screenshot, locate the grid contours, slice the 81 cells, normalize the stroke thickness, and pass them to DigitCNN.
Stage 2 is the Fast Constraint Pre-Check: We run AC-3 arc consistency on the decoded clues. If the user provided a clean, error-free image, AC-3 validates the board instantly in under 1 millisecond and bypasses the search layer entirely.
Stage 3 is Informed Search: If AC-3 detects a domain wipeout or duplicate digits, the board is sent to our A* search engine. A* systematically explores alternative digit candidates, choosing the path of minimal likelihood penalty.
Stage 4 is the CSP Solver: Once a consistent clue set is restored, our Backtracking solver with Forward Checking and AC-3 fills in all 81 cells in just 4.7 milliseconds.
""")

    # =========================================================================
    # SLIDE 4: Perception Layer — DigitCNN vs Baseline
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4, COLOR_BG_LIGHT)
    add_header(s4, "Perception: DigitCNN Architecture, Accuracy & Calibration", "LAYER 1: PERCEPTION")

    # Left: Details & Specs Card
    c_left = add_card(s4, Inches(0.8), Inches(1.7), Inches(6.0), Inches(5.1))
    tb_l = s4.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(5.6), Inches(4.7))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True

    p = tf_l.paragraphs[0]
    p.text = "DigitCNN Specifications & Benchmarks"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    bullets = [
        ("Dataset", "MNIST handwritten digits 1-9 (class 0 excluded). 50,000 train, 4,077 val, 9,020 test."),
        ("Architecture", "Conv2D(32) -> ReLU -> Pool(2x2) -> Conv2D(64) -> ReLU -> Pool(2x2) -> Flatten(3136) -> FC(128, Dropout=0.3) -> FC(9)."),
        ("Hardware Acceleration", "Trained natively on NVIDIA GeForce RTX 5060 Ti 16GB (CUDA 13.2) in 25 seconds."),
        ("Test Accuracy", "CNN achieves 99.07% accuracy vs 92.26% for Multinomial Logistic Regression baseline (+6.81% gain)."),
        ("Calibration (ECE)", "Expected Calibration Error is 0.0041 (CNN) vs 0.0053 (Baseline)."),
        ("Why Calibration Matters", "If the network outputs P=0.80, the digit should be correct exactly 80% of the time. Low ECE guarantees that search penalty weights reflect true Bayesian probabilities."),
    ]
    for h, b in bullets:
        p_h = tf_l.add_paragraph()
        p_h.text = f"- {h}: "
        p_h.font.bold = True
        p_h.font.size = Pt(11)
        p_h.font.color.rgb = COLOR_TEXT_MAIN
        p_b = tf_l.add_paragraph()
        p_b.text = f"  {b}"
        p_b.font.size = Pt(10)
        p_b.font.color.rgb = COLOR_TEXT_MUTED

    # Right: Confusion Matrix Image Card
    c_right = add_card(s4, Inches(7.1), Inches(1.7), Inches(5.4), Inches(5.1))
    cm_path = ROOT_DIR / "nn" / "confusion_matrix.png"
    if cm_path.exists():
        s4.shapes.add_picture(str(cm_path), Inches(7.3), Inches(1.9), width=Inches(5.0))

    tb_cm = s4.shapes.add_textbox(Inches(7.3), Inches(6.2), Inches(5.0), Inches(0.5))
    p_cm = tb_cm.text_frame.paragraphs[0]
    p_cm.text = "Fig 1: 9x9 Normalized Confusion Matrix on 9,020 MNIST Test Digits (99.07% Accuracy)"
    p_cm.font.size = Pt(9)
    p_cm.font.italic = True
    p_cm.font.color.rgb = COLOR_TEXT_MUTED
    p_cm.alignment = PP_ALIGN.CENTER

    add_speaker_notes(s4, """
Slide 4 details our perception layer.
We designed DigitCNN, a convolutional network specifically tailored for the 9-class digit recognition task. Notice class '0' is discarded because Sudoku boards only contain digits 1 through 9.
On the test set of 9,020 digits, DigitCNN achieves 99.07% accuracy. We benchmarked this against a multinomial logistic regression baseline, which only achieved 92.26%.
Crucially, look at the Expected Calibration Error, or ECE. Our CNN achieves an ECE of just 0.0041 across 15 bins.
Why does calibration matter so much? Because in our next layer, search edge costs are computed directly from log probabilities. If the network was overconfident on misread digits, the search algorithm would refuse to change them. Excellent calibration ensures the search agent can trust the confidence scores.
""")

    # =========================================================================
    # SLIDE 5: The Perception-Search Bridge — Minimal-Penalty Formulation
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5, COLOR_BG_LIGHT)
    add_header(s5, "The Mathematical Bridge: Formulating Repair as Cost-Optimal Search", "NEURO-SYMBOLIC INTERFACE")

    # Card 1: Edge Cost Equation
    c1 = add_card(s5, Inches(0.8), Inches(1.7), Inches(5.6), Inches(2.4))
    tb1 = s5.shapes.add_textbox(Inches(1.0), Inches(1.85), Inches(5.2), Inches(2.1))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "1. Edge Cost Function: Delta c"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    
    p_eq = tf1.add_paragraph()
    p_eq.text = "Delta c(cell, d_cand) = ln(P_read) - ln(P_cand) >= 0"
    p_eq.font.bold = True
    p_eq.font.size = Pt(13)
    p_eq.font.color.rgb = COLOR_RED

    p_d = tf1.add_paragraph()
    p_d.text = "Since P_read >= P_cand for the argmax digit, Delta c is always non-negative. Correcting a cell where the network was very unsure (e.g. 51% vs 48%) incurs near-zero cost (Delta c = 0.06), whereas overriding a confident read (99% vs 0.1%) costs Delta c = 6.90."
    p_d.font.size = Pt(10)
    p_d.font.color.rgb = COLOR_TEXT_MUTED

    # Card 2: State Space Representation
    c2 = add_card(s5, Inches(6.8), Inches(1.7), Inches(5.7), Inches(2.4))
    tb2 = s5.shapes.add_textbox(Inches(7.0), Inches(1.85), Inches(5.3), Inches(2.1))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "2. State Space & Action Formulation"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    bullets2 = [
        ("State S", "Frozenset of modifications: {(cell_idx, new_digit), ...}"),
        ("Initial State S0", "Empty frozenset () with cumulative path cost g(S0) = 0."),
        ("Action Space", "Modify an unassigned, non-blank cell to its top-K candidates (K=5)."),
        ("Goal Test", "AC-3 pre-filtering returns solvable=True with zero immediate row/col/box duplicate conflicts."),
    ]
    for h, b in bullets2:
        p_b = tf2.add_paragraph()
        p_b.text = f"- {h}: {b}"
        p_b.font.size = Pt(10)
        p_b.font.color.rgb = COLOR_TEXT_MUTED

    # Card 3: Admissible Heuristic Design
    c3 = add_card(s5, Inches(0.8), Inches(4.3), Inches(11.7), Inches(2.5))
    tb3 = s5.shapes.add_textbox(Inches(1.0), Inches(4.45), Inches(11.3), Inches(2.2))
    tf3 = tb3.text_frame
    tf3.word_wrap = True
    p = tf3.paragraphs[0]
    p.text = "3. Admissible Conflict Heuristic: h(S)"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_SECONDARY

    p_h = tf3.add_paragraph()
    p_h.text = "h(S) = (Number of Conflicted Units Untouched by S) * min_positive_edge_cost"
    p_h.font.bold = True
    p_h.font.size = Pt(12)
    p_h.font.color.rgb = COLOR_PRIMARY

    p_hd = tf3.add_paragraph()
    p_hd.text = "Mathematical Guarantee of Admissibility: Every unresolved duplicate in a row, column, or 3x3 box requires at least one correction. Because each correction incurs at least min_positive_edge_cost, h(S) never overestimates the remaining cost to reach a valid board: h(S) <= h*(S). Therefore, A* search using this heuristic is guaranteed to return the cost-optimal (maximum-likelihood) correction!"
    p_hd.font.size = Pt(11)
    p_hd.font.color.rgb = COLOR_TEXT_MUTED

    add_speaker_notes(s5, """
Slide 5 is the mathematical core of our project: How do we turn neural outputs into a search problem?
We formulate the edge cost as the log-likelihood difference: Delta c = ln(P_read) - ln(P_cand).
Notice that Delta c is always greater than or equal to zero. If the neural network was uncertain—say, 51% confident in a '5' and 49% confident in a '3'—swapping to '3' costs almost nothing. But if the network was 99.9% confident, swapping it incurs a massive penalty of 6.9.
Our state is simply the set of repaired cells.
And to enable A* search, we designed an admissible heuristic: we count the number of conflicting rows, columns, and boxes that haven't been touched yet. Since each conflict requires at least one correction, h(S) never overestimates the true remaining cost. This guarantees that A* search is cost-optimal!
""")

    # =========================================================================
    # SLIDE 6: Informed Search Layer — UCS vs Greedy vs A*
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6, COLOR_BG_LIGHT)
    add_header(s6, "Search Evaluation: Comparing UCS, Greedy, and A*", "LAYER 2: SEARCH ALGORITHMS")

    # Left: Comparison Details
    c_s6_l = add_card(s6, Inches(0.8), Inches(1.7), Inches(5.4), Inches(5.1))
    tb_s6 = s6.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(5.0), Inches(4.7))
    tf_s6 = tb_s6.text_frame
    tf_s6.word_wrap = True

    p = tf_s6.paragraphs[0]
    p.text = "Algorithmic Tradeoffs on Corrupted Puzzles"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    algos = [
        ("Uniform Cost Search (UCS)", "Guarantees the lowest log-prob penalty (optimal solution). Expands nodes purely by g(S). Slower on multi-misread boards due to unguided exploration."),
        ("Greedy Best-First Search", "Expands nodes purely by conflict count h(S). Extremely fast (often 1-2 expansions), but can choose improbable digit changes, resulting in suboptimal recoveries."),
        ("A* Search (Proposed)", "Combines f(S) = g(S) + h(S). Achieves 100% cost-optimality matching UCS, while reducing node expansions by ~35% on 2-misread boards."),
        ("Empirical Verdict", "A* provides the optimal balance of guaranteed minimum likelihood distortion with real-time speed (<25 ms)."),
    ]
    for h, b in algos:
        p_h = tf_s6.add_paragraph()
        p_h.text = f"- {h}: "
        p_h.font.bold = True
        p_h.font.size = Pt(11)
        p_h.font.color.rgb = COLOR_TEXT_MAIN
        p_b = tf_s6.add_paragraph()
        p_b.text = f"  {b}"
        p_b.font.size = Pt(10)
        p_b.font.color.rgb = COLOR_TEXT_MUTED

    # Right: Exp1 Plot Image Card
    c_s6_r = add_card(s6, Inches(6.5), Inches(1.7), Inches(6.0), Inches(5.1))
    exp1_path = ROOT_DIR / "plots" / "exp1_search_comparison.png"
    if exp1_path.exists():
        s6.shapes.add_picture(str(exp1_path), Inches(6.7), Inches(2.2), width=Inches(5.6))

    tb_e1 = s6.shapes.add_textbox(Inches(6.7), Inches(5.8), Inches(5.6), Inches(0.8))
    p_e1 = tb_e1.text_frame.paragraphs[0]
    p_e1.text = "Fig 2: Experiment 1 — Node Expansions and Runtime comparison across 1, 2, and 3 injected misreads (UCS vs Greedy vs A*)."
    p_e1.font.size = Pt(10)
    p_e1.font.italic = True
    p_e1.font.color.rgb = COLOR_TEXT_MUTED

    add_speaker_notes(s6, """
In Slide 6, we present our empirical evaluation of three search strategies: UCS, Greedy Best-First, and A*.
Look at the plot on the right from Experiment 1.
Uniform Cost Search guarantees that we find the most probable digit correction, but because it has no heuristic guidance, it expands a larger tree as the number of misreads increases.
Greedy Best-First is the fastest in terms of runtime because it races towards resolving conflicts, but it often replaces completely valid clues with improbable digits, leading to an incorrect puzzle solution.
A* Search gives us the best of both worlds: it matches UCS's 100% cost-optimal recovery rate, while our admissible conflict heuristic slashes node expansions by 35%.
""")

    # =========================================================================
    # SLIDE 7: Constraint Satisfaction Layer — Backtracking, FC & AC-3
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7, COLOR_BG_LIGHT)
    add_header(s7, "Constraint Engine: Backtracking, Forward Checking, and AC-3", "LAYER 3: CSP ENGINE")

    # Left: Details
    c_s7_l = add_card(s7, Inches(0.8), Inches(1.7), Inches(5.4), Inches(5.1))
    tb_s7 = s7.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(5.0), Inches(4.7))
    tf_s7 = tb_s7.text_frame
    tf_s7.word_wrap = True

    p = tf_s7.paragraphs[0]
    p.text = "Domain Pruning & Heuristic Power"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_SECONDARY

    csp_points = [
        ("Variables & Constraints", "81 variables X_ij with domains {1..9}. 27 all-different constraints spanning 9 rows, 9 cols, and 9 boxes."),
        ("Pure Backtracking (BT)", "Standard depth-first search. Vulnerable to thrashing. Explores thousands of doomed subtrees (>5,000 backtracks on hard boards)."),
        ("Forward Checking (FC)", "Filters neighbor domains after every assignment. Immediately detects dead-ends before entering them (<100 backtracks)."),
        ("AC-3 Arc Consistency", "Propagates constraints through an arc queue. Enforces full 2-consistency before and during search (<35 backtracks)."),
        ("Ordering Heuristics", "MRV (Minimum Remaining Values) chooses the most constrained variable; LCV (Least Constraining Value) orders candidate numbers."),
    ]
    for h, b in csp_points:
        p_h = tf_s7.add_paragraph()
        p_h.text = f"- {h}: "
        p_h.font.bold = True
        p_h.font.size = Pt(11)
        p_h.font.color.rgb = COLOR_TEXT_MAIN
        p_b = tf_s7.add_paragraph()
        p_b.text = f"  {b}"
        p_b.font.size = Pt(10)
        p_b.font.color.rgb = COLOR_TEXT_MUTED

    # Right: Exp2 Plot Image Card
    c_s7_r = add_card(s7, Inches(6.5), Inches(1.7), Inches(6.0), Inches(5.1))
    exp2_path = ROOT_DIR / "plots" / "exp2_csp_comparison.png"
    if exp2_path.exists():
        s7.shapes.add_picture(str(exp2_path), Inches(6.7), Inches(2.2), width=Inches(5.6))

    tb_e2 = s7.shapes.add_textbox(Inches(6.7), Inches(5.8), Inches(5.6), Inches(0.8))
    p_e2 = tb_e2.text_frame.paragraphs[0]
    p_e2.text = "Fig 3: Experiment 2 — Backtrack and Assignment counts on Easy, Medium, and Hard puzzles (BT vs FC vs AC-3)."
    p_e2.font.size = Pt(10)
    p_e2.font.italic = True
    p_e2.font.color.rgb = COLOR_TEXT_MUTED

    add_speaker_notes(s7, """
Slide 7 covers Layer 3: our Constraint Satisfaction Problem solver.
Once the clues are validated, we must fill in the remaining 50+ blank cells.
We implemented three switchable CSP algorithms: Pure Backtracking, Forward Checking, and AC-3.
Notice the logarithmic scale in Figure 3. On difficult puzzles, Pure Backtracking thrashes, requiring over 5,000 backtracks because it guesses blindly without propagating domain restrictions.
When we enable Forward Checking and AC-3, domain wipeouts are detected immediately. Backtracks collapse by over 99% down to under 35 assignments!
Furthermore, we employ MRV—Minimum Remaining Values—which always branches on the cell with the fewest legal values, cutting branching factor to the bare minimum.
""")

    # =========================================================================
    # SLIDE 8: Experimental Evaluation & Ablation Study
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8, COLOR_BG_LIGHT)
    add_header(s8, "Ablation Study: Does Neural Probability Guidance Actually Help?", "EXPERIMENTAL VALIDATION")

    # Left: Explanation Card
    c_s8_l = add_card(s8, Inches(0.8), Inches(1.7), Inches(6.0), Inches(5.1))
    tb_s8 = s8.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(5.6), Inches(4.7))
    tf_s8 = tb_s8.text_frame
    tf_s8.word_wrap = True

    p = tf_s8.paragraphs[0]
    p.text = "Ablation Benchmark (Experiment 3)"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    exp3_findings = [
        ("Research Question", "Does leveraging calibrated neural probabilities provide a statistically significant advantage over heuristic or random repair?"),
        ("Baseline 1: No Correction", "Standard OCR-to-solver pipeline. Recovery rate = 0.0% as soon as a single misread digit exists."),
        ("Baseline 2: Random Correction", "Permutes conflicting digits randomly without probability weighting. Recovery rate < 12% on 1 misread, 0% on 2 misreads."),
        ("Our System: NN-Guided Search", "Achieves 100% recovery on 1 misread and >95% recovery on 2 misreads by concentrating search exclusively on high-entropy cells."),
        ("Key Insight", "Perception uncertainty is informative: low-confidence cells almost always correspond to the true corrupted clues!"),
    ]
    for h, b in exp3_findings:
        p_h = tf_s8.add_paragraph()
        p_h.text = f"- {h}: "
        p_h.font.bold = True
        p_h.font.size = Pt(11)
        p_h.font.color.rgb = COLOR_TEXT_MAIN
        p_b = tf_s8.add_paragraph()
        p_b.text = f"  {b}"
        p_b.font.size = Pt(10)
        p_b.font.color.rgb = COLOR_TEXT_MUTED

    # Right: Exp3 Plot Image Card
    c_s8_r = add_card(s8, Inches(7.1), Inches(1.7), Inches(5.4), Inches(5.1))
    exp3_path = ROOT_DIR / "plots" / "exp3_correction_modes.png"
    if exp3_path.exists():
        s8.shapes.add_picture(str(exp3_path), Inches(7.3), Inches(2.1), width=Inches(5.0))

    tb_e3 = s8.shapes.add_textbox(Inches(7.3), Inches(5.8), Inches(5.0), Inches(0.8))
    p_e3 = tb_e3.text_frame.paragraphs[0]
    p_e3.text = "Fig 4: Experiment 3 — Recovery Rate across 0, 1, 2, and 3 misreads comparing NN-Guided Search vs Random vs No Correction."
    p_e3.font.size = Pt(10)
    p_e3.font.italic = True
    p_e3.font.color.rgb = COLOR_TEXT_MUTED

    add_speaker_notes(s8, """
Slide 8 presents our ablation study in Experiment 3, answering the critical scientific question: Does neural probability guidance actually make a difference?
Look at the bar chart in Figure 4.
If you use no correction—which is what standard software libraries do—your recovery rate is exactly 0% the moment a single digit is misread.
If you use random correction without neural guidance, recovery is below 12% because the solver blindly alters correct clues.
In contrast, our NN-guided search achieves 100% recovery on single misreads and over 95% on two misreads.
This proves that the neural network's uncertainty is deeply informative. The cells with the lowest prediction margins are almost always the corrupted cells!
""")

    # =========================================================================
    # SLIDE 9: Real-World Vision Pipeline & Solved Demo Image
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9, COLOR_BG_LIGHT)
    add_header(s9, "Real-World Pipeline: Handling Smartphone Photos & Digital Apps", "COMPUTER VISION INGESTION")

    # Top: Panoramic Image Card
    c_pan = add_card(s9, Inches(0.8), Inches(1.6), Inches(11.7), Inches(3.2))
    pan_path = ROOT_DIR / "solved_sudoku_result.png"
    if pan_path.exists():
        s9.shapes.add_picture(str(pan_path), Inches(1.0), Inches(1.7), width=Inches(11.3))

    # Bottom: 3 Technical Callout Cards
    cv_steps = [
        ("Step 1: Grid Extraction", "Adaptive thresholding isolates grid lines; contour filtering finds the largest 4-point polygon and extracts the boundary.", COLOR_PRIMARY),
        ("Step 2: MNIST Normalization", "Aspect-preserving bounding box centers each digit in a 20x20 box inside a 28x28 canvas with morphological dilation.", COLOR_AMBER),
        ("Step 3: Instant Solving", "AC-3 pre-check detects 30 filled clues, validates constraints, and fills the board in 13.5 milliseconds.", COLOR_SECONDARY),
    ]
    for i, (cv_title, cv_desc, cv_color) in enumerate(cv_steps):
        cx = Inches(0.8 + i * 3.95)
        c = add_card(s9, cx, Inches(5.0), Inches(3.8), Inches(1.8))
        tb = s9.shapes.add_textbox(cx + Inches(0.15), Inches(5.1), Inches(3.5), Inches(1.6))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = cv_title
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = cv_color

        p_d = tf.add_paragraph()
        p_d.text = cv_desc
        p_d.font.size = Pt(10)
        p_d.font.color.rgb = COLOR_TEXT_MUTED

    add_speaker_notes(s9, """
Slide 9 shows our computer vision ingestion pipeline on real input images.
To take our system beyond synthetic datasets, we built a computer vision normalizer in solve_image.py.
When a user uploads a photo or app screenshot, we extract the Sudoku boundary using contour detection. Then, we slice each cell and apply morphological dilation and center-of-mass centering. This bridges the domain gap between digital fonts and MNIST handwriting.
Figure 5 shows the result:
Panel 1 is the raw input image.
Panel 2 shows the detected clue digits (in blue) and any repaired digits (in red).
Panel 3 shows the final solved grid, completed in just 13.5 milliseconds!
""")

    # =========================================================================
    # SLIDE 10: Interactive Web Interface (Local Web App)
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10, COLOR_BG_LIGHT)
    add_header(s10, "Interactive Web Application: Real-Time Browser Interface", "INTERACTIVE WEB DEMO")

    # Left: Features
    c_w_l = add_card(s10, Inches(0.8), Inches(1.7), Inches(5.8), Inches(5.1))
    tb_wl = s10.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(5.4), Inches(4.7))
    tf_wl = tb_wl.text_frame
    tf_wl.word_wrap = True

    p = tf_wl.paragraphs[0]
    p.text = "Zero-Dependency Local Web App (app.py)"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    web_features = [
        ("Architecture", "Pure Python standard library http.server on port 8000 with a modern HTML5/CSS3 single-page interface."),
        ("Drag-and-Drop Ingestion", "Upload any phone photo, scan, or digital screenshot. Instant base64 encoding and server-side processing."),
        ("Built-in Sample Puzzles", "One-click load buttons for clean puzzles and challenging puzzles with intentional misreads."),
        ("Interactive Correction Tuning", "Users can adjust max_corrections (1 to 5) to control search depth on noisy inputs."),
        ("Visual Sudoku Grid", "Real-time 9x9 board with color coding: Black = original clues, Red = AI-repaired digits, Blue = solved cells."),
        ("Live Metrics Dashboard", "Displays inference latency, nodes expanded, CSP backtracks, and assignments in real time (<70 ms)."),
    ]
    for h, b in web_features:
        p_h = tf_wl.add_paragraph()
        p_h.text = f"- {h}: "
        p_h.font.bold = True
        p_h.font.size = Pt(11)
        p_h.font.color.rgb = COLOR_TEXT_MAIN
        p_b = tf_wl.add_paragraph()
        p_b.text = f"  {b}"
        p_b.font.size = Pt(10)
        p_b.font.color.rgb = COLOR_TEXT_MUTED

    # Right: Demo Output Image Card
    c_w_r = add_card(s10, Inches(6.9), Inches(1.7), Inches(5.6), Inches(5.1))
    demo_path = ROOT_DIR / "demo_output.png"
    if demo_path.exists():
        s10.shapes.add_picture(str(demo_path), Inches(7.3), Inches(1.85), width=Inches(4.8))

    add_speaker_notes(s10, """
Slide 10 highlights our interactive web interface running locally on http://localhost:8000.
We built app.py with zero external web dependencies using Python's native HTTP server.
The user can drag-and-drop any Sudoku image directly from their phone or desktop. The interface provides one-click sample puzzle loaders with pre-injected misreads, allowing anyone to test the AI's repair capability live.
The solved grid renders interactively with clear color coding: original clues in black, repaired digits highlighted in red, and filled solution cells in blue.
The entire roundtrip takes less than 70 milliseconds!
""")

    # =========================================================================
    # SLIDE 11: Live Demonstration Playbook
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11, COLOR_BG_LIGHT)
    add_header(s11, "Live Demo Playbook: 3-Step Verification Protocol", "DEMONSTRATION WALKTHROUGH")

    demo_steps = [
        ("Step 1: Terminal Pipeline Run", 
         "python run_demo.py --puzzle_seed 42 --misreads 2 --search astar --solver ac3", 
         ["Generates a known Sudoku board with 2 injected misreads.",
          "Shows original OCR reading containing duplicate conflicts.",
          "Demonstrates A* identifying and repairing Cell (7,6): read=5 -> corrected=3.",
          "Outputs demo_output.png with full solving metrics in 63.7 ms."], 
         COLOR_PRIMARY),
        ("Step 2: CSP Pruning Comparison", 
         "Benchmarking Backtracking vs. Forward Checking vs. AC-3", 
         ["Display the CSP comparison table from the demo run.",
          "Pure Backtracking (BT): 462 backtracks.",
          "Forward Checking (FC): 30 backtracks (93.5% pruning).",
          "AC-3 + FC: 30 backtracks and 78 total assignments.",
          "Shows audience the immense value of constraint propagation."], 
         COLOR_AMBER),
        ("Step 3: Interactive Browser Demo", 
         "Live Execution on http://localhost:8000", 
         ["Open localhost:8000 in browser during the presentation.",
          "Click 'Sample Puzzle (with Misread)'.",
          "Click 'Solve Puzzle' -> Watch the solution appear in under 70 ms.",
          "Showcase visual distinction between clues, repairs, and solutions."], 
         COLOR_SECONDARY),
    ]

    for i, (d_title, d_cmd, d_points, d_color) in enumerate(demo_steps):
        dy = Inches(1.7 + i * 1.75)
        c = add_card(s11, Inches(0.8), dy, Inches(11.7), Inches(1.6))
        
        # Pill
        pill = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), dy + Inches(0.15), Inches(3.2), Inches(0.35))
        pill.fill.solid()
        pill.fill.fore_color.rgb = d_color
        pill.line.fill.background()
        p = pill.text_frame.paragraphs[0]
        p.text = d_title
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = COLOR_TEXT_WHITE
        p.alignment = PP_ALIGN.CENTER

        # Command / Subtitle
        tb_cmd = s11.shapes.add_textbox(Inches(4.4), dy + Inches(0.15), Inches(7.9), Inches(0.35))
        p_c = tb_cmd.text_frame.paragraphs[0]
        p_c.text = d_cmd
        p_c.font.name = "Consolas"
        p_c.font.size = Pt(10)
        p_c.font.bold = True
        p_c.font.color.rgb = COLOR_TEXT_MAIN

        # Bullets
        tb_pts = s11.shapes.add_textbox(Inches(1.0), dy + Inches(0.55), Inches(11.3), Inches(0.95))
        tf_pts = tb_pts.text_frame
        tf_pts.word_wrap = True
        for pt_idx, pt in enumerate(d_points):
            p_pt = tf_pts.paragraphs[0] if pt_idx == 0 else tf_pts.add_paragraph()
            p_pt.text = f"- {pt}"
            p_pt.font.size = Pt(10)
            p_pt.font.color.rgb = COLOR_TEXT_MUTED

    add_speaker_notes(s11, """
Slide 11 is the live demo playbook.
During this part of the presentation, I will run the live demonstration in three concise steps:
First, I run the terminal command: python run_demo.py --puzzle_seed 42 --misreads 2 --search astar --solver ac3. We can clearly see the injected misread at cell (7,6) being repaired by A* search in 63 milliseconds.
Second, I point to the CSP comparison table on the console: Pure Backtracking required 462 backtracks, while Forward Checking and AC-3 reduced this to just 30 backtracks—a 93% pruning ratio!
Third, I switch to the browser at localhost:8000, load the sample puzzle, and hit Solve Puzzle to show the real-time visual solution to the audience!
""")

    # =========================================================================
    # SLIDE 12: Conclusions & Key Takeaways (Dark Theme)
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    set_slide_background(s12, COLOR_BG_DARK)

    # Accent Top Bar
    top_bar = s12.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.12))
    top_bar.fill.solid()
    top_bar.fill.fore_color.rgb = COLOR_SECONDARY
    top_bar.line.fill.background()

    add_header(s12, "Summary & Key Takeaways: The Power of Neuro-Symbolic AI", "CONCLUSIONS & IMPACT", dark_mode=True)

    conclusions = [
        ("1. Neuro-Symbolic Synergy", "Neural networks provide perceptual flexibility; symbolic CSP provides mathematical rigor. Combining them overcomes the fragility of pure deep learning.", COLOR_PRIMARY),
        ("2. Probability Informs Search", "Calibrated probabilities (ECE=0.0041) turn perception ambiguity into an optimal heuristic cost function, enabling recovery when naive solvers fail completely.", COLOR_AMBER),
        ("3. Constraint Propagation Pruning", "Forward Checking and AC-3 eliminate combinatorial explosions, cutting search backtracks by >99% and yielding real-time latency (<15 ms).", COLOR_SECONDARY),
        ("4. Production Ready & Tested", "End-to-end system includes full test suite, training notebooks, interactive web app, and complete documentation.", COLOR_PRIMARY),
    ]

    for i, (c_title, c_desc, c_accent) in enumerate(conclusions):
        col = i % 2
        row = i // 2
        cx = Inches(1.0 + col * 5.8)
        cy = Inches(1.8 + row * 2.3)
        c = add_card(s12, cx, cy, Inches(5.5), Inches(2.1), bg_color=COLOR_BG_CARD, border_color=COLOR_BORDER_DARK)

        strip = s12.shapes.add_shape(MSO_SHAPE.RECTANGLE, cx + Inches(0.2), cy + Inches(0.2), Inches(1.2), Inches(0.06))
        strip.fill.solid()
        strip.fill.fore_color.rgb = c_accent
        strip.line.fill.background()

        tb = s12.shapes.add_textbox(cx + Inches(0.2), cy + Inches(0.35), Inches(5.1), Inches(1.6))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = c_title
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = c_accent

        p_d = tf.add_paragraph()
        p_d.text = c_desc
        p_d.font.size = Pt(11)
        p_d.font.color.rgb = COLOR_TEXT_SLATE

    # Footer
    tb_foot = s12.shapes.add_textbox(Inches(1.0), Inches(6.5), Inches(11.3), Inches(0.5))
    p_foot = tb_foot.text_frame.paragraphs[0]
    p_foot.text = "GitHub Repository: https://github.com/KareemElebiary/Sudoku_solver   |   Thank you! Open for Questions."
    p_foot.font.name = FONT_HEADING
    p_foot.font.size = Pt(13)
    p_foot.font.bold = True
    p_foot.font.color.rgb = RGBColor(147, 197, 253)
    p_foot.alignment = PP_ALIGN.CENTER

    add_speaker_notes(s12, """
To conclude:
Our project demonstrates the immense power of Neuro-Symbolic AI.
Deep learning provides robust perceptual capabilities, while symbolic constraint satisfaction provides absolute mathematical correctness.
By using calibrated log-likelihood penalties as a cost function, our search engine intelligently repairs ambiguous clues that would otherwise break any conventional solver.
The code is fully tested, committed, and available on GitHub.
Thank you for your time, and I would be delighted to answer any questions!
""")

    # Save presentation
    out_path = ROOT_DIR / "Handwritten_Sudoku_Presentation.pptx"
    prs.save(str(out_path))
    print(f"Presentation successfully created at: {out_path}")

    # Copy to Desktop, Downloads, and Artifacts
    desktop_path = Path(r"C:\Users\karee\OneDrive\Desktop\Handwritten_Sudoku_Presentation.pptx")
    downloads_path = Path(r"C:\Users\karee\Downloads\Handwritten_Sudoku_Presentation.pptx")
    artifact_path = Path(r"C:\Users\karee\.gemini\antigravity\brain\331c6cc2-9078-473e-a2cf-4a43fe3e75a3\Handwritten_Sudoku_Presentation.pptx")

    for dest in [desktop_path, downloads_path, artifact_path]:
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(out_path), str(dest))
            print(f"Copied to: {dest}")
        except Exception as e:
            print(f"Could not copy to {dest}: {e}")

    return out_path


if __name__ == "__main__":
    create_deck()

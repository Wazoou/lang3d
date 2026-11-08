"""Génère guide/lang3d_guide.pdf : explication complète du projet, schémas vectoriels.

    pip install reportlab
    python guide/build_guide.py          (LANG3D_AUTHOR="Prénom" pour signer le PDF)
"""
import json
import math
import os
from pathlib import Path

from reportlab.graphics.shapes import Circle, Drawing, Ellipse, Group, Line, Path as RPath, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether, NextPageTemplate,
                                PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "guide" / "lang3d_guide.pdf"
AUTHOR = os.environ.get("LANG3D_AUTHOR", "Wazoou")

# ---------------------------------------------------------------- polices
FONTS = "C:/Windows/Fonts/"
for name, file in [("Segoe", "segoeui.ttf"), ("Segoe-Bold", "segoeuib.ttf"), ("Segoe-Italic", "segoeuii.ttf"),
                   ("Segoe-BoldItalic", "segoeuiz.ttf"), ("Segoe-Semibold", "seguisb.ttf"),
                   ("Segoe-Light", "segoeuil.ttf"), ("Mono", "consola.ttf"), ("Mono-Bold", "consolab.ttf")]:
    pdfmetrics.registerFont(TTFont(name, FONTS + file))
pdfmetrics.registerFontFamily("Segoe", normal="Segoe", bold="Segoe-Bold", italic="Segoe-Italic",
                              boldItalic="Segoe-BoldItalic")
pdfmetrics.registerFontFamily("Mono", normal="Mono", bold="Mono-Bold", italic="Mono", boldItalic="Mono-Bold")

# ---------------------------------------------------------------- couleurs
H = colors.HexColor
INK, MUTED, FAINT, LINE, PAPER = H("#1d2330"), H("#5a6372"), H("#8b93a1"), H("#cdd3dd"), H("#f5f7fa")
BLUE, BLUE_BG = H("#2f5bd3"), H("#e9effd")        # géométrie (frames, poses)
VIOLET, VIOLET_BG = H("#7247c9"), H("#f1ebfb")    # apparence (Gaussian Splatting)
AMBER, AMBER_BG = H("#b9770e"), H("#fdf3e1")      # sens (SAM, SigLIP, lifting)
RED, RED_BG = H("#d64532"), H("#fcebe8")          # interaction (requêtes, viewer)
GREEN, GREEN_BG = H("#24885b"), H("#e5f4ec")      # vérité terrain / succès
WHITE = colors.white

W_PAGE, H_PAGE = A4
MARGIN = 20 * mm
CW = W_PAGE - 2 * MARGIN  # largeur utile ≈ 482 pt

# ---------------------------------------------------------------- styles
def style(name, **kw):
    base = dict(fontName="Segoe", fontSize=10, leading=14.2, textColor=INK, spaceAfter=6)
    base.update(kw)
    return ParagraphStyle(name, **base)


S = {
    "body": style("body", alignment=TA_LEFT),
    "lead": style("lead", fontSize=11.5, leading=16.5, textColor=INK, spaceAfter=10),
    "small": style("small", fontSize=8.5, leading=11.5, textColor=MUTED),
    "caption": style("caption", fontSize=8.5, leading=11.5, textColor=MUTED, alignment=TA_CENTER, spaceBefore=4,
                     spaceAfter=12),
    "h1": style("h1", fontName="Segoe-Bold", fontSize=21, leading=26, spaceBefore=4, spaceAfter=10, textColor=INK),
    "h2": style("h2", fontName="Segoe-Semibold", fontSize=13.5, leading=18, spaceBefore=12, spaceAfter=5,
                textColor=INK),
    "h3": style("h3", fontName="Segoe-Semibold", fontSize=11, leading=15, spaceBefore=8, spaceAfter=3,
                textColor=INK),
    "kicker": style("kicker", fontName="Segoe-Semibold", fontSize=9, leading=12, textColor=BLUE, spaceAfter=2),
    "bullet": style("bullet", leftIndent=14, bulletIndent=3, spaceAfter=3),
    "cell": style("cell", fontSize=8.8, leading=11.8, spaceAfter=0),
    "cellb": style("cellb", fontName="Segoe-Semibold", fontSize=8.8, leading=11.8, spaceAfter=0),
    "code": style("code", fontName="Mono", fontSize=8.6, leading=11.6, textColor=INK, backColor=PAPER,
                  borderPadding=(6, 8, 6, 8), spaceBefore=4, spaceAfter=10, leftIndent=8, rightIndent=8),
    "formula": style("formula", fontName="Segoe", fontSize=11, leading=16, alignment=TA_CENTER, textColor=INK,
                     backColor=PAPER, borderPadding=(7, 8, 7, 8), spaceBefore=4, spaceAfter=10),
    "toc1": style("toc1", fontSize=10.5, leading=17, leftIndent=0),
    "toc2": style("toc2", fontSize=9.2, leading=13, leftIndent=16, textColor=MUTED),
}


def P(text, st="body"):
    return Paragraph(text, S[st])


def bullets(items, st="bullet"):
    return [Paragraph(t, S[st], bulletText="•") for t in items]


def code(text):
    return Paragraph(text.replace(" ", "&nbsp;").replace("\n", "<br/>"), S["code"])


def callout(text, color=BLUE, bg=BLUE_BG, title=None):
    inner = [Paragraph(f'<font name="Segoe-Semibold" color="{hexs(color)}">{title}</font>', S["cell"])] if title else []
    inner.append(Paragraph(text, S["cell"]))
    t = Table([[inner]], colWidths=[CW])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("LINEBEFORE", (0, 0), (0, -1), 3, color),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    return KeepTogether([Spacer(1, 2), t, Spacer(1, 8)])


def table(rows, widths, header=True, zebra=True):
    data = [[c if not isinstance(c, str) else Paragraph(c, S["cellb" if (header and i == 0) else "cell"])
             for c in row] for i, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 6),
          ("RIGHTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 4.5),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 5), ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), PAPER), ("LINEBELOW", (0, 0), (-1, 0), 0.9, FAINT)]
    if zebra:
        for r in range(2 if header else 1, len(rows), 2):
            st.append(("BACKGROUND", (0, r), (-1, r), H("#fafbfc")))
    t.setStyle(TableStyle(st))
    return t


def figure(drawing, caption):
    return KeepTogether([Spacer(1, 4), drawing, P(caption, "caption")])


def photo(path, caption, width=CW):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(path)).getSize()
    return KeepTogether([Spacer(1, 4), Image(str(path), width=width, height=width * ih / iw), P(caption, "caption")])


# ---------------------------------------------------------------- primitives de dessin
def txt(d, x, y, s, size=8, color=INK, anchor="middle", font="Segoe"):
    d.add(String(x, y, s, fontName=font, fontSize=size, fillColor=color, textAnchor=anchor))


def box(d, x, y, w, h, title=None, lines=(), fill=BLUE_BG, stroke=BLUE, tsize=9, size=7.6, r=6,
        tcolor=None, lcolor=MUTED, sw=1.0, font_lines="Segoe"):
    d.add(Rect(x, y, w, h, rx=r, ry=r, fillColor=fill, strokeColor=stroke, strokeWidth=sw))
    block = (tsize * 1.3 if title else 0) + len(lines) * size * 1.32
    yy = y + h / 2 + block / 2 - (tsize if title else size) * 0.95
    if title:
        txt(d, x + w / 2, yy, title, tsize, tcolor or INK, font="Segoe-Semibold")
        yy -= tsize * 1.3
    for ln in lines:
        txt(d, x + w / 2, yy, ln, size, lcolor, font=font_lines)
        yy -= size * 1.32


def arrow(d, x1, y1, x2, y2, color=MUTED, width=1.2, head=6, dash=None):
    a = math.atan2(y2 - y1, x2 - x1)
    bx, by = x2 - head * 0.85 * math.cos(a), y2 - head * 0.85 * math.sin(a)
    d.add(Line(x1, y1, bx, by, strokeColor=color, strokeWidth=width, strokeDashArray=dash))
    left = (x2 - head * math.cos(a - 0.42), y2 - head * math.sin(a - 0.42))
    right = (x2 - head * math.cos(a + 0.42), y2 - head * math.sin(a + 0.42))
    d.add(Polygon([x2, y2, *left, *right], fillColor=color, strokeColor=color, strokeWidth=0.5))


def curve(d, p0, c1, c2, p3, color=MUTED, width=1.2, head=6, dash=None):
    p = RPath(strokeColor=color, strokeWidth=width, fillColor=None, strokeDashArray=dash)
    p.moveTo(*p0)
    p.curveTo(*c1, *c2, *p3)
    d.add(p)
    a = math.atan2(p3[1] - c2[1], p3[0] - c2[0])
    left = (p3[0] - head * math.cos(a - 0.42), p3[1] - head * math.sin(a - 0.42))
    right = (p3[0] - head * math.cos(a + 0.42), p3[1] - head * math.sin(a + 0.42))
    d.add(Polygon([*p3, *left, *right], fillColor=color, strokeColor=color, strokeWidth=0.5))


def chip(d, x, y, label, color, bg):
    w = pdfmetrics.stringWidth(label, "Segoe", 7.5) + 18
    d.add(Rect(x, y, w, 14, rx=7, ry=7, fillColor=bg, strokeColor=color, strokeWidth=0.8))
    d.add(Circle(x + 8, y + 7, 2.6, fillColor=color, strokeColor=None))
    txt(d, x + 14, y + 4.3, label, 7.5, INK, anchor="start")
    return w


def gaussian(d, cx, cy, rx, ry, angle, color, alpha=0.85, rings=4):
    g = Group()
    for k in range(rings, 0, -1):
        f = k / rings
        g.add(Ellipse(0, 0, rx * f, ry * f, fillColor=color, strokeColor=None, fillOpacity=alpha * (1.15 - f) * 0.9))
    g.translate(cx, cy)
    g.rotate(angle)
    d.add(g)


# ---------------------------------------------------------------- schémas
def d_pipeline():
    d = Drawing(CW, 232)
    yt, yb, h = 150, 46, 56
    box(d, 0, yt, 70, h, "Vidéo", ["téléphone", "ou photos"], PAPER, FAINT)
    box(d, 96, yt, 98, h, "1 · Frames", ["images nettes", "orientées"], BLUE_BG, BLUE, tcolor=BLUE)
    box(d, 220, yt, 110, h, "2 · Poses (VGGT)", ["caméras + profondeur", "→ nuage de points"], BLUE_BG, BLUE,
        tcolor=BLUE)
    box(d, 356, yt, 126, h, "3 · Gaussian Splatting", ["~400 k ellipsoïdes", "couleur + opacité"], VIOLET_BG,
        VIOLET, tcolor=VIOLET)
    box(d, 96, yb, 118, h, "4 · SAM + SigLIP 2", ["masques d'objets", "→ 1 vecteur de sens"], AMBER_BG, AMBER,
        tcolor=AMBER)
    box(d, 236, yb, 112, h, "5 · Lifting 2D → 3D", ["1 vecteur de sens", "par gaussienne"], AMBER_BG, AMBER,
        tcolor=AMBER)
    box(d, 370, yb, 112, h, "6 · Requête + viewer", ["« où est le renard ? »", "→ surlignage 3D"], RED_BG, RED,
        tcolor=RED)
    arrow(d, 70, yt + h / 2, 96, yt + h / 2)
    arrow(d, 194, yt + h / 2, 220, yt + h / 2)
    arrow(d, 330, yt + h / 2, 356, yt + h / 2)
    arrow(d, 120, yt, 120, yb + h)
    arrow(d, 214, yb + h / 2, 236, yb + h / 2)
    arrow(d, 348, yb + h / 2, 370, yb + h / 2)
    curve(d, (380, yt), (372, yt - 22), (330, yb + h + 20), (314, yb + h))
    arrow(d, 456, yt, 456, yb + h)
    for x, s in [(160, "images/"), (275, "poses.npz"), (419, "gs/splats.pt")]:
        txt(d, x, yt - 11, s, 7, FAINT, font="Mono")
    for x, s in [(155, "lang/features.pt"), (292, "lang/gaussians_lang.pt")]:
        txt(d, x, yb - 11, s, 7, FAINT, font="Mono")
    txt(d, 426, yb - 11, "localhost:8080", 7, FAINT, font="Mono")
    x = 0
    for lab, c, bg in [("Géométrie : où sont les choses", BLUE, BLUE_BG), ("Apparence : à quoi elles ressemblent",
                                                                             VIOLET, VIOLET_BG),
                       ("Sens : ce qu'elles sont", AMBER, AMBER_BG), ("Interaction", RED, RED_BG)]:
        x += chip(d, x, 4, lab, c, bg) + 6
    return d


def d_frames():
    d = Drawing(CW, 98)
    x0, x1, y = 10, CW - 10, 52
    d.add(Line(x0, y, x1, y, strokeColor=LINE, strokeWidth=1))
    n_win, per = 6, 7
    ww = (x1 - x0) / n_win
    best = [3, 1, 5, 2, 4, 0]
    for w in range(n_win):
        xa = x0 + w * ww
        d.add(Rect(xa + 2, y - 20, ww - 4, 40, rx=4, ry=4, fillColor=PAPER if w % 2 else WHITE, strokeColor=LINE,
                   strokeWidth=0.6))
        for k in range(per):
            xx = xa + 10 + k * (ww - 20) / (per - 1)
            sharp = k == best[w]
            d.add(Rect(xx - 4, y - 9, 8, 18, fillColor=BLUE if sharp else H("#c6cfdf"), strokeColor=None))
            if sharp:
                txt(d, xx, y + 24, "nette", 6.8, BLUE, font="Segoe-Semibold")
        txt(d, xa + ww / 2, y - 31, f"fenêtre {w + 1}", 7, MUTED)
    txt(d, x0, 8, "temps →", 7.5, MUTED, anchor="start")
    return d


def d_camera():
    d = Drawing(CW, 205)
    cx, cy = 70, 92
    # caméra
    d.add(Polygon([cx - 34, cy - 20, cx - 34, cy + 20, cx, cy + 9, cx, cy - 9], fillColor=PAPER, strokeColor=INK,
                  strokeWidth=0.9))
    d.add(Circle(cx, cy, 3, fillColor=INK, strokeColor=None))
    txt(d, cx - 6, cy - 34, "centre optique C", 7.5, MUTED)
    # axe optique
    d.add(Line(cx, cy, 455, cy, strokeColor=LINE, strokeWidth=0.8, strokeDashArray=[3, 3]))
    txt(d, 455, cy - 12, "axe z (profondeur)", 7.2, FAINT, anchor="end")
    # plan image
    px = 190
    d.add(Polygon([px - 14, cy - 62, px - 14, cy + 50, px + 14, cy + 70, px + 14, cy - 42], fillColor=BLUE_BG,
                  strokeColor=BLUE, strokeWidth=1, fillOpacity=0.9))
    txt(d, px, cy + 82, "plan image (pixels)", 7.5, BLUE, font="Segoe-Semibold")
    # point 3D et rayon
    X = (400, 172)
    t = (px - cx) / (X[0] - cx)
    u = (cx + t * (X[0] - cx), cy + t * (X[1] - cy))
    d.add(Line(cx, cy, *X, strokeColor=AMBER, strokeWidth=1.3))
    d.add(Circle(*X, 5, fillColor=AMBER, strokeColor=WHITE, strokeWidth=1))
    txt(d, X[0] - 8, X[1] + 10, "point 3D (X, Y, Z)", 8, INK, anchor="end", font="Segoe-Semibold")
    d.add(Circle(*u, 3.6, fillColor=RED, strokeColor=WHITE, strokeWidth=0.8))
    txt(d, u[0] + 22, u[1] - 14, "pixel (u, v)", 7.5, RED, anchor="start", font="Segoe-Semibold")
    # focale
    d.add(Line(cx, cy - 74, px, cy - 74, strokeColor=MUTED, strokeWidth=0.7))
    for xx in (cx, px):
        d.add(Line(xx, cy - 78, xx, cy - 70, strokeColor=MUTED, strokeWidth=0.7))
    txt(d, (cx + px) / 2, cy - 86, "focale f", 7.5, MUTED)
    # repère monde
    wx, wy = 330, 22
    arrow(d, wx, wy, wx + 30, wy, INK, 0.9, 4)
    arrow(d, wx, wy, wx, wy + 30, INK, 0.9, 4)
    arrow(d, wx, wy, wx - 17, wy - 12, INK, 0.9, 4)
    txt(d, wx + 36, wy + 4, "repère monde", 7.2, MUTED, anchor="start")
    txt(d, wx + 36, wy - 8, "[R | t] : monde → caméra", 7.5, INK, anchor="start", font="Segoe-Semibold")
    return d


def d_vggt():
    d = Drawing(CW, 222)
    y0 = 70
    for k in range(3):
        d.add(Rect(4 + k * 6, y0 + 62 - k * 6, 44, 58, fillColor=WHITE, strokeColor=BLUE, strokeWidth=0.9))
    txt(d, 32, y0 + 36, "N images", 7.6, INK, font="Segoe-Semibold")
    txt(d, 32, y0 + 26, "(≤ 50 sur 8 Go)", 7, MUTED)
    arrow(d, 64, y0 + 88, 82, y0 + 88)
    gx, gy, cell = 86, y0 + 64, 8
    for i in range(6):
        for j in range(6):
            d.add(Rect(gx + i * cell, gy + j * cell - 12, cell - 1, cell - 1,
                       fillColor=BLUE_BG if (i + j) % 2 else H("#d6e1fb"), strokeColor=None))
    txt(d, gx + 24, gy - 24, "patches 14×14 px", 7, MUTED)
    txt(d, gx + 24, gy - 33, "→ 1 jeton chacun", 7, MUTED)
    arrow(d, 140, y0 + 88, 158, y0 + 88)
    # bloc d'attention
    bx, bw, bh = 162, 168, 132
    d.add(Rect(bx, y0 + 22, bw, bh, rx=8, ry=8, fillColor=WHITE, strokeColor=BLUE, strokeWidth=1.1))
    txt(d, bx + bw / 2, y0 + 22 + bh - 15, "Transformer : 24 blocs", 8.6, BLUE, font="Segoe-Semibold")
    for k, (lab, sub) in enumerate([("Attention intra-image", "chaque jeton regarde son image"),
                                    ("Attention globale", "chaque jeton regarde toutes les images")]):
        yy = y0 + 22 + bh - 58 - k * 46
        box(d, bx + 10, yy, bw - 20, 36, lab, [sub], BLUE_BG, BLUE, tsize=8, size=6.9, r=5)
    txt(d, bx + bw / 2, y0 + 28, "× 24, en alternance", 7, MUTED)
    # têtes
    hx = 362
    box(d, hx, y0 + 98, 120, 48, "Tête caméra", ["K et [R | t] par image"], BLUE_BG, BLUE, tsize=8.4, size=7)
    box(d, hx, y0 + 32, 120, 48, "Tête profondeur", ["profondeur + confiance", "par pixel"], BLUE_BG, BLUE,
        tsize=8.4, size=7)
    arrow(d, bx + bw, y0 + 110, hx, y0 + 122)
    arrow(d, bx + bw, y0 + 70, hx, y0 + 56)
    box(d, hx, -2 + 10, 120, 38, "Nuage de points", ["rétroprojection des pixels sûrs"], VIOLET_BG, VIOLET,
        tsize=8.4, size=7, tcolor=VIOLET)
    arrow(d, hx + 60, y0 + 32, hx + 60, 48)
    txt(d, 0, 26, "Une seule passe, sans optimisation itérative", 7.6, MUTED, anchor="start")
    txt(d, 0, 15, "(COLMAP : détection → appariement → triangulation → ajustement)", 7, FAINT, anchor="start")
    return d


def d_padding():
    d = Drawing(CW, 128)
    # image portrait
    d.add(Rect(20, 22, 54, 96, fillColor=H("#dfe7f7"), strokeColor=BLUE, strokeWidth=0.9))
    txt(d, 47, 66, "1080 × 1920", 7, INK)
    txt(d, 47, 8, "image portrait", 7, MUTED)
    arrow(d, 86, 60, 120, 60)
    txt(d, 103, 68, "réduction", 6.8, MUTED)
    d.add(Rect(130, 22, 42, 76, fillColor=H("#dfe7f7"), strokeColor=BLUE, strokeWidth=0.9))
    txt(d, 151, 56, "291 × 518", 7, INK)
    arrow(d, 184, 60, 218, 60)
    txt(d, 201, 68, "padding", 6.8, MUTED)
    d.add(Rect(228, 22, 76, 76, fillColor=WHITE, strokeColor=BLUE, strokeWidth=0.9, strokeDashArray=[3, 2]))
    d.add(Rect(245, 22, 42, 76, fillColor=H("#dfe7f7"), strokeColor=BLUE, strokeWidth=0.9))
    txt(d, 266, 8, "carré 518 × 518 (bandes blanches)", 7, MUTED)
    arrow(d, 314, 60, 340, 60)
    box(d, 348, 26, 134, 70, "VGGT", ["sur un carré : champ de vision", "horizontal = vertical", "→ fx = fy correct",
                                       "puis K ramené à l'image d'origine"], BLUE_BG, BLUE, tsize=8.6, size=6.9)
    return d


def d_gaussian():
    d = Drawing(CW, 190)
    cx, cy = 92, 98
    gaussian(d, cx, cy, 70, 34, 28, VIOLET, 0.9, 7)
    d.add(Circle(cx, cy, 2.6, fillColor=INK, strokeColor=None))
    a = math.radians(28)
    arrow(d, cx, cy, cx + 70 * math.cos(a), cy + 70 * math.sin(a), INK, 0.9, 4.5)
    arrow(d, cx, cy, cx - 34 * math.sin(a), cy + 34 * math.cos(a), INK, 0.9, 4.5)
    txt(d, cx + 64 * math.cos(a) + 6, cy + 64 * math.sin(a) + 6, "s₁".replace("₁", "1"), 8, INK, anchor="start",
        font="Segoe-Semibold")
    txt(d, cx - 34 * math.sin(a) - 8, cy + 34 * math.cos(a) + 2, "s2", 8, INK, anchor="end", font="Segoe-Semibold")
    txt(d, cx + 6, cy - 12, "μ (centre)", 7.5, INK, anchor="start")
    p = RPath(strokeColor=MUTED, strokeWidth=0.8, fillColor=None)
    p.moveTo(cx + 30, cy)
    p.curveTo(cx + 30, cy + 7, cx + 28, cy + 12, cx + 26.5, cy + 14)
    d.add(p)
    d.add(Line(cx, cy, cx + 44, cy, strokeColor=MUTED, strokeWidth=0.6, strokeDashArray=[2, 2]))
    txt(d, cx + 36, cy + 6, "q", 7.5, MUTED, anchor="start", font="Segoe-Italic")
    txt(d, cx, 14, "1 gaussienne = 1 ellipsoïde flou", 8, INK, font="Segoe-Semibold")
    # tableau des paramètres
    tx, tw = 186, 186
    rows = [("μ", "position du centre (x, y, z)", "3"), ("s", "taille sur chaque axe (en log)", "3"),
            ("q", "orientation (quaternion)", "4"), ("α", "opacité (en logit)", "1"),
            ("SH", "couleur selon l'angle de vue", "48")]
    txt(d, tx, 170, "symbole", 7.4, FAINT, anchor="start")
    txt(d, tx + 44, 170, "rôle", 7.4, FAINT, anchor="start")
    txt(d, tx + tw, 170, "nombres", 7.4, FAINT, anchor="end")
    for k, (sym, role, n) in enumerate(rows):
        yy = 150 - k * 22
        d.add(Line(tx, yy - 7, tx + tw, yy - 7, strokeColor=LINE, strokeWidth=0.5))
        txt(d, tx + 4, yy, sym, 9, VIOLET, anchor="start", font="Segoe-Semibold")
        txt(d, tx + 44, yy, role, 7.8, INK, anchor="start")
        txt(d, tx + tw, yy, n, 7.8, MUTED, anchor="end")
    txt(d, tx, 22, "≈ 59 nombres par gaussienne", 7.6, MUTED, anchor="start")
    # nuage
    import random
    rnd = random.Random(3)
    ox, oy = 438, 98
    for _ in range(70):
        ang = rnd.random() * 2 * math.pi
        r = rnd.random() ** 0.7 * 32
        x, y = ox + r * math.cos(ang) * 1.1, oy + r * math.sin(ang)
        col = [VIOLET, H("#9c7be0"), H("#c3aef0"), AMBER, H("#5a3a9a")][rnd.randrange(5)]
        gaussian(d, x, y, 4 + rnd.random() * 9, 2 + rnd.random() * 4, rnd.random() * 180, col, 0.8, 3)
    txt(d, ox, 30, "des centaines de milliers", 7.6, INK, font="Segoe-Semibold")
    txt(d, ox, 20, "→ une scène", 7.6, MUTED)
    return d


def d_render():
    d = Drawing(CW, 186)
    py, px = 112, 40
    d.add(Rect(px - 14, py - 14, 28, 28, fillColor=RED_BG, strokeColor=RED, strokeWidth=1))
    txt(d, px, py - 26, "pixel p", 7.6, RED, font="Segoe-Semibold")
    d.add(Line(px + 14, py, 470, py, strokeColor=FAINT, strokeWidth=0.9, strokeDashArray=[3, 2]))
    txt(d, 470, py + 8, "rayon", 7, FAINT, anchor="end")
    gs = [(150, VIOLET, "0,6", "1", "0,6"), (260, AMBER, "0,5", "0,4", "0,2"), (370, H("#3d8fd1"), "0,8", "0,2", "0,16")]
    for k, (x, col, a, T, w) in enumerate(gs):
        gaussian(d, x, py + (8 if k == 1 else -4), 34, 24, 15 + 20 * k, col, 0.95, 6)
        txt(d, x, py + 40, f"gaussienne {k + 1}", 7.6, INK, font="Segoe-Semibold")
        txt(d, x, 58, f"α{k + 1} = {a}", 7.8, INK)
        txt(d, x, 46, f"T{k + 1} = {T}", 7.8, MUTED)
        txt(d, x, 32, f"w = α·T = {w}", 7.8, VIOLET if k == 0 else (AMBER if k == 1 else H("#2f73b1")),
            font="Segoe-Semibold")
    txt(d, 64, 58, "opacité", 7, FAINT, anchor="start")
    txt(d, 64, 46, "transmittance", 7, FAINT, anchor="start")
    txt(d, 64, 32, "poids final", 7, FAINT, anchor="start")
    txt(d, 260, 168, "triées de la plus proche à la plus lointaine", 7.6, MUTED)
    arrow(d, 120, 160, 400, 160, FAINT, 0.8, 4.5)
    txt(d, CW / 2, 10, "couleur(p) = Σ cᵢ · wᵢₚ  avec  wᵢₚ = αᵢ · Tᵢ  et  Tᵢ = ∏ (1 − αⱼ) pour j devant i"
        .replace("ᵢₚ", "ip").replace("ᵢ", "i").replace("ⱼ", "j"), 8.6, INK, font="Segoe-Semibold")
    return d


def d_training():
    d = Drawing(CW, 214)
    w, h = 150, 48
    pts = {"vue": (20, 150), "rendu": (312, 150), "perte": (312, 60), "maj": (20, 60)}
    box(d, *pts["vue"], w, h, "1 · Tirer une vue au hasard", ["une des photos + sa caméra"], BLUE_BG, BLUE, 8.6, 7.2)
    box(d, *pts["rendu"], w, h, "2 · Rendu (gsplat)", ["image calculée depuis", "les gaussiennes actuelles"],
        VIOLET_BG, VIOLET, 8.6, 7.2)
    box(d, *pts["perte"], w, h, "3 · Comparer à la photo", ["perte = 0,8 · L1 + 0,2 · (1 − SSIM)"], RED_BG, RED,
        8.6, 7.2)
    box(d, *pts["maj"], w, h, "4 · Corriger (Adam)", ["gradients → μ, s, q, α, SH"], VIOLET_BG, VIOLET, 8.6, 7.2)
    arrow(d, 170, 174, 312, 174)
    arrow(d, 387, 150, 387, 108)
    arrow(d, 312, 84, 170, 84)
    arrow(d, 95, 108, 95, 150)
    txt(d, 241, 180, "", 7)
    txt(d, 241, 129, "× 7 000 itérations", 9, INK, font="Segoe-Semibold")
    txt(d, 241, 116, "≈ 4 à 5 min sur RTX 2070", 7.4, MUTED)
    txt(d, 241, 90, "rétropropagation", 7.2, MUTED)
    # densification
    box(d, 20, 2, 442, 40, None, [], PAPER, LINE, r=6)
    txt(d, 32, 26, "Tous les 100 pas (de 500 à 4 200) : densification", 8, INK, anchor="start",
        font="Segoe-Semibold")
    txt(d, 32, 13, "clone les petites gaussiennes mal placées · coupe en deux les grandes · supprime les transparentes",
        7.2, MUTED, anchor="start")
    return d


def d_sam_siglip():
    d = Drawing(CW, 200)
    # image + masques
    ix, iy = 6, 100
    d.add(Rect(ix, iy, 86, 64, fillColor=H("#eef1f6"), strokeColor=FAINT, strokeWidth=0.8))
    d.add(Ellipse(ix + 30, iy + 30, 16, 20, fillColor=H("#86b86b"), strokeColor=None))
    d.add(Rect(ix + 54, iy + 14, 22, 22, fillColor=H("#e2a33b"), strokeColor=None))
    d.add(Ellipse(ix + 66, iy + 50, 10, 7, fillColor=H("#d9534f"), strokeColor=None))
    txt(d, ix + 43, iy - 12, "image", 7.4, MUTED)
    arrow(d, 98, iy + 32, 122, iy + 32)
    box(d, 126, iy + 8, 58, 48, "SAM", ["« segmente", "tout »"], AMBER_BG, AMBER, 8.4, 6.8, tcolor=AMBER)
    arrow(d, 188, iy + 32, 210, iy + 32)
    # crops
    for k, (col, shape) in enumerate([(H("#86b86b"), "e"), (H("#e2a33b"), "r"), (H("#d9534f"), "e2")]):
        x = 214 + k * 34
        d.add(Rect(x, iy + 16, 30, 30, fillColor=INK, strokeColor=None))
        if shape == "e":
            d.add(Ellipse(x + 15, iy + 31, 9, 12, fillColor=col, strokeColor=None))
        elif shape == "r":
            d.add(Rect(x + 5, iy + 21, 20, 20, fillColor=col, strokeColor=None))
        else:
            d.add(Ellipse(x + 15, iy + 31, 12, 8, fillColor=col, strokeColor=None))
    txt(d, 264, iy - 2, "1 crop par masque, fond noir", 7, MUTED)
    arrow(d, 318, iy + 32, 338, iy + 32)
    box(d, 342, iy + 6, 74, 52, "SigLIP 2", ["encodeur", "d'images"], AMBER_BG, AMBER, 8.4, 6.8, tcolor=AMBER)
    arrow(d, 416, iy + 32, 432, iy + 32)
    for k in range(10):
        d.add(Rect(436 + (k % 5) * 9, iy + 40 - (k // 5) * 14, 8, 12,
                   fillColor=[AMBER, H("#e6b45a"), H("#f3d59c"), H("#c98a12"), H("#f7e3bb")][k % 5],
                   strokeColor=None))
    txt(d, 458, iy + 4, "768 nombres", 7, MUTED)
    # texte
    ty = 22
    box(d, 126, ty, 110, 36, None, [], WHITE, RED, r=6)
    txt(d, 181, ty + 14, "« pomme verte »", 8.6, RED, font="Segoe-Semibold")
    arrow(d, 236, ty + 18, 338, ty + 18)
    box(d, 342, ty - 6, 74, 48, "SigLIP 2", ["encodeur", "de texte"], RED_BG, RED, 8.4, 6.8, tcolor=RED)
    arrow(d, 416, ty + 18, 432, ty + 18)
    for k in range(5):
        d.add(Rect(436 + k * 9, ty + 12, 8, 12, fillColor=[RED, H("#e98a7c"), H("#f4c2b9"), H("#b8372a"),
                                                           H("#f9dcd6")][k], strokeColor=None))
    arrow(d, 458, iy - 3, 458, ty + 30, GREEN, 1.1, 5, [3, 2])
    arrow(d, 458, ty + 30, 458, iy - 3, GREEN, 1.1, 5, [3, 2])
    txt(d, 450, 80, "cosinus élevé", 7.4, GREEN, anchor="end", font="Segoe-Semibold")
    txt(d, 450, 70, "= même sens", 7, GREEN, anchor="end")
    return d


def d_space():
    d = Drawing(CW, 190)
    ox, oy = 40, 20
    d.add(Rect(ox, oy, 300, 160, rx=10, ry=10, fillColor=PAPER, strokeColor=LINE, strokeWidth=0.8))
    txt(d, ox + 10, oy + 146, "espace commun images + textes (vue en 2D)", 7.4, MUTED, anchor="start")
    groups = [((115, 118), "renard", [("crop renard", -22, -20), ("« fox »", 6, -4), ("« renard »", -24, 14)]),
              ((250, 120), "rideau", [("crop rideau", -20, -16), ("« rideau rouge »", 8, 14)]),
              ((210, 52), "fleurs", [("crop fleur", -24, 10), ("« fleurs »", 22, -8)])]
    for (gx, gy), _, items in groups:
        d.add(Circle(gx, gy, 34, fillColor=WHITE, strokeColor=LINE, strokeWidth=0.6, fillOpacity=0.8))
        for lab, dx, dy in items:
            is_text = lab.startswith("«")
            d.add(Circle(gx + dx, gy + dy, 3.4, fillColor=RED if is_text else AMBER, strokeColor=None))
            txt(d, gx + dx + 6, gy + dy - 2.5, lab, 7, INK, anchor="start")
    for lab, (x, y) in [("« object »", (70, 48)), ("« stuff »", (95, 36)), ("« texture »", (300, 40))]:
        d.add(Circle(x, y, 3, fillColor=FAINT, strokeColor=None))
        txt(d, x + 5, y - 2.5, lab, 6.8, FAINT, anchor="start")
    lx = 360
    d.add(Circle(lx + 4, 150, 3.4, fillColor=AMBER, strokeColor=None))
    txt(d, lx + 12, 147, "embedding d'un crop d'image", 7.4, INK, anchor="start")
    d.add(Circle(lx + 4, 132, 3.4, fillColor=RED, strokeColor=None))
    txt(d, lx + 12, 129, "embedding d'un texte", 7.4, INK, anchor="start")
    d.add(Circle(lx + 4, 114, 3, fillColor=FAINT, strokeColor=None))
    txt(d, lx + 12, 111, "négatifs génériques", 7.4, INK, anchor="start")
    txt(d, lx, 84, "Proches = même sens.", 7.6, INK, anchor="start", font="Segoe-Semibold")
    txt(d, lx, 72, "« renard » et « fox » :", 7.4, MUTED, anchor="start")
    txt(d, lx, 61, "cosinus 0,95 (SigLIP 2", 7.4, MUTED, anchor="start")
    txt(d, lx, 50, "est multilingue)", 7.4, MUTED, anchor="start")
    return d


def d_levels():
    d = Drawing(CW, 176)
    def panel(x, title, ear_col, head_col, note):
        d.add(Rect(x, 30, 150, 112, fillColor=WHITE, strokeColor=LINE, strokeWidth=0.8))
        d.add(Ellipse(x + 75, 82, 44, 36, fillColor=head_col, strokeColor=None))
        d.add(Polygon([x + 48, 108, x + 58, 140, x + 70, 112], fillColor=ear_col, strokeColor=None))
        d.add(Polygon([x + 82, 112, x + 96, 140, x + 104, 106], fillColor=ear_col, strokeColor=None))
        d.add(Circle(x + 92, 126, 3.4, fillColor=INK, strokeColor=WHITE, strokeWidth=0.8))
        txt(d, x + 75, 150, title, 8.6, INK, font="Segoe-Semibold")
        txt(d, x + 75, 16, note, 7.2, MUTED)
    panel(10, "niveau « small »", H("#e69a3a"), H("#f6dcae"), "le pixel • reçoit le masque « oreille »")
    panel(176, "niveau « large »", H("#f6dcae"), H("#f6dcae"), "le même pixel reçoit le masque « renard »")
    box(d, 346, 70, 136, 78, "Choix à la requête", ["on calcule les deux niveaux", "et on garde celui qui",
                                                    "répond le mieux", "(ou on l'impose)"], AMBER_BG, AMBER, 8.4, 7,
        tcolor=AMBER)
    box(d, 346, 8, 136, 50, "Zones sans masque", ["fond texturé → ajouté", "comme masque s'il > 1 %"], PAPER, LINE,
        8.4, 7)
    return d


def d_lifting():
    d = Drawing(CW, 226)
    # gaussiennes
    gx = 40
    gcol = [VIOLET, AMBER, H("#3d8fd1"), H("#9c7be0")]
    for k in range(4):
        gaussian(d, gx, 172 - k * 40, 15, 10, 20 * k, gcol[k], 0.95, 4)
        txt(d, gx - 24, 169 - k * 40, f"c{k + 1}", 8, INK, anchor="end", font="Segoe-Semibold")
    txt(d, gx, 18, "gaussiennes i", 7.6, MUTED)
    # pixels
    px0, py0, cs = 170, 60, 22
    for i in range(4):
        for j in range(4):
            d.add(Rect(px0 + i * cs, py0 + j * cs, cs - 2, cs - 2, fillColor=H("#fde9c8") if (i + j) % 3 else AMBER_BG,
                       strokeColor=None))
    txt(d, px0 + 2 * cs, 40, "pixels p d'une image", 7.6, MUTED)
    txt(d, px0 + 2 * cs, 29, "(F(p) = vecteur de sens du masque)", 7, FAINT)
    for k, (i, j, wv) in enumerate([(0, 3, "w = 0,6"), (1, 2, ""), (2, 2, ""), (1, 1, "w = 0,2"), (3, 0, "")]):
        src = (gx + 16, 172 - (k % 4) * 40)
        dst = (px0 + i * cs + cs / 2, py0 + j * cs + cs / 2)
        d.add(Line(*src, *dst, strokeColor=gcol[k % 4], strokeWidth=0.9, strokeOpacity=0.7))
        if wv:
            txt(d, (src[0] + dst[0]) / 2, (src[1] + dst[1]) / 2 + 5, wv, 6.8, MUTED)
    # explication
    ex = 282
    box(d, ex, 152, 200, 56, "1 · Rendu (aller)", ["image(p) = Σᵢ wᵢₚ · cᵢ".replace("ᵢₚ", "ip").replace("ᵢ", "i"),
                                               "linéaire en c (les « couleurs »)"], VIOLET_BG, VIOLET, 8.4, 7.4,
        tcolor=VIOLET)
    box(d, ex, 84, 200, 56, "2 · Gradient (retour)", ["L = Σₚ ⟨image(p), F(p)⟩".replace("ₚ", "p")
                                                    .replace("⟨", "<").replace("⟩", ">"),
                                                    "∂L/∂cᵢ = Σₚ wᵢₚ · F(p)".replace("ᵢₚ", "ip").replace("ₚ", "p")
                                                    .replace("ᵢ", "i")], AMBER_BG, AMBER, 8.4, 7.4, tcolor=AMBER)
    box(d, ex, 14, 200, 58, "3 · Résultat", ["somme des sens des pixels couverts,", "pondérée par la contribution",
                                         "→ diviser par Σ w (canal de 1)"], GREEN_BG, GREEN, 8.4, 7.2, tcolor=GREEN)
    arrow(d, ex + 100, 152, ex + 100, 140)
    arrow(d, ex + 100, 84, ex + 100, 72)
    return d


def d_svd():
    d = Drawing(CW, 140)
    def bar(x, y, h, n, col, label):
        for k in range(n):
            d.add(Rect(x, y + k * h / n, 16, h / n - 0.6, fillColor=col, strokeColor=None,
                       fillOpacity=0.35 + 0.65 * ((k * 7) % 10) / 10))
        txt(d, x + 8, y - 11, label, 7.2, MUTED)
    bar(20, 18, 110, 24, AMBER, "768 dims")
    arrow(d, 46, 73, 112, 73)
    box(d, 118, 48, 112, 50, "× P  (768 × 128)", ["base de la SVD", "non centrée"], AMBER_BG, AMBER, 8.2, 7)
    arrow(d, 230, 73, 268, 73)
    bar(274, 46, 54, 12, AMBER, "128 dims")
    txt(d, 310, 106, "Le produit scalaire avec un texte t est conservé :", 7.8, INK, anchor="start",
        font="Segoe-Semibold")
    txt(d, 310, 90, "<t, f>  ≈  <t·P, f·P>", 9.4, AMBER, anchor="start", font="Segoe-Semibold")
    txt(d, 310, 72, "98,3 % de l'« énergie » des vecteurs gardée", 7.4, MUTED, anchor="start")
    txt(d, 310, 60, "6 fois moins de mémoire", 7.4, MUTED, anchor="start")
    txt(d, 310, 42, "(LangSplat : autoencodeur vers 3 dims,", 7, FAINT, anchor="start")
    txt(d, 310, 31, "qui ne conserve pas ces produits scalaires)", 7, FAINT, anchor="start")
    return d


def d_relevance():
    d = Drawing(CW, 170)
    vals = [("« renard »", 0.18, RED), ("object", 0.12, FAINT), ("things", 0.10, FAINT), ("stuff", 0.11, FAINT),
            ("texture", 0.09, FAINT)]
    x0, y0, scale = 22, 30, 520
    txt(d, x0, 150, "cosinus entre la gaussienne et…", 7.6, MUTED, anchor="start")
    for k, (lab, v, col) in enumerate(vals):
        x = x0 + k * 40
        d.add(Rect(x, y0, 24, v * scale, fillColor=col, strokeColor=None, rx=2, ry=2))
        txt(d, x + 12, y0 + v * scale + 4, f"{v:.2f}".replace(".", ","), 7, INK)
        txt(d, x + 12, y0 - 11, lab, 6.9, INK if k == 0 else MUTED)
    arrow(d, 232, 80, 262, 80)
    tx = 270
    txt(d, tx, 140, "pour chaque négatif n :", 7.6, MUTED, anchor="start")
    txt(d, tx, 124, "sigmoïde(10 · (cos_requête − cos_n))", 8.6, INK, anchor="start", font="Segoe-Semibold")
    for k, (lab, v) in enumerate([("object", 0.65), ("things", 0.69), ("stuff", 0.67), ("texture", 0.71)]):
        txt(d, tx + 10, 104 - k * 13, f"{lab} → {v:.2f}".replace(".", ","), 7.6, MUTED, anchor="start")
    txt(d, tx, 40, "pertinence = minimum = 0,65", 9, RED, anchor="start", font="Segoe-Semibold")
    txt(d, tx, 26, "> 0,5 : plus proche de « renard » que de n'importe quel « truc »", 7.2, MUTED, anchor="start")
    txt(d, x0, 4, "valeurs illustratives", 6.8, FAINT, anchor="start", font="Segoe-Italic")
    return d


def d_eval():
    d = Drawing(CW, 214)
    y = 128
    box(d, 0, y, 82, 56, "Vue annotée", ["caméra du dataset"], BLUE_BG, BLUE, 8.2, 7)
    arrow(d, 82, y + 28, 98, y + 28)
    box(d, 98, y, 86, 56, "Rendu", ["carte de sens", "par pixel"], VIOLET_BG, VIOLET, 8.2, 7, tcolor=VIOLET)
    arrow(d, 184, y + 28, 200, y + 28)
    # heatmap
    import random
    rnd = random.Random(1)
    for i in range(8):
        for j in range(6):
            v = math.exp(-((i - 5) ** 2 + (j - 3) ** 2) / 5) + rnd.random() * 0.12
            col = colors.Color(1, 1 - 0.75 * min(v, 1), 1 - 0.85 * min(v, 1))
            d.add(Rect(204 + i * 9, y + 2 + j * 9, 8.5, 8.5, fillColor=col, strokeColor=None))
    txt(d, 240, y - 10, "pertinence de la requête", 7, MUTED)
    arrow(d, 280, y + 28, 296, y + 28)
    box(d, 296, y, 94, 56, "Lissage", ["moyenne avec un", "flou 30 × 30"], PAPER, LINE, 8.2, 7)
    # branche IoU
    arrow(d, 390, y + 34, 404, y + 50, MUTED)
    box(d, 404, y + 36, 78, 48, "→ IoU", ["seuil 0,4", "+ vote 7 × 7"], RED_BG, RED, 8, 6.8, tcolor=RED)
    # branche loc
    arrow(d, 390, y + 20, 404, y + 4, MUTED)
    box(d, 404, y - 30, 78, 48, "→ Localisation", ["point maximum", "de la carte floutée"], RED_BG, RED, 8, 6.8,
        tcolor=RED)
    # IoU illustration
    iy = 10
    d.add(Circle(70, iy + 46, 32, fillColor=GREEN, strokeColor=None, fillOpacity=0.35))
    d.add(Circle(100, iy + 46, 32, fillColor=RED, strokeColor=None, fillOpacity=0.35))
    txt(d, 52, iy + 44, "annoté", 7, INK)
    txt(d, 118, iy + 44, "prédit", 7, INK)
    txt(d, 175, iy + 52, "IoU = intersection / union", 8.6, INK, anchor="start", font="Segoe-Semibold")
    txt(d, 175, iy + 38, "1 = recouvrement parfait, 0 = aucun", 7.4, MUTED, anchor="start")
    txt(d, 175, iy + 18, "Localisation = le maximum tombe-t-il", 8.6, INK, anchor="start", font="Segoe-Semibold")
    txt(d, 175, iy + 5, "dans la boîte de l'objet ? (oui / non)", 7.4, MUTED, anchor="start")
    return d


def d_code():
    d = Drawing(CW, 248)
    box(d, 171, 208, 140, 36, "__main__.py", ["python -m lang3d <commande> …"], PAPER, FAINT, 8.4, 6.8)
    d.add(Line(45, 199, 437, 199, strokeColor=LINE, strokeWidth=0.8))
    d.add(Line(241, 208, 241, 199, strokeColor=LINE, strokeWidth=0.8))
    for xs in (45, 143, 241, 339, 437):
        arrow(d, xs, 199, xs, 191, LINE, 0.8, 3.5)
    txt(d, CW / 2, 134, "chaque commande appelle le module du même nom", 7, FAINT)
    mods = [
        ("frames.py", "vidéo → images", BLUE_BG, BLUE, 0, 150),
        ("poses.py", "VGGT", BLUE_BG, BLUE, 98, 150),
        ("colmap.py", "poses existantes", BLUE_BG, BLUE, 196, 150),
        ("train.py", "boucle 3DGS", VIOLET_BG, VIOLET, 294, 150),
        ("demo.py", "GIF", RED_BG, RED, 392, 150),
        ("features.py", "SAM + crops", AMBER_BG, AMBER, 0, 84),
        ("lift.py", "2D → 3D", AMBER_BG, AMBER, 98, 84),
        ("query.py", "pertinence", RED_BG, RED, 196, 84),
        ("viewer.py", "web (viser)", RED_BG, RED, 294, 84),
        ("evaluate.py", "benchmark LERF", GREEN_BG, GREEN, 392, 84),
    ]
    for name, sub, bg, col, x, y in mods:
        box(d, x, y, 90, 40, name, [sub], bg, col, 8.2, 6.9, tcolor=col, font_lines="Segoe")
    for name, sub, x in [("scene.py", "charger caméras + images", 40), ("gaussians.py", "rendu, sauvegarde, .ply", 182),
                         ("encoder.py", "SigLIP 2 (image + texte)", 324)]:
        box(d, x, 18, 118, 40, name, [sub], WHITE, FAINT, 8.2, 6.9, font_lines="Segoe")
    txt(d, 312, 64, "modules partagés", 7.2, FAINT)
    for xs in (99, 241, 383):
        d.add(Line(xs, 58, xs, 74, strokeColor=LINE, strokeWidth=0.8))
    return d


def d_cuda():
    d = Drawing(CW, 128)
    rows = [("CUDA Toolkit (nvcc)", "compiler du code CUDA", "absent : pas nécessaire ici", PAPER, FAINT, "✗"),
            ("CUDA Runtime (cuBLAS, cuDNN…)", "exécuter les calculs", "fourni par PyTorch (cu124)", GREEN_BG, GREEN,
             "✓"),
            ("Pilote NVIDIA", "parler à la carte", "installé (jusqu'à CUDA 13.5)", GREEN_BG, GREEN, "✓")]
    for k, (name, role, state, bg, col, mark) in enumerate(rows):
        y = 86 - k * 40
        d.add(Rect(60, y, 420, 34, rx=5, ry=5, fillColor=bg, strokeColor=col, strokeWidth=0.9))
        txt(d, 74, y + 13, name, 8.6, INK, anchor="start", font="Segoe-Semibold")
        txt(d, 260, y + 13, role, 7.6, MUTED, anchor="start")
        txt(d, 470, y + 13, state, 7.6, col if col != FAINT else MUTED, anchor="end", font="Segoe-Semibold")
    txt(d, 30, 103, "haut", 7, FAINT)
    txt(d, 30, 22, "bas", 7, FAINT)
    arrow(d, 30, 30, 30, 96, FAINT, 0.8, 4)
    return d


def d_iou_chart():
    res = json.loads((ROOT / "data/lerf_ovs/eval_results.json").read_text())["scenes"]["figurines"]["queries"]
    agg = {}
    for q in res:
        agg.setdefault(q["query"], []).append(q["iou"])
    rows = sorted(((k, sum(v) / len(v)) for k, v in agg.items()), key=lambda r: -r[1])
    lh, top = 15.5, 24
    hgt = top + lh * len(rows) + 28
    d = Drawing(CW, hgt)
    lx, x0, x1 = 116, 122, CW - 34
    y_axis = 22
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        x = x0 + v * (x1 - x0)
        d.add(Line(x, y_axis, x, hgt - top + 4, strokeColor=H("#eceff3"), strokeWidth=0.6))
        txt(d, x, y_axis - 11, f"{int(v * 100)} %", 7, FAINT)
    mean = sum(q["iou"] for q in res) / len(res)
    xm = x0 + mean * (x1 - x0)
    d.add(Line(xm, y_axis, xm, hgt - top + 8, strokeColor=INK, strokeWidth=0.8, strokeDashArray=[3, 2]))
    txt(d, xm + 4, hgt - top + 2, f"moyenne {mean * 100:.1f} %".replace(".", ","), 7.2, INK, anchor="start")
    for k, (name, v) in enumerate(rows):
        y = hgt - top - (k + 1) * lh + 4
        txt(d, lx, y + 2.5, name, 7.6, INK, anchor="end")
        w = max(v * (x1 - x0), 1.5)
        d.add(Rect(x0, y, w, lh - 6, fillColor=AMBER, strokeColor=None, rx=2, ry=2))
        d.add(Rect(x0 + w + 2, y - 1, 14, lh - 4, fillColor=WHITE, strokeColor=None))  # étiquette lisible sur la ligne
        txt(d, x0 + w + 4, y + 2.4, f"{v * 100:.0f}", 7.2, MUTED, anchor="start")
    return d


# ---------------------------------------------------------------- mise en page
class Doc(BaseDocTemplate):
    def __init__(self, path):
        super().__init__(str(path), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=22 * mm,
                         bottomMargin=20 * mm, title="lang3d — le projet de fond en comble", author=AUTHOR,
                         subject="Scène 3D interrogeable en langage naturel")
        frame = Frame(MARGIN, 20 * mm, CW, H_PAGE - 42 * mm, id="f", leftPadding=0, rightPadding=0, topPadding=0,
                      bottomPadding=0)
        self.addPageTemplates([PageTemplate("cover", [frame], onPage=self.cover_bg),
                               PageTemplate("normal", [frame], onPage=self.decor)])

    def cover_bg(self, c, doc):
        c.saveState()
        c.setFillColor(H("#121826"))
        c.rect(0, H_PAGE - 118 * mm, W_PAGE, 118 * mm, stroke=0, fill=1)
        clip = c.beginPath()
        clip.rect(0, H_PAGE - 118 * mm, W_PAGE, 118 * mm)
        c.clipPath(clip, stroke=0, fill=0)
        import random
        rnd = random.Random(7)
        for _ in range(140):
            x, y = rnd.random() * W_PAGE, H_PAGE - rnd.random() * 118 * mm
            col = [VIOLET, BLUE, AMBER, RED, H("#9c7be0")][rnd.randrange(5)]
            c.setFillColor(col)
            c.setFillAlpha(0.10 + rnd.random() * 0.25)
            c.saveState()
            c.translate(x, y)
            c.rotate(rnd.random() * 180)
            rx = 4 + rnd.random() * 26
            c.ellipse(-rx, -rx * 0.45, rx, rx * 0.45, stroke=0, fill=1)
            c.restoreState()
        c.restoreState()

    def decor(self, c, doc):
        c.saveState()
        c.setFont("Segoe", 7.8)
        c.setFillColor(FAINT)
        c.drawString(MARGIN, H_PAGE - 13 * mm, "lang3d — le projet de fond en comble")
        c.drawRightString(W_PAGE - MARGIN, 11 * mm, str(doc.page))
        c.setStrokeColor(LINE)
        c.setLineWidth(0.5)
        c.line(MARGIN, H_PAGE - 15 * mm, W_PAGE - MARGIN, H_PAGE - 15 * mm)
        c.restoreState()

    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and f.style.name in ("h1", "h2"):
            level = 0 if f.style.name == "h1" else 1
            text = f.getPlainText()
            key = f"s{self.seq.nextf('toc')}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=level > 0)
            self.notify("TOCEntry", (level, text, self.page, key))


def H1(t):
    # Nouvelle page seulement s'il reste moins de 40 % de la page : évite les pages à moitié vides.
    return [CondPageBreak(0.4 * H_PAGE), P(t, "h1")]


def H2(t):
    return [CondPageBreak(90), P(t, "h2")]


def stat(value, label, color):
    return [Paragraph(f'<font name="Segoe-Bold" size="20" color="{hexs(color)}">{value}</font>',
                      style("s", alignment=TA_LEFT, leading=24, spaceAfter=0)),
            Paragraph(label, style("sl", fontSize=8.4, leading=11, textColor=MUTED, spaceAfter=0))]


def hexs(c):
    return "#" + c.hexval()[2:]


def build():
    story = []
    # ---------------------------------------------------------- couverture
    story += [Spacer(1, 26 * mm),
              Paragraph('<font color="#9fb3ff">PROJET DE VISION PAR ORDINATEUR · NOVEMBRE 2026</font>',
                        style("ck", fontName="Segoe-Semibold", fontSize=9, textColor=WHITE, spaceAfter=10)),
              Paragraph("lang3d", style("ct", fontName="Segoe-Bold", fontSize=46, leading=52, textColor=WHITE,
                                        spaceAfter=6)),
              Paragraph("Le projet de fond en comble", style("cs", fontName="Segoe-Light", fontSize=22, leading=28,
                                                             textColor=WHITE, spaceAfter=14)),
              Paragraph("De la vidéo filmée au téléphone à une scène 3D que l'on interroge en langage naturel : "
                        "chaque étape, chaque idée, chaque choix, expliqués avec des schémas.",
                        style("cd", fontSize=11, leading=16, textColor=H("#c9d2e6"))),
              Spacer(1, 36 * mm)]
    stats = [stat("83,9 %", "de localisation sur le benchmark LERF (figurines), contre 80,4 % pour LangSplat", RED),
             stat("49,0 %", "d'IoU moyen sur la même scène, contre 44,7 % pour LangSplat", AMBER),
             stat("8 Go", "de VRAM suffisent : tout tourne sur une RTX 2070 grand public", VIOLET),
             stat("0", "itération d'entraînement pour passer les features de 2D à 3D", BLUE)]
    t = Table([[stats[0], stats[1]], [stats[2], stats[3]]], colWidths=[CW / 2] * 2)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 18), ("BOTTOMPADDING", (0, 0), (-1, -1), 16)]))
    story += [t, Spacer(1, 6 * mm),
              P(f"{AUTHOR} · code : github.com/Wazoou/lang3d", "small"), NextPageTemplate("normal"), PageBreak()]

    # ---------------------------------------------------------- sommaire
    toc = TableOfContents()
    toc.levelStyles = [S["toc1"], S["toc2"]]
    story += [Paragraph("Sommaire", style("h1toc", fontName="Segoe-Bold", fontSize=21, leading=26, spaceAfter=10)),
              Spacer(1, 4), toc, PageBreak()]

    # ---------------------------------------------------------- 1. idée
    story += H1("1. L'idée en une page")
    story += [P("On filme une pièce ou un objet en tournant autour. lang3d en tire une <b>maquette 3D</b> que "
                "l'on peut faire tourner dans un navigateur, puis on lui pose une question : « où est le renard ? ». "
                "Les parties de la maquette qui correspondent s'allument en rouge, sous tous les angles.", "lead"),
              P("Pour y arriver, il faut répondre à trois questions, une par famille d'étapes :"),
              *bullets(['<font color="%s"><b>Géométrie</b></font> : d\'où chaque photo a-t-elle été prise, et où sont '
                        'les choses dans l\'espace ?' % hexs(BLUE),
                        '<font color="%s"><b>Apparence</b></font> : comment représenter la scène pour pouvoir la '
                        'dessiner depuis n\'importe quel point de vue ?' % hexs(VIOLET),
                        '<font color="%s"><b>Sens</b></font> : comment savoir que tel morceau de la scène est « un '
                        'renard », sans jamais avoir appris le mot « renard » ?' % hexs(AMBER)]),
              figure(d_pipeline(), "Figure 1 — Le pipeline complet. Chaque boîte est un module Python et une "
                                   "commande <font name='Mono'>python -m lang3d …</font> ; sous chaque boîte, le fichier "
                                   "qu'elle produit."),
              P("Le projet s'inspire de deux travaux de recherche : <b>LERF</b> (Kerr et al., ICCV 2023), qui a "
                "introduit l'idée de « champs de langage » 3D, et <b>LangSplat</b> (Qin et al., CVPR 2024), qui l'a "
                "adaptée au Gaussian Splatting. lang3d reprend leurs idées et les simplifie pour un GPU grand public, "
                "avec trois différences clés, détaillées plus loin : des poses de caméra calculées par un réseau "
                "(VGGT), un modèle vision-langage plus récent et multilingue (SigLIP 2), et un passage 2D → 3D sans "
                "aucun entraînement.")]

    # ---------------------------------------------------------- 2. notions
    story += H1("2. Les notions de base")
    story += H2("2.1 La caméra : intrinsèques et extrinsèques")
    story += [P("Tout le projet repose sur un modèle simple de caméra, le <b>sténopé</b> (« pinhole ») : un point "
                "3D se projette sur l'image le long de la droite qui le relie au centre optique."),
              figure(d_camera(), "Figure 2 — Le modèle de caméra sténopé. Deux familles de paramètres : où est la "
                                 "caméra ([R | t]) et comment elle transforme des rayons en pixels (K)."),
              P("Les <b>intrinsèques</b> décrivent l'objectif et le capteur. Elles tiennent dans une matrice 3 × 3 "
                "appelée <b>K</b> :"),
              P("u = f<sub>x</sub> · X / Z + c<sub>x</sub>" + "&nbsp;" * 12 + "v = f<sub>y</sub> · Y / Z + c<sub>y</sub>",
                "formula"),
              P("f<sub>x</sub> et f<sub>y</sub> sont les focales en pixels (presque toujours égales : les pixels sont "
                "carrés), et (c<sub>x</sub>, c<sub>y</sub>) le centre de l'image. Diviser par Z, la profondeur, "
                "explique pourquoi les objets lointains paraissent plus petits."),
              P("Les <b>extrinsèques</b> disent où est la caméra : une rotation R et une translation t qui font "
                "passer un point du repère du monde au repère de la caméra. Dans le code, on les range dans une "
                "matrice 4 × 4 appelée <font name='Mono'>viewmat</font>. Convention OpenCV : x vers la droite, y vers "
                "le bas, z vers l'avant."),
              callout("Une erreur sur K ou sur [R | t], et tout le reste s'effondre : les gaussiennes s'entraînent "
                      "à de mauvais endroits, et les features de sens tombent à côté des objets. C'est pour ça qu'on "
                      "a vérifié les poses à chaque fois, par exemple en projetant les points 3D dans les images.",
                      BLUE, BLUE_BG, "Pourquoi c'est critique")]
    story += H2("2.2 Transformers et attention, en bref")
    story += [P("Les trois réseaux utilisés (VGGT, SAM, SigLIP 2) sont des <b>Vision Transformers</b> (ViT). "
                "L'image est découpée en petits carrés, les <i>patches</i> (14 × 14 ou 16 × 16 pixels), et chaque "
                "patch devient un vecteur, un <i>jeton</i>. Une couche d'<b>attention</b> permet à chaque jeton de "
                "« regarder » tous les autres et de mélanger leurs informations selon leur pertinence. Empiler des "
                "dizaines de couches donne une compréhension globale de l'image."),
              P("Un <b>embedding</b> est un vecteur (ici 768 nombres) qui résume le contenu d'une image ou d'un "
                "texte. Deux embeddings proches, au sens du <b>cosinus</b> de l'angle qui les sépare, désignent des "
                "choses proches. C'est la brique qui permet de relier les mots aux pixels.")]

    # ---------------------------------------------------------- 3. frames
    story += H1("3. Étape 1 — De la vidéo aux images")
    story += [P("Module <font name='Mono'>frames.py</font>. Une vidéo de 25 secondes contient ~750 images, très "
                "redondantes et souvent floues à cause du mouvement. On en garde 40 : la vidéo est découpée en 40 "
                "fenêtres, et dans chacune on prend l'image la plus nette."),
              figure(d_frames(), "Figure 3 — Sélection des images : une par fenêtre temporelle, la plus nette."),
              P("La netteté est mesurée par la <b>variance du Laplacien</b>. Le Laplacien est un filtre qui réagit "
                "aux variations brusques d'intensité, c'est-à-dire aux contours. Une image nette a des contours francs, "
                "donc un Laplacien très variable ; une image floue a des contours mous, donc une variance faible. "
                "Test : sur une vidéo où chaque image était dupliquée 6 fois dont 5 copies floutées, la fonction a "
                "retrouvé la copie nette 50 fois sur 50."),
              callout("Les téléphones enregistrent souvent la vidéo « couchée », avec une métadonnée qui indique de "
                      "la tourner de 90°. OpenCV 4.11 ignore cette métadonnée par défaut : les images sortaient "
                      "couchées. Une ligne a suffi à corriger (<font name='Mono'>CAP_PROP_ORIENTATION_AUTO</font>), "
                      "mais ce bug était invisible sur les jeux de photos publics : seule une vraie vidéo l'a révélé.",
                      RED, RED_BG, "Bug trouvé grâce à une vraie vidéo")]

    # ---------------------------------------------------------- 4. VGGT
    story += H1("4. Étape 2 — Les poses de caméra avec VGGT")
    story += [P("Module <font name='Mono'>poses.py</font>. Pour reconstruire la scène, il faut savoir d'où chaque "
                "photo a été prise : K et [R | t] pour chacune. L'approche classique est la <b>Structure from "
                "Motion</b>, avec le logiciel COLMAP : détecter des points caractéristiques dans chaque image, les "
                "apparier entre images, trianguler leur position 3D, puis ajuster toutes les caméras ensemble. C'est "
                "précis, mais lent et fragile."),
              P("<b>VGGT</b> (Visual Geometry Grounded Transformer, Meta, prix du meilleur papier au CVPR 2025) fait "
                "tout en une seule passe de réseau : on lui donne les images, il renvoie les caméras et une carte de "
                "profondeur par image, en quelques secondes."),
              figure(d_vggt(), "Figure 4 — L'architecture de VGGT. L'alternance entre attention intra-image et "
                               "attention globale est son idée centrale : les jetons échangent d'abord dans leur "
                               "image, puis avec toutes les autres, ce qui permet de mettre les vues en "
                               "correspondance."),
              P("La tête de profondeur donne, pour chaque pixel, sa distance d à la caméra et une confiance. En "
                "inversant la projection de la figure 2, on obtient un point 3D par pixel : X = d · K<super>-1</super> "
                "(u, v, 1), puis on le ramène dans le repère du monde avec [R | t]. On garde la moitié la plus sûre "
                "des points, avec au plus 200 000 points : ce nuage sert à initialiser les gaussiennes."),
              P("<b>Pourquoi VGGT plutôt que COLMAP ?</b> Au départ, c'était une contrainte : Windows (Smart App "
                "Control) bloque la bibliothèque compilée de pycolmap, alors que VGGT est du PyTorch pur. C'est "
                "devenu un atout : VGGT est l'état de l'art de 2025, et il marche même avec peu d'images.")]
    story += H2("4.1 Faire tenir VGGT dans 8 Go")
    story += [*bullets(["Poids en <b>fp16</b> (16 bits au lieu de 32) : la RTX 2070 (architecture Turing) ne gère "
                        "pas le format bf16 utilisé par défaut.",
                        "<b>Autocast</b> : PyTorch choisit automatiquement la précision des calculs internes.",
                        "Suppression des têtes inutiles (points 3D directs et suivi de points) : on reconstruit les "
                        "points depuis la profondeur.",
                        "Résultat : 50 images = 7,1 Go au maximum. Le réglage par défaut est 40 images, pour garder "
                        "de la marge."])]
    story += H2("4.2 Le bug des images portrait")
    story += [P("Premier essai sur des images portrait (1080 × 1920) : VGGT renvoyait f<sub>x</sub> ≈ 1048 et "
                "f<sub>y</sub> ≈ 1857, alors que ces deux valeurs devraient être presque égales. Explication : "
                "VGGT a surtout été entraîné sur des images paysage ou carrées, et il prédit le même champ de vision "
                "horizontal et vertical. La solution officielle est le mode « pad » : compléter l'image en carré."),
              figure(d_padding(), "Figure 5 — Le padding : l'image est réduite puis centrée dans un carré blanc. "
                                  "Après correction, f<sub>x</sub> / f<sub>y</sub> ≈ 1,00 sur toutes les vues."),
              P("Il faut ensuite ramener K dans le repère de l'image d'origine : retrancher le décalage du padding au "
                "centre optique, puis diviser par le facteur de réduction.")]

    # ---------------------------------------------------------- 5. 3DGS
    story += H1("5. Étape 3 — Le Gaussian Splatting")
    story += [P("Modules <font name='Mono'>gaussians.py</font> et <font name='Mono'>train.py</font>. Le <b>3D "
                "Gaussian Splatting</b> (Kerbl et al., SIGGRAPH 2023) représente une scène par des centaines de "
                "milliers de petits ellipsoïdes flous et colorés, les gaussiennes. On les « projette » (en anglais "
                "<i>splat</i>) sur l'image pour la dessiner."),
              figure(d_gaussian(), "Figure 6 — Une gaussienne et ses paramètres. La taille est stockée en log et "
                                   "l'opacité en logit : l'optimiseur peut ainsi les faire varier librement sans "
                                   "jamais produire une taille négative ou une opacité hors de [0, 1]."),
              P("La couleur est stockée en <b>harmoniques sphériques</b> (SH) : 16 coefficients par canal RVB, qui "
                "permettent à la couleur de changer selon l'angle de vue (reflets, brillance). Le premier coefficient "
                "porte la couleur moyenne ; c'est lui qu'on modifie pour surligner un objet.")]
    story += H2("5.1 Dessiner une image : le mélange alpha")
    story += [P("Pour dessiner un pixel, on lance un rayon depuis la caméra. On trie les gaussiennes qu'il traverse "
                "de la plus proche à la plus lointaine, puis on mélange leurs couleurs. Chaque gaussienne cache en "
                "partie celles qui sont derrière elle."),
              figure(d_render(), "Figure 7 — Le mélange alpha le long d'un rayon. La transmittance T est la part de "
                                 "lumière qui arrive encore jusqu'à la gaussienne ; son poids final est w = α · T."),
              callout("Retenez le poids w<sub>ip</sub> : la part du pixel p qui vient de la gaussienne i. Le rendu "
                      "est une somme de couleurs pondérée par ces poids, donc il est <b>linéaire</b> en les couleurs. "
                      "C'est exactement cette propriété qu'exploite l'astuce du chapitre 7.", VIOLET, VIOLET_BG,
                      "Le détail qui servira plus tard"),
              P("En pratique, la bibliothèque <b>gsplat</b> fait tout ce calcul sur GPU en CUDA : projection de "
                "chaque ellipsoïde en ellipse 2D, tri, mélange. Surtout, elle est <b>différentiable</b> : elle sait "
                "calculer comment chaque paramètre de chaque gaussienne a influencé chaque pixel.")]
    story += H2("5.2 L'entraînement")
    story += [figure(d_training(), "Figure 8 — La boucle d'entraînement. On part des points de VGGT : une "
                                   "gaussienne par point, de la taille de son voisinage et d'opacité 0,1."),
              P("La perte combine la <b>L1</b> (écart moyen pixel à pixel) et la <b>SSIM</b> (similarité de "
                "structure, sensible aux textures et aux contours). L'optimiseur <b>Adam</b> ajuste chaque famille "
                "de paramètres avec son propre pas d'apprentissage ; celui des positions diminue jusqu'à 1 % de sa "
                "valeur initiale. Les couleurs dépendant de l'angle sont débloquées progressivement, un degré de SH "
                "tous les 1 000 pas."),
              P("La <b>densification</b> est essentielle : là où l'image rendue est mauvaise, le gradient est fort. "
                "On y clone les petites gaussiennes et on coupe en deux les grandes. On supprime les gaussiennes "
                "devenues transparentes, et on remet régulièrement les opacités à une valeur basse pour éliminer les "
                "« floaters », ces gaussiennes parasites qui flottent dans le vide."),
              P("<b>Qualité.</b> On la mesure en <b>PSNR</b> = 10 · log<sub>10</sub>(1 / erreur quadratique "
                "moyenne), en décibels : plus c'est haut, mieux c'est, et 30 dB est déjà très bon. Scène fox : 29,7 dB "
                "en 3 min 43 s, 419 000 gaussiennes. Vidéo de la gourde : 35,6 dB en 5 min 17 s, 797 000 gaussiennes. "
                "Attention : ces valeurs sont mesurées sur les vues d'entraînement, donc flatteuses."),
              photo(ROOT / "docs/rendu_vs_photo.jpg", "Figure 9 — Scène fox. Pour chaque paire : photo d'origine à "
                                                       "gauche, rendu des gaussiennes à droite.", CW * 0.72)]

    # ---------------------------------------------------------- 6. sens
    story += H1("6. Étape 4 — Donner du sens : SAM et SigLIP 2")
    story += [P("Modules <font name='Mono'>features.py</font> et <font name='Mono'>encoder.py</font>. Les "
                "gaussiennes savent à quoi la scène ressemble, pas ce qu'elle contient. On s'appuie pour ça sur deux "
                "réseaux pré-entraînés, utilisés tels quels."),
              figure(d_sam_siglip(), "Figure 10 — Des pixels au sens. SAM isole les objets ; chaque objet découpé "
                                     "est transformé en vecteur par SigLIP 2, dans le même espace que les textes."),
              P("<b>SAM</b> (Segment Anything, Meta, 2023) découpe une image en masques d'objets et de parties "
                "d'objets, sans savoir ce qu'ils sont. On l'interroge avec une grille de 32 × 32 points et on garde "
                "les masques stables et de bonne qualité, environ 83 par image."),
              P("Chaque masque est isolé : on découpe sa boîte englobante, on met en noir ce qui n'appartient pas au "
                "masque, et on complète en carré de 224 pixels. <b>SigLIP 2</b> (Google, 2025) transforme ce crop en "
                "un vecteur de 768 nombres. SigLIP 2 a été entraîné sur des milliards de paires image-texte pour "
                "rapprocher une image de sa légende : son espace d'embeddings est <b>commun aux images et aux "
                "textes</b>."),
              figure(d_space(), "Figure 11 — L'espace des embeddings. Un crop de renard, le mot « renard » et le "
                                "mot « fox » se retrouvent au même endroit."),
              P("Pourquoi SigLIP 2 plutôt que CLIP, utilisé par LERF et LangSplat ? Il est plus récent, plus "
                "précis, et surtout <b>multilingue</b> : on peut taper « pomme verte » aussi bien que « green "
                "apple ».")]
    story += H2("6.1 Deux niveaux de masques")
    story += [P("Un pixel de l'oreille du renard appartient à plusieurs masques à la fois : « oreille », "
                "« tête », « renard ». Lequel lui associer ? LangSplat garde trois niveaux ; lang3d en garde deux, "
                "partagés avec la même table d'embeddings :"),
              figure(d_levels(), "Figure 12 — Le même pixel, deux niveaux. On stocke une carte d'index par niveau "
                                 "(chaque pixel pointe vers un masque) et une seule table d'embeddings par image."),
              P("SAM ne découpe pas les grands fonds texturés comme le papier peint ou le sol. Ces zones non "
                "couvertes sont ajoutées comme masques supplémentaires quand elles dépassent 1 % de l'image. Stocker "
                "des index et une table plutôt qu'un vecteur par pixel est des centaines de fois plus léger.")]

    # ---------------------------------------------------------- 7. lifting
    story += H1("7. Étape 5 — Du 2D au 3D sans entraînement")
    story += [P("Module <font name='Mono'>lift.py</font>. On a maintenant des vecteurs de sens <i>par pixel</i>, "
                "dans 40 à 300 images. Il faut un vecteur <i>par gaussienne</i>. Idée naturelle : prendre la moyenne "
                "des vecteurs des pixels auxquels chaque gaussienne contribue, pondérée par sa contribution "
                "w<sub>ip</sub>. Mais calculer ces poids un par un, pour 400 000 gaussiennes et des millions de "
                "pixels, serait très lourd."),
              figure(d_lifting(), "Figure 13 — L'astuce du gradient. Un seul rendu et une seule rétropropagation par "
                                  "image donnent la somme pondérée pour toutes les gaussiennes à la fois."),
              P("L'astuce : on donne à chaque gaussienne une « couleur » c<sub>i</sub> à 129 canaux, toute à zéro, et "
                "on rend l'image. Puisque le rendu est linéaire en c (chapitre 5.1), dériver le produit scalaire "
                "entre l'image rendue et la carte de sens F donne exactement Σ<sub>p</sub> w<sub>ip</sub> · F(p) "
                "pour chaque gaussienne. Le 129<super>e</super> canal, rempli de 1, donne Σ<sub>p</sub> "
                "w<sub>ip</sub> : il suffit de diviser."),
              table([["", "LangSplat", "lang3d"],
                     ["Passage 2D → 3D", "entraînement des features (30 000 itérations)", "1 rendu + 1 retour par image"],
                     ["Temps", "≈ 25 min sur RTX 3090 (d'après le papier)", "≈ 8 s par niveau (fox, 50 images)"],
                     ["Taille des vecteurs", "3 dims (autoencodeur)", "128 dims (SVD)"]],
                    [CW * 0.28, CW * 0.38, CW * 0.34]),
              Spacer(1, 8)]
    story += H2("7.1 Compresser sans fausser les requêtes")
    story += [P("Stocker 768 nombres par gaussienne coûterait cher. On calcule une base de 128 directions par "
                "<b>SVD</b> (décomposition en valeurs singulières) sur les embeddings de la scène, et on projette "
                "tout dessus."),
              figure(d_svd(), "Figure 14 — La compression par SVD non centrée."),
              P("Le détail « non centrée » compte : sans retrancher la moyenne, la projection conserve les produits "
                "scalaires entre un texte et une image, tant que l'image est bien représentée dans la base. Or une "
                "requête est justement un produit scalaire. L'autoencodeur à 3 dimensions de LangSplat n'a pas cette "
                "garantie.")]

    # ---------------------------------------------------------- 8. requête
    story += H1("8. Étape 6 — Interroger la scène")
    story += [P("Modules <font name='Mono'>query.py</font> et <font name='Mono'>viewer.py</font>. Quand on tape "
                "une requête :"),
              *bullets(["<b>Nettoyage</b> : « Où est la chaise rouge ? » devient « chaise rouge ». Une expression "
                        "régulière retire « où est », « montre-moi », les articles et la ponctuation.",
                        "<b>Encodage</b> : SigLIP 2 transforme le texte en vecteur, projeté dans la base de 128 "
                        "dimensions.",
                        "<b>Pertinence</b> de chaque gaussienne, façon LERF (figure 15), pour chacun des deux niveaux ; "
                        "on garde le niveau dont la meilleure gaussienne est la plus pertinente.",
                        "<b>Surlignage</b> : au-dessus du seuil, la couleur de base (premier coefficient SH) vire au "
                        "rouge ; le reste passe en gris."]),
              figure(d_relevance(), "Figure 15 — Le score de pertinence. Un cosinus seul ne veut rien dire : on le "
                                    "compare à celui de mots génériques, les « négatifs »."),
              P("Le score final est renormalisé entre 0 (pas mieux qu'un négatif) et 1 (la gaussienne la plus "
                "pertinente de la scène), et le curseur « Seuil » du viewer agit sur cette valeur. L'option "
                "<b>Objet entier</b> moyenne trois formulations (« {} », « une photo de {} », « a photo of {} ») : "
                "elle aide pour les objets entiers mais noie les parties d'objets, d'où sa désactivation par défaut."),
              P("<b>Le viewer</b> utilise viser : un serveur Python envoie les gaussiennes au navigateur, qui les "
                "dessine lui-même en WebGL. À chaque requête, seules les couleurs et les opacités sont renvoyées. Une "
                "requête prend 0,02 à 0,3 s."),
              photo(ROOT / "docs/viewer_renard.jpg", "Figure 16 — Le viewer web : « Où est le renard ? ».", CW * 0.55)]

    # ---------------------------------------------------------- 9. code
    story += H1("9. L'architecture du code")
    story += [figure(d_code(), "Figure 17 — Les modules de <font name='Mono'>lang3d/</font>. Chaque commande de "
                               "la ligne de commande correspond à un module."),
              table([["Commande", "Lit", "Écrit"],
                     ["<font name='Mono'>frames</font>", "vidéo", "<font name='Mono'>images/*.jpg</font>"],
                     ["<font name='Mono'>poses</font> / <font name='Mono'>colmap</font>", "images ou sparse/0",
                      "<font name='Mono'>poses.npz</font> (K, viewmats, points)"],
                     ["<font name='Mono'>train</font>", "images, poses", "<font name='Mono'>gs/splats.pt</font>, "
                                                                        "<font name='Mono'>gs/scene.ply</font>"],
                     ["<font name='Mono'>features</font>", "images, poses", "<font name='Mono'>lang/features.pt</font>"],
                     ["<font name='Mono'>lift</font>", "splats, features",
                      "<font name='Mono'>lang/gaussians_lang.pt</font>"],
                     ["<font name='Mono'>view</font> / <font name='Mono'>query</font> / "
                      "<font name='Mono'>demo</font>", "tout", "viewer web, .ply surligné, GIF"],
                     ["<font name='Mono'>evaluate</font>", "tout + annotations", "<font name='Mono'>eval_results.json</font>"]],
                    [CW * 0.27, CW * 0.27, CW * 0.46]),
              Spacer(1, 10),
              P("Chaque étape s'arrête sur un fichier : on peut relancer n'importe quelle étape sans refaire les "
                "précédentes, et inspecter chaque résultat intermédiaire. Le pipeline complet sur une scène :"),
              code("python -m lang3d frames   scenes/salon --video salon.mp4\n"
                   "python -m lang3d poses    scenes/salon\n"
                   "python -m lang3d train    scenes/salon\n"
                   "python -m lang3d features scenes/salon\n"
                   "python -m lang3d lift     scenes/salon\n"
                   "python -m lang3d view     scenes/salon      # http://localhost:8080")]

    # ---------------------------------------------------------- 10. résultats
    story += H1("10. Les résultats")
    story += H2("10.1 Les scènes de test")
    story += [photo(ROOT / "docs/requetes_fox.jpg", "Figure 18 — Scène fox, requêtes en français. En rouge, les "
                                                    "gaussiennes au-dessus du seuil.", CW),
              table([["Scène", "Réussi", "Raté"],
                     ["fox (photos publiques)", "« renard », « oreille du renard », « rideau rouge » (option objet "
                                                "entier), « cadre photo »", "« papier peint à fleurs », « plaque en "
                                                                           "bois »"],
                     ["gourde (vidéo personnelle, non publiée)", "« gourde violette » (isolée parfaitement), "
                                                                 "« chaussures », « couvercle »",
                      "« carrelage », « parquet », « mur blanc », « bouchon »"]],
                    [CW * 0.26, CW * 0.44, CW * 0.30]),
              Spacer(1, 6),
              P("Le choix des mots compte : « bouchon » échoue, SigLIP 2 le comprend plutôt comme un bouchon de liège, "
                "alors que « couvercle » réussit.")]
    story += H2("10.2 Le benchmark LERF : une mesure objective")
    story += [P("Des exemples choisis ne prouvent rien. Pour une mesure objective, on utilise le benchmark du "
                "domaine : le dataset <b>LERF</b>, annoté par les auteurs de LangSplat (« lerf_ovs »). Sur quelques "
                "vues de chaque scène, chaque objet est détouré à la main et nommé (« green apple », « rubics "
                "cube »…). Le protocole d'évaluation de LangSplat a été <b>reproduit à l'identique</b> : leur code a "
                "été lu ligne à ligne, et le lissage des masques donne des résultats identiques à 100 % sur des tests "
                "aléatoires. On utilise aussi les poses COLMAP fournies par le dataset, comme eux."),
              figure(d_eval(), "Figure 19 — Le protocole d'évaluation : deux scores calculés pour chaque couple "
                               "(vue annotée, requête)."),
              table([["Scène figurines (56 requêtes)", "Localisation", "IoU moyen"],
                     ["LERF (ICCV 2023)", "75,0 %", "38,6 %"],
                     ["LangSplat (CVPR 2024)", "80,4 %", "44,7 %"],
                     ["<b>lang3d</b>", "<b>83,9 %</b>", "<b>49,0 %</b>"],
                     ["lang3d, option objet entier (indicatif)", "83,9 %", "52,9 %"],
                     ["Contrôle : requêtes mélangées", "1,8 %", "1,7 %"]],
                    [CW * 0.5, CW * 0.25, CW * 0.25]),
              Spacer(1, 8),
              photo(ROOT / "docs/lerf_figurines.jpg", "Figure 20 — Trois requêtes sur figurines. Contour vert : "
                                                      "annotation manuelle. Rouge : prédiction de lang3d. Images : "
                                                      "dataset LERF.", CW),
              figure(d_iou_chart(), "Figure 21 — IoU moyen par objet sur figurines. Les échecs sont des petits "
                                    "objets, des noms propres ou de marque (Miffy, Tesla) et des objets mal séparés "
                                    "de leur voisinage.")]
    story += H2("10.3 Comment lire ces chiffres honnêtement")
    story += [*bullets(["<b>Le contrôle</b> est la preuve la plus importante : en échangeant les noms entre les "
                        "objets, les scores tombent à ~2 %. L'évaluation mesure donc bien la compréhension des "
                        "requêtes, et non un biais du protocole.",
                        "<b>Une seule scène sur quatre</b> a été évaluée. On compare aux chiffres de LERF et "
                        "LangSplat sur la même scène, pas à leur moyenne.",
                        "<b>Le chiffre principal utilise les réglages par défaut</b>, fixés avant l'évaluation. "
                        "Choisir l'option objet entier après avoir vu les scores reviendrait à régler la méthode sur "
                        "le jeu de test.",
                        "<b>Les conditions diffèrent</b> : LangSplat utilise une RTX 3090, 30 000 itérations, SAM "
                        "ViT-H et CLIP ; lang3d une RTX 2070, 7 000 itérations, SAM ViT-B et SigLIP 2. Le gain vient "
                        "probablement en partie de SigLIP 2, plus récent que CLIP."])]

    # ---------------------------------------------------------- 11. essais
    story += H1("11. Ce qui a été essayé")
    story += [P("Chaque idée a été mesurée avant d'être gardée, et les échecs sont documentés autant que les "
                "succès. C'est souvent ce qui intéresse le plus en entretien."),
              table([["Essai", "Résultat mesuré", "Décision"],
                     ["Un seul niveau de masques", "parties OK, grands objets morcelés", "→ deux niveaux"],
                     ["Deux niveaux small / large", "« cadre photo », « rideau » passent au niveau large",
                      "<font color='%s'><b>gardé</b></font>" % hexs(GREEN)],
                     ["Zones non couvertes par SAM", "le fond (29 % de l'image) a enfin une feature",
                      "<font color='%s'><b>gardé</b></font>" % hexs(GREEN)],
                     ["Carrés intérieurs pour les fonds", "aucun gain", "<font color='%s'><b>retiré</b></font>"
                      % hexs(RED)],
                     ["Ensemble de prompts", "objets : 168 → 1 080 gaussiennes ; parties : tout le renard",
                      "<font color='%s'><b>en option</b></font>" % hexs(AMBER)],
                     ["Retirer le négatif « texture »", "aucun gain visible sur les surfaces",
                      "<font color='%s'><b>non intégré</b></font>" % hexs(RED)]],
                    [CW * 0.32, CW * 0.46, CW * 0.22])]

    # ---------------------------------------------------------- 12. ingénierie
    story += H1("12. Les galères d'ingénierie")
    story += [P("Une bonne partie du travail n'a rien à voir avec la recherche : faire tourner des outils de "
                "pointe sur un PC Windows standard."),
              table([["Problème", "Cause", "Solution"],
                     ["PyTorch ne charge pas ses DLL", "chemin du dossier > 260 caractères (limite Windows)",
                      "projet déplacé dans Documents\\lang3d"],
                     ["Pas de compilateur CUDA", "CUDA Toolkit et Visual Studio absents",
                      "version précompilée de gsplat → Python 3.10, PyTorch 2.4"],
                     ["pycolmap bloqué", "Smart App Control refuse la DLL non signée",
                      "VGGT (PyTorch pur) + lecteur COLMAP en Python pur"],
                     ["transformers désactive PyTorch", "les versions récentes exigent PyTorch ≥ 2.6",
                      "version 4.49 épinglée"],
                     ["Accents illisibles dans les logs", "console Windows en cp1252", "sortie forcée en UTF-8"],
                     ["fx ≠ fy sur images portrait", "VGGT entraîné surtout en paysage", "mode pad (figure 5)"],
                     ["Frames couchées", "métadonnée de rotation ignorée par OpenCV",
                      "CAP_PROP_ORIENTATION_AUTO"],
                     ["Script .sh cassé une fois cloné", "fins de ligne converties en CRLF",
                      ".gitattributes avec eol=lf"],
                     ["Gmail visible sur GitHub", "adresse dans les métadonnées des commits",
                      "historique réécrit avec l'adresse noreply, repo recréé"]],
                    [CW * 0.3, CW * 0.36, CW * 0.34]),
              Spacer(1, 10)]
    story += H2("12.1 « Je n'ai pas CUDA ? » — si, en partie")
    story += [figure(d_cuda(), "Figure 22 — Les trois étages de CUDA. Pour exécuter du code GPU, les deux du bas "
                               "suffisent ; le Toolkit ne sert qu'à compiler."),
              P("Le PyTorch installé (<font name='Mono'>torch 2.4.1+cu124</font>) embarque ses propres bibliothèques "
                "CUDA 12.4. Le Toolkit (le compilateur <font name='Mono'>nvcc</font>) ne servirait qu'à compiler "
                "gsplat ou d'autres bibliothèques depuis leurs sources.")]

    # ---------------------------------------------------------- 13. limites
    story += H1("13. Limites et pistes")
    story += [table([["Limite", "Piste"],
                     ["~40-50 vues maximum sur 8 Go (VGGT)", "traiter la vidéo par blocs qui se chevauchent et les "
                                                             "aligner entre eux"],
                     ["PSNR mesuré sur les vues d'entraînement", "garder 1 image sur 8 de côté pour mesurer la "
                                                                 "vraie qualité"],
                     ["« floaters » loin des vues filmées", "ajouter la profondeur de VGGT à la perte d'entraînement"],
                     ["surfaces texturées (sol, mur, papier peint)", "segmenter les grands plans à partir de la "
                                                                      "profondeur"],
                     ["requêtes courtes sans contexte (« les oreilles »)", "compléter avec l'objet le plus "
                                                                           "pertinent de la scène"],
                     ["1 scène évaluée sur 4", "évaluer ramen, teatime et waldo_kitchen"],
                     ["installation lourde (Python 3.10, GPU)", "démo web statique avec requêtes précalculées"]],
                    [CW * 0.45, CW * 0.55])]

    # ---------------------------------------------------------- 14. entretien
    story += H1("14. En parler en entretien")
    story += [callout("« J'ai construit un pipeline qui transforme une vidéo de téléphone en scène 3D interrogeable "
                      "en langage naturel : on tape \u00ab où est la gourde ? \u00bb et l'objet s'allume en 3D. Il "
                      "combine trois modèles récents : VGGT pour les caméras, le Gaussian Splatting pour la 3D, SAM "
                      "et SigLIP 2 pour le sens. J'ai remplacé l'entraînement des features de LangSplat par une "
                      "astuce de gradient, qui fait le passage 2D → 3D en quelques secondes. Sur le benchmark de "
                      "référence, j'obtiens 83,9 % de localisation sur la scène figurines, au-dessus de LangSplat, "
                      "avec une carte graphique grand public. »", BLUE, BLUE_BG, "Le pitch en 30 secondes"),
              table([["Question probable", "Éléments de réponse"],
                     ["Comment savez-vous que ça marche ?", "benchmark LERF avec le protocole officiel, et un "
                                                           "contrôle à requêtes mélangées qui tombe à ~2 %"],
                     ["Pourquoi VGGT plutôt que COLMAP ?", "contrainte Windows au départ ; en plus, c'est l'état de "
                                                           "l'art de 2025, rapide, et il marche avec peu d'images"],
                     ["Quelle est votre contribution ?", "l'astuce du gradient pour le passage 2D → 3D, la SVD non "
                                                        "centrée, l'adaptation à 8 Go, les deux niveaux, le "
                                                        "multilingue"],
                     ["Qu'est-ce qui n'a pas marché ?", "les surfaces texturées, avec la cause mesurée : la "
                                                       "similarité ne dépasse pas celle du négatif « texture »"],
                     ["Que feriez-vous ensuite ?", "évaluer les 4 scènes, mesurer sur des vues mises de côté, "
                                                   "faire une démo web"]],
                    [CW * 0.36, CW * 0.64])]

    # ---------------------------------------------------------- 15. glossaire
    story += H1("15. Glossaire")
    gl = [("Alpha (α)", "opacité d'une gaussienne, entre 0 (invisible) et 1 (opaque)."),
          ("Autocast", "PyTorch choisit automatiquement la précision (16 ou 32 bits) de chaque calcul."),
          ("Cosinus", "mesure de proximité entre deux vecteurs : 1 = même direction, 0 = sans rapport."),
          ("Embedding", "vecteur de nombres qui résume le contenu d'une image ou d'un texte."),
          ("Extrinsèques", "position et orientation de la caméra : [R | t]."),
          ("Feed-forward", "réseau qui donne sa réponse en une seule passe, sans optimisation itérative."),
          ("fp16 / bf16", "nombres sur 16 bits : moitié moins de mémoire que le 32 bits."),
          ("Gaussienne", "ellipsoïde flou et coloré ; la brique du Gaussian Splatting."),
          ("Harmoniques sphériques (SH)", "façon de coder une couleur qui varie selon l'angle de vue."),
          ("Intrinsèques (K)", "focales et centre optique : comment la caméra transforme des rayons en pixels."),
          ("IoU", "intersection sur union : recouvrement entre deux zones, de 0 à 1."),
          ("Lifting", "faire passer une information des pixels 2D vers les éléments 3D."),
          ("Négatifs", "mots génériques (« object », « stuff »…) qui servent de référence au score de pertinence."),
          ("Patch / jeton", "petit carré d'image transformé en vecteur à l'entrée d'un transformer."),
          ("PSNR", "mesure de fidélité d'une image en décibels ; plus c'est haut, mieux c'est."),
          ("SVD", "décomposition qui trouve les directions principales d'un ensemble de vecteurs."),
          ("Transmittance (T)", "part de lumière qui n'a pas encore été arrêtée le long d'un rayon."),
          ("VRAM", "mémoire de la carte graphique (8 Go sur une RTX 2070).")]
    story += [table([[f"<b>{a}</b>", b] for a, b in gl], [CW * 0.3, CW * 0.7], header=False)]

    doc = Doc(OUT)
    doc.multiBuild(story)
    print(f"→ {OUT}")


if __name__ == "__main__":
    build()

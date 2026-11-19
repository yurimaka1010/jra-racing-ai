"""Render anonymous aggregate learning curves with bundled ReportLab charts.

Run with the existing desktop dependency Python; installs no packages.
"""
import argparse
import json
import math
from pathlib import Path

from reportlab.graphics.shapes import Drawing, String, Line
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.widgets.markers import makeMarker
from reportlab.graphics import renderSVG, renderPDF
from reportlab.lib.colors import HexColor
import pypdfium2 as pdfium


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    path = args.run.resolve(strict=True)
    root = Path(r"C:\Users\acbd5\AppData\Local\JraRacingAI\ml-step2").resolve()
    if root not in path.parents or ".." in args.run.parts:
        raise SystemExit("unsafe_path")
    result = json.loads(path.read_text(encoding="utf-8-sig"))
    entries = [entry for entry in result["models"] if entry["experiment"] == "controlled_learning_size"]
    sizes = [entry["training_races"] for entry in entries]
    old = result["saved_v03"]["primary"]
    drawing = Drawing(1180, 440)
    drawing.add(String(40, 412, "STEP 2 V2: controlled nested training windows", fontName="Helvetica-Bold", fontSize=18))
    part = result["partitions"]["test"]
    drawing.add(String(40, 391, f"Same {part['races']} test races / {part['rows']} runners, {part['start']} to {part['end']}", fontSize=11))
    colors = [HexColor("#1769aa"), HexColor("#d76800"), HexColor("#7b536f"), HexColor("#777777")]
    labels = ["Fixed T=2.0", "Independent recalibration", "Saved V0.3 T=2", "Uniform"]
    for i, label in enumerate(labels):
        x = 40 + i * 270
        drawing.add(Line(x, 366, x + 24, 366, strokeColor=colors[i], strokeWidth=2))
        drawing.add(String(x + 30, 362, label, fontSize=10))
    for panel, (metric, title) in enumerate((("log_loss", "Horse-binary Log Loss (lower is better)"),
                                           ("brier_score", "Brier Score (lower is better)"),
                                           ("top1_accuracy_percent", "Top-1 accuracy % (higher is better)"))):
        x = 63 + panel * 375
        drawing.add(String(x - 15, 331, title, fontName="Helvetica-Bold", fontSize=11))
        plot = LinePlot()
        plot.x, plot.y, plot.width, plot.height = x, 88, 302, 220
        series = [[(math.log10(entry["training_races"]), entry["series"][name]["primary"]["model"][metric])
                   for entry in entries] for name in ("fixed", "recalibrated")]
        series += [[(math.log10(sizes[0]), old[key][metric]), (math.log10(sizes[-1]), old[key][metric])]
                   for key in ("model", "uniform")]
        plot.data = series
        plot.xValueAxis.valueMin = math.log10(sizes[0]) - .04
        plot.xValueAxis.valueMax = math.log10(sizes[-1]) + .04
        plot.xValueAxis.valueSteps = [math.log10(n) for n in sizes]
        plot.xValueAxis.labelTextFormat = lambda value: f"{10 ** value:,.0f}"
        plot.xValueAxis.labels.fontSize = 7.5
        values = [value for items in series for _, value in items]
        span = max(values) - min(values)
        plot.yValueAxis.valueMin = max(0, min(values) - max(span * .15, .0002))
        plot.yValueAxis.valueMax = max(values) + max(span * .15, .0002)
        plot.yValueAxis.labels.fontSize = 8
        plot.yValueAxis.visibleGrid = True
        plot.yValueAxis.gridStrokeColor = HexColor("#e6ebef")
        for i in range(4):
            plot.lines[i].strokeColor = colors[i]
            plot.lines[i].strokeWidth = 1.8 if i < 2 else 1
            if i < 2:
                plot.lines[i].symbol = makeMarker("FilledCircle" if i == 0 else "FilledSquare")
                plot.lines[i].symbol.size = 5
            else:
                plot.lines[i].strokeDashArray = [4, 3]
        drawing.add(plot)
        drawing.add(String(x + 65, 56, "Training races (log scale)", fontSize=10))
    drawing.add(String(40, 29, "100-round primary comparison. The conditional 300-round reference is tabulated separately.", fontSize=9))
    drawing.add(String(40, 13, "No betting results or model deployment are inferred from these curves. Market comparison uses matching odds cohorts in the report.", fontSize=8))
    svg = path.parent / "learning_curve.svg"
    png = path.parent / "learning_curve.png"
    if svg.exists() or png.exists():
        raise SystemExit("artifact_exists")
    renderSVG.drawToFile(drawing, str(svg))
    document = pdfium.PdfDocument(renderPDF.drawToString(drawing))
    page = document[0]
    bitmap = page.render(scale=2)
    bitmap.to_pil().save(png)
    bitmap.close()
    page.close()
    document.close()
    svg.chmod(0o444)
    png.chmod(0o444)
    print(json.dumps({"status":"ok","plot_svg":str(svg),"plot_png":str(png)}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate test PDFs for benchmarking."""

import random
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

random.seed(42)


def create_single_page_table_pdf(output_path: str):
    """Create a PDF with a single-page table (10 rows)."""
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("Single Page Table Test", styles['Heading1']))
    elements.append(Spacer(1, 12))

    data = [["ID", "Name", "Value", "Category"]]
    for i in range(10):
        data.append([
            str(i + 1),
            f"Item {i + 1}",
            f"${random.randint(100, 999)}",
            random.choice(["A", "B", "C"])
        ])

    table = Table(data)
    table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ]))
    elements.append(table)

    doc.build(elements)
    print(f"  Created: {output_path}")


def create_multi_page_table_pdf(output_path: str):
    """Create a PDF with a multi-page table (100 rows, ~3 pages)."""
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("Multi-Page Table Test", styles['Heading1']))
    elements.append(Paragraph("This table spans multiple pages.", styles['Normal']))
    elements.append(Spacer(1, 12))

    data = [["ID", "Name", "Value", "Category", "Status", "Notes"]]
    for i in range(100):
        data.append([
            str(i + 1),
            f"Item {i + 1}",
            f"${random.randint(100, 9999)}",
            random.choice(["A", "B", "C", "D"]),
            random.choice(["Active", "Inactive", "Pending"]),
            f"Note {i + 1}"
        ])

    table = Table(data)
    table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
    ]))
    elements.append(table)

    doc.build(elements)
    print(f"  Created: {output_path}")


def create_mixed_content_pdf(output_path: str):
    """Create a PDF with prose, tables, and lists."""
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("Mixed Content Document", styles['Heading1']))
    elements.append(Spacer(1, 12))

    # Prose
    elements.append(Paragraph("Introduction", styles['Heading2']))
    elements.append(Paragraph(
        "This document contains a mix of prose, tables, and lists for testing "
        "document extraction pipelines. The content is designed to exercise "
        "various extraction capabilities including text, structured data, and lists.",
        styles['Normal']
    ))
    elements.append(Spacer(1, 12))

    # Table
    elements.append(Paragraph("Data Table", styles['Heading2']))
    data = [["Metric", "Q1", "Q2", "Q3", "Q4"]]
    for i in range(5):
        data.append([
            f"Metric {i + 1}",
            str(random.randint(100, 500)),
            str(random.randint(100, 500)),
            str(random.randint(100, 500)),
            str(random.randint(100, 500))
        ])
    table = Table(data)
    table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    # More prose
    elements.append(Paragraph("Analysis", styles['Heading2']))
    elements.append(Paragraph(
        "The table above shows quarterly metrics for various measurements. "
        "Key observations include the consistent growth in Metric 1 and the "
        "seasonal variation visible in Metric 3.",
        styles['Normal']
    ))
    elements.append(Spacer(1, 12))

    doc.build(elements)
    print(f"  Created: {output_path}")


def create_formula_heavy_pdf(output_path: str):
    """Create a PDF with mathematical formulas using fpdf2 for Unicode support."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("ArialUni", fname="/Library/Fonts/Arial Unicode.ttf")
    pdf.set_font("ArialUni", size=16)
    pdf.cell(text="Formula Heavy Document", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("ArialUni", size=12)
    pdf.cell(text="Mathematical Formulas:", ln=True)
    pdf.ln(5)

    formulas = [
        "Quadratic Formula: x = (-b ± √(b² - 4ac)) / 2a",
        "Euler's Identity: e^(iπ) + 1 = 0",
        "Pythagorean Theorem: a² + b² = c²",
        "Newton's Second Law: F = ma",
        "Einstein's Mass-Energy: E = mc²",
        "Black-Scholes: C = S·N(d1) - K·e^(-rT)·N(d2)",
        "Normal Distribution: f(x) = (1/σ√(2π))·e^(-(x-μ)²/2σ²)",
        "Bayes' Theorem: P(A|B) = P(B|A)·P(A) / P(B)",
    ]

    for formula in formulas:
        pdf.cell(text=f"• {formula}", ln=True)
        pdf.ln(3)

    pdf.ln(10)
    pdf.cell(text="Integral and Sum Notation:", ln=True)
    pdf.ln(5)

    more_formulas = [
        "Integral: ∫ f(x)dx = F(b) - F(a)",
        "Sum: Σ(i=1 to n) x_i = x₁ + x₂ + ... + xₙ",
        "Product: Π(i=1 to n) x_i = x₁ · x₂ · ... · xₙ",
        "Limit: lim(x→a) f(x) = L",
    ]

    for formula in more_formulas:
        pdf.cell(text=f"• {formula}", ln=True)
        pdf.ln(3)

    pdf.output(output_path)
    print(f"  Created: {output_path}")


def main():
    """Generate all test PDFs."""
    output_dir = Path(__file__).parent.parent / "data" / "sample_pdfs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating test PDFs...")

    create_single_page_table_pdf(str(output_dir / "single_page_table.pdf"))
    create_multi_page_table_pdf(str(output_dir / "multi_page_table.pdf"))
    create_mixed_content_pdf(str(output_dir / "mixed_content.pdf"))
    create_formula_heavy_pdf(str(output_dir / "formula_heavy.pdf"))

    print(f"\n✅ All PDFs generated in {output_dir}")


if __name__ == "__main__":
    main()
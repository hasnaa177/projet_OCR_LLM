"""
Synthetic loan-application generator for the credit-ocr-system project.

For each generated application it produces TWO paired artifacts:
  1. a PDF that mimics the original `data/loan_application.pdf` layout
  2. a ground-truth JSON holding the exact field values used to build that PDF

The ground-truth file is what lets you later measure extraction accuracy:
compare what the OCR + LLM pipeline pulls out against these known-correct values.

Usage:
    pip install faker reportlab
    python generate_fake_loan_data.py --count 25 --outdir ./synthetic_data

Optional (adds messy "scanned" versions for stress-testing OCR):
    pip install pdf2image pillow   # pdf2image needs poppler installed on the OS
    python generate_fake_loan_data.py --count 25 --degrade
"""

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

from faker import Faker
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

fake = Faker("de_DE")  # German locale to match the original sample's style

# ---- colours / styling to match the original form -------------------------
HEADER_BLUE = colors.HexColor("#2E5C8A")
BORDER_GREY = colors.HexColor("#333333")


# ---------------------------------------------------------------------------
# 1. Build one record of random-but-internally-consistent values
# ---------------------------------------------------------------------------
def make_record() -> dict:
    """Create a single fake loan application as a flat dict of field values."""
    legal_forms = [
        "Limited Liability Company (GmbH)",
        "Stock Corporation (AG)",
        "Sole Proprietorship",
        "Limited Partnership (KG)",
        "Entrepreneurial Company (UG)",
    ]
    property_types = [
        "Office and Commercial Building",
        "Retail Property",
        "Warehouse / Logistics Center",
        "Mixed-Use Building",
        "Industrial Facility",
    ]
    purposes = [
        "Purchase and Renovation", "Purchase", "New Construction",
        "Refinancing", "Modernization",
    ]

    city = fake.city()
    incorporation = fake.date_between(start_date="-25y", end_date="-2y")

    # Financials kept internally consistent: equity + loan ≈ purchase price
    purchase_price = random.randint(500, 5000) * 1000
    equity = round(purchase_price * random.uniform(0.10, 0.30) / 1000) * 1000
    loan_amount = purchase_price - equity
    term_years = random.choice([10, 15, 20, 25, 30])
    # rough monthly installment, just plausible not exact amortization
    monthly = round(loan_amount / (term_years * 12) * random.uniform(1.1, 1.4) / 100) * 100

    early_repay = random.choice([True, False])
    subsidies = random.choice([True, False])

    record = {
        # Section 1 - Applicant
        "company_name": fake.company() + " " + random.choice(["GmbH", "AG", "UG"]),
        "legal_form": random.choice(legal_forms),
        "date_of_incorporation": incorporation.strftime("%d/%m/%Y"),
        "business_address": f"{fake.street_name()} {random.randint(1,200)}, "
                            f"{fake.postcode()} {city}, Germany",
        "commercial_register": f"HRB {random.randint(100000,999999)} / {city} Local Court",
        "vat_id": "DE" + str(random.randint(100000000, 999999999)),
        # Section 2 - Property
        "property_type": random.choice(property_types),
        "property_name": f"{fake.word().capitalize()} Center {city}",
        "property_address": f"{fake.street_name()} {random.randint(1,200)}, "
                            f"{fake.postcode()} {city}, Germany",
        "purchase_price": purchase_price,
        "financing_amount": loan_amount,
        "purpose_of_use": random.choice(purposes),
        "equity_contribution": equity,
        "year_of_construction": random.randint(1960, 2023),
        "total_area_m2": random.randint(200, 6000),
        # Section 3 - Financing proposal
        "desired_loan_amount": loan_amount,
        "term_years": term_years,
        "monthly_installment": monthly,
        "interest_rate": random.choice(["Fixed", "Variable"]),
        "early_repayment": early_repay,
        "public_subsidies": subsidies,
        # Section 4 - Declaration
        "signature_city": city,
        "signature_date": fake.date_between(
            start_date="-6m", end_date="today").strftime("%d/%m/%Y"),
    }
    return record


# ---------------------------------------------------------------------------
# 2. Render a record into a PDF that mimics the original layout
# ---------------------------------------------------------------------------
def euro(n: int) -> str:
    return f"\u20ac{n:,}"


def checkbox(checked: bool) -> str:
    yes = "[x] yes" if checked else "[ ] yes"
    no = "[ ] no" if checked else "[x] no"
    return f"{yes}   {no}"


def build_pdf(record: dict, path: Path) -> None:
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "FormTitle", parent=styles["Title"], fontSize=20, textColor=HEADER_BLUE,
        alignment=0, spaceAfter=4, leading=24,
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"], fontSize=12, textColor=HEADER_BLUE,
        spaceBefore=14, spaceAfter=6,
    )
    cell_style = ParagraphStyle("Cell", parent=styles["Normal"], fontSize=9, leading=12)

    def row(label, value):
        return [Paragraph(label, cell_style), Paragraph(str(value), cell_style)]

    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        topMargin=18 * mm, bottomMargin=18 * mm,
        leftMargin=18 * mm, rightMargin=18 * mm,
    )
    elems = []
    elems.append(Paragraph(
        "Loan Application for a Commercial Real Estate Loan - Company Profile",
        title_style))
    elems.append(Spacer(1, 8))

    table_style = TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.6, BORDER_GREY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ])
    col_widths = [70 * mm, 100 * mm]

    # Section 1
    elems.append(Paragraph("1. Applicant", section_style))
    s1 = [
        row("Company Name", record["company_name"]),
        row("Legal Form", record["legal_form"]),
        row("Date of Incorporation", record["date_of_incorporation"]),
        row("Business Address", record["business_address"]),
        row("Commercial Register Number / Court", record["commercial_register"]),
        row("VAT ID / Tax Number", record["vat_id"]),
    ]
    t = Table(s1, colWidths=col_widths); t.setStyle(table_style); elems.append(t)

    # Section 2
    elems.append(Paragraph("2. Financial Request / Property Description", section_style))
    s2 = [
        row("Type of Property", record["property_type"]),
        row("Property Name", record["property_name"]),
        row("Adress", record["property_address"]),
        row("Purchase Price / Construction Costs", euro(record["purchase_price"])),
        row("Desired Financing Amount", euro(record["financing_amount"])),
        row("Purpose of Use", record["purpose_of_use"]),
        row("Equity Contribution", euro(record["equity_contribution"])),
        row("Year of Construction", record["year_of_construction"]),
        row("Total Area", f"{record['total_area_m2']:,} m\u00b2"),
    ]
    t = Table(s2, colWidths=col_widths); t.setStyle(table_style); elems.append(t)

    # Section 3
    elems.append(Paragraph("3. Financing Proposal", section_style))
    s3 = [
        row("Desired Loan Amount", euro(record["desired_loan_amount"])),
        row("Term", f"{record['term_years']} years"),
        row("Preferred Installment Amount", f"{euro(record['monthly_installment'])} per month"),
        row("Interest Rate", record["interest_rate"]),
        row("Early Repayment Desired?", checkbox(record["early_repayment"])),
        row("Public Subsidies Applied For?", checkbox(record["public_subsidies"])),
    ]
    t = Table(s3, colWidths=col_widths); t.setStyle(table_style); elems.append(t)

    # Section 4
    elems.append(Paragraph("4. Declaration & Signature", section_style))
    elems.append(Paragraph(
        "I hereby confirm the accuracy and completeness of the information provided.",
        cell_style))
    elems.append(Spacer(1, 6))
    elems.append(Paragraph(
        f"{record['signature_city']}, {record['signature_date']}", cell_style))
    elems.append(Spacer(1, 18))
    elems.append(Paragraph(
        "____________________________&nbsp;&nbsp;&nbsp;&nbsp;"
        "____________________________________", cell_style))
    elems.append(Paragraph(
        "Place / Date&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
        "Signature (Managing Director)", cell_style))

    doc.build(elems)


# ---------------------------------------------------------------------------
# 3. Optional: degrade a PDF into a "scanned" image-based PDF for OCR stress tests
# ---------------------------------------------------------------------------
def degrade_to_scanned(pdf_path: Path, out_path: Path) -> None:
    """Render PDF -> image, add noise/rotation/blur, save as image-based PDF.
    Requires: pdf2image (and poppler), pillow."""
    from pdf2image import convert_from_path
    from PIL import Image, ImageFilter
    import numpy as np

    pages = convert_from_path(str(pdf_path), dpi=150)
    processed = []
    for img in pages:
        img = img.convert("L")                       # grayscale, like a scan
        img = img.rotate(random.uniform(-1.5, 1.5),  # slight skew
                         expand=False, fillcolor=255)
        arr = np.array(img).astype(np.int16)
        noise = np.random.normal(0, 8, arr.shape)    # sensor noise
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        img = img.filter(ImageFilter.GaussianBlur(0.5))
        processed.append(img.convert("RGB"))
    processed[0].save(str(out_path), save_all=True, append_images=processed[1:])


# ---------------------------------------------------------------------------
# 4. Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--outdir", default="./synthetic_data")
    ap.add_argument("--degrade", action="store_true",
                    help="also produce scanned-style noisy versions")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed); Faker.seed(args.seed)

    out = Path(args.outdir)
    (out / "pdfs").mkdir(parents=True, exist_ok=True)
    (out / "ground_truth").mkdir(parents=True, exist_ok=True)
    if args.degrade:
        (out / "scanned").mkdir(parents=True, exist_ok=True)

    index = []
    for i in range(1, args.count + 1):
        doc_id = f"loan_{i:04d}"
        record = make_record()
        pdf_path = out / "pdfs" / f"{doc_id}.pdf"
        gt_path = out / "ground_truth" / f"{doc_id}.json"

        build_pdf(record, pdf_path)
        gt_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))

        if args.degrade:
            try:
                degrade_to_scanned(pdf_path, out / "scanned" / f"{doc_id}.pdf")
            except Exception as e:
                print(f"  (degrade skipped for {doc_id}: {e})")

        index.append({"document_id": doc_id,
                      "pdf": str(pdf_path),
                      "ground_truth": str(gt_path)})
        print(f"generated {doc_id}")

    (out / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\nDone. {args.count} documents in '{out}'.")
    print("Each PDF in pdfs/ has a matching JSON in ground_truth/ with the correct values.")


if __name__ == "__main__":
    main()

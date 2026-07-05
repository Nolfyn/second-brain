"""
Run this once to create the test fixture files:
    python tests/make_fixtures.py

It creates:
  tests/fixtures/sample_plain.docx      — plain unprotected Word doc
  tests/fixtures/sample_protected.docx  — password-protected Word doc
  tests/fixtures/sample_text.pdf        — text-based PDF
"""

import os
import sys
from pathlib import Path

# Make sure we can import from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()


def make_plain_docx(output: Path) -> None:
    from docx import Document

    doc = Document()
    doc.add_heading("Test Document", level=1)
    doc.add_paragraph(
        "This is a sample document for testing the second-brain parser. "
        "It contains English and some Russian text."
    )
    doc.add_paragraph(
        "Это тестовый абзац на русском языке. "
        "Семантический поиск должен находить этот текст."
    )
    doc.add_paragraph(
        "A second paragraph with different content about philosophy and stoicism."
    )
    doc.save(output)
    print(f"Created: {output}")


def make_protected_docx(output: Path, password: str) -> None:
    import io
    import msoffcrypto
    from docx import Document

    # Create a plain docx in memory first
    doc = Document()
    doc.add_heading("Protected Document", level=1)
    doc.add_paragraph(
        "This document is password-protected. If you can read this, decryption worked."
    )
    doc.add_paragraph("Этот документ защищён паролем.")

    plain_buffer = io.BytesIO()
    doc.save(plain_buffer)
    plain_buffer.seek(0)

    # Encrypt it
    encrypted_buffer = io.BytesIO()
    office_file = msoffcrypto.OfficeFile(plain_buffer)
    office_file.encrypt(password, encrypted_buffer)
    encrypted_buffer.seek(0)

    with open(output, "wb") as f:
        f.write(encrypted_buffer.read())
    print(f"Created: {output} (password: {password})")


def make_text_pdf(output: Path) -> None:
    try:
        from reportlab.pdfgen import canvas as rl_canvas

        c = rl_canvas.Canvas(str(output))
        c.drawString(72, 750, "Second Brain — Test PDF")
        c.drawString(
            72,
            720,
            "This is a text-based PDF for testing pdfminer extraction.",
        )
        c.drawString(
            72,
            700,
            "It contains readable text that pdfminer can extract directly.",
        )
        c.save()
        print(f"Created: {output}")
    except ImportError:
        # Fallback: create a minimal PDF by hand (no external lib needed)
        pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 120>>
stream
BT /F1 12 Tf 72 750 Td (Second Brain Test PDF) Tj 0 -20 Td (This is a text-based PDF for parser testing.) Tj ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000274 00000 n
0000000446 00000 n
trailer<</Size 6/Root 1 0 R>>
startxref
528
%%EOF"""
        with open(output, "wb") as f:
            f.write(pdf_content)
        print(
            f"Created: {output} (minimal PDF — install reportlab for a better one)"
        )


def main():
    fixtures_dir = Path(__file__).parent / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)

    password = os.getenv("DOC_PASSWORD", "test123")

    make_plain_docx(fixtures_dir / "sample_plain.docx")
    make_protected_docx(fixtures_dir / "sample_protected.docx", password)
    make_text_pdf(fixtures_dir / "sample_text.pdf")

    print()
    print("Done! Remaining fixtures to add manually:")
    print(
        "  tests/fixtures/sample_scanned.pdf  — copy any scanned PDF from your collection"
    )
    print(
        "  tests/fixtures/sample_legacy.doc   — copy any old .doc file from your collection"
    )


if __name__ == "__main__":
    main()

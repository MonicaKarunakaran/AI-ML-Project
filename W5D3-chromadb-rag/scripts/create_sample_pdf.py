from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def create_pdf():
    input_file = Path("data/raw/sample_docs.txt")
    output_file = Path("data/raw/sample.pdf")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    pdf = canvas.Canvas(str(output_file), pagesize=A4)

    width, height = A4
    y = height - 50

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, y, "Machine Learning and AI Concepts")

    y -= 35
    pdf.setFont("Helvetica", 10)

    for line in lines:
        # Wrap long lines
        words = line.split()
        current_line = ""

        for word in words:
            test_line = current_line + " " + word

            if pdf.stringWidth(test_line, "Helvetica", 10) < width - 100:
                current_line = test_line.strip()
            else:
                pdf.drawString(50, y, current_line)
                y -= 18
                current_line = word

                if y < 50:
                    pdf.showPage()
                    pdf.setFont("Helvetica", 10)
                    y = height - 50

        if current_line:
            pdf.drawString(50, y, current_line)
            y -= 25

        if y < 50:
            pdf.showPage()
            pdf.setFont("Helvetica", 10)
            y = height - 50

    pdf.save()

    print(f"PDF created successfully: {output_file}")


if __name__ == "__main__":
    create_pdf()
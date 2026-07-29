"""
report_generator.py
-------------------------------------------------------
Medical Report Generator for MedAssist AI

Generates structured reports from CNN predictions.
-------------------------------------------------------
"""

from datetime import datetime
import os

from dotenv import load_dotenv
import google.generativeai as genai
from pypdf import PdfReader

load_dotenv()


class MedicalPDFReportAnalyzer:
    """
    Extracts and summarizes uploaded medical report PDFs.
    """

    def extract_text(self, pdf_file) -> str:
        reader = PdfReader(pdf_file)
        pages = []

        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text.strip())

        return "\n\n".join(pages)

    def generate_summary(self, report_text: str) -> str:
        report_text = (report_text or "").strip()

        if not report_text:
            return "No readable report text was found."

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY not found in .env file."
            )

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = f"""
You are MedAssist AI.

Summarize the following medical report in simple language for a patient.
Do not diagnose. Include:
- Key findings
- Values or observations that may need doctor review
- Suggested questions to ask a qualified clinician
- A clear educational-use disclaimer

Medical report text:

{report_text[:12000]}
"""

        response = model.generate_content(prompt)
        return response.text


class MedicalReportGenerator:
    """
    Generates structured medical reports.
    """

    def __init__(self, hospital_name="MedAssist AI"):

        self.hospital_name = hospital_name

    def generate(
        self,
        dataset_name,
        prediction,
        confidence,
        probabilities,
        processing_time,
        patient_name="Unknown",
        patient_id="N/A"
    ):

        report = {
            "hospital": self.hospital_name,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "patient_name": patient_name,
            "patient_id": patient_id,
            "dataset": dataset_name,
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "processing_time": round(processing_time, 3),
            "probabilities": probabilities,
            "disclaimer": (
                "This prediction is generated using an AI model "
                "and should not be considered a medical diagnosis. "
                "Please consult a qualified healthcare professional."
            )
        }

        return report

    def print_report(self, report):

        print("=" * 60)
        print("MEDASSIST AI MEDICAL REPORT")
        print("=" * 60)

        print(f"Hospital          : {report['hospital']}")
        print(f"Date              : {report['date']}")
        print(f"Time              : {report['time']}")
        print(f"Patient Name      : {report['patient_name']}")
        print(f"Patient ID        : {report['patient_id']}")

        print("-" * 60)

        print(f"Dataset           : {report['dataset']}")
        print(f"Prediction        : {report['prediction']}")
        print(f"Confidence        : {report['confidence']} %")
        print(f"Processing Time   : {report['processing_time']} sec")

        print("-" * 60)

        print("Probability Distribution")

        if isinstance(report["probabilities"], dict):

            for label, value in report["probabilities"].items():

                print(f"{label:<25}: {value:.2f}%")

        elif isinstance(report["probabilities"], list):

            for idx, value in enumerate(report["probabilities"]):

                print(f"Class {idx:<18}: {value * 100:.2f}%")

        print("-" * 60)

        print("DISCLAIMER")
        print(report["disclaimer"])

        print("=" * 60)

    def save_report(self, report, filename):

        with open(filename, "w", encoding="utf-8") as file:

            file.write("=" * 60 + "\n")
            file.write("MEDASSIST AI MEDICAL REPORT\n")
            file.write("=" * 60 + "\n\n")

            for key, value in report.items():

                if key == "probabilities":

                    file.write("Probabilities\n")

                    if isinstance(value, dict):

                        for label, prob in value.items():

                            file.write(
                                f"{label}: {prob:.2f}%\n"
                            )

                    elif isinstance(value, list):

                        for idx, prob in enumerate(value):

                            file.write(
                                f"Class {idx}: {prob * 100:.2f}%\n"
                            )

                    file.write("\n")

                else:

                    file.write(
                        f"{key}: {value}\n"
                    )

        print(f"\n✅ Report saved to {filename}")


if __name__ == "__main__":

    generator = MedicalReportGenerator()

    report = generator.generate(

        dataset_name="PneumoniaMNIST",

        prediction="Pneumonia",

        confidence=98.64,

        probabilities={
            "Normal": 1.36,
            "Pneumonia": 98.64
        },

        processing_time=0.18,

        patient_name="John Doe",

        patient_id="MED001"

    )

    generator.print_report(report)

    generator.save_report(
        report,
        "medical_report.txt"
    )

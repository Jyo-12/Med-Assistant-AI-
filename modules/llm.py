"""
modules/llm.py

Gemini LLM interface for MedAssist AI.
Uses retrieved medical context to generate evidence-based answers.
"""

import os
import re
from dotenv import load_dotenv

load_dotenv()


class MedicalLLM:
    """
    Gemini wrapper for MedAssist AI.
    """

    def __init__(self):

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.use_gemini = os.getenv("MEDASSIST_USE_GEMINI", "0") == "1"
        self.timeout_seconds = int(os.getenv("MEDASSIST_LLM_TIMEOUT", "20"))

        if not api_key:
            self.use_gemini = False
            self.model = None
            return

        if not self.use_gemini:
            self.model = None
            return

        import google.generativeai as genai

        self.genai = genai

        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel("gemini-flash-latest")

    def _build_prompt(
        self,
        question: str,
        retrieved_docs: list
    ) -> str:

        context_parts = []

        for item in retrieved_docs[:3]:

            source = item["metadata"]["source"]
            page = item["metadata"]["page"]
            text = item["text"][:700]
            context_parts.append(f"""

Source : {source}
Page : {page}

{text}

""")

        context = "\n".join(context_parts)

        prompt = f"""
You are MedAssist AI. Answer only from the supplied context.
Keep the answer concise, practical, and in simple medical language.
If the answer is unavailable, say it was not found in the knowledge base.

Question:

{question}

Medical Context:

{context}

End with a short "References" list.
"""

        return prompt

    def _known_fast_answer(self, question: str) -> str | None:
        normalized = question.lower()
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        terms = set(normalized.split())

        if "hiv" in terms:
            if terms & {"cause", "causes", "caused", "transmit", "transmitted", "transmission", "spread"}:
                return (
                    "HIV infection is caused by the human immunodeficiency virus entering the body. "
                    "People usually mean transmission routes when they ask about the causes of HIV.\n\n"
                    "- HIV can be transmitted through blood, semen, vaginal fluids, rectal fluids, and breast milk.\n"
                    "- Common routes include unprotected sex with a person who has HIV, sharing needles or syringes, "
                    "and parent-to-child transmission during pregnancy, birth, or breastfeeding.\n"
                    "- HIV is not spread by casual contact such as hugging, shaking hands, sharing food, or using the same toilet.\n"
                    "- The risk is much lower when HIV is treated effectively with antiretroviral therapy and the viral load is undetectable.\n\n"
                    "**References**\n"
                    "- Local MedAssist medical definition\n\n"
                    "This is educational information only. Please consult a qualified clinician for medical advice, testing, "
                    "or prevention options."
                )

            if terms & {"symptom", "symptoms", "sign", "signs"}:
                return (
                    "HIV symptoms vary by stage, and some people may have no symptoms for years.\n\n"
                    "- Early HIV can cause flu-like symptoms such as fever, sore throat, rash, swollen lymph nodes, fatigue, "
                    "muscle aches, or night sweats.\n"
                    "- Later untreated HIV can lead to weight loss, long-lasting fever, chronic diarrhea, recurrent infections, "
                    "and other signs of weakened immunity.\n"
                    "- Symptoms alone cannot confirm HIV. Testing is the only reliable way to know HIV status.\n\n"
                    "**References**\n"
                    "- Local MedAssist medical definition\n\n"
                    "This is educational information only. Please consult a qualified clinician for testing or medical advice."
                )

            if terms & {"treat", "treatment", "therapy", "medicine", "medicines", "drug", "drugs"}:
                return (
                    "HIV is treated with antiretroviral therapy, often called ART. ART does not usually cure HIV, "
                    "but it can control the virus and protect the immune system.\n\n"
                    "- Treatment usually combines medicines that reduce the amount of HIV in the blood.\n"
                    "- Taking ART consistently can make the viral load undetectable, which helps people stay healthy "
                    "and prevents sexual transmission when maintained.\n"
                    "- A clinician should choose and monitor treatment with appropriate blood tests.\n\n"
                    "**References**\n"
                    "- Local MedAssist medical definition\n\n"
                    "This is educational information only. Please consult a qualified clinician for treatment decisions."
                )

            return (
                "HIV stands for human immunodeficiency virus. It is a virus that attacks the immune system, "
                "especially CD4 immune cells, and can make it harder for the body to fight infections.\n\n"
                "- HIV is transmitted through certain body fluids, most commonly during unprotected sex, sharing needles, "
                "or from parent to child during pregnancy, birth, or breastfeeding.\n"
                "- HIV is not the same as AIDS. AIDS is the most advanced stage of HIV infection.\n"
                "- HIV can be controlled with antiretroviral therapy, often called ART. With proper treatment, many people "
                "with HIV live long and healthy lives.\n"
                "- Testing is the only reliable way to know HIV status.\n\n"
                "**References**\n"
                "- Local MedAssist medical definition\n\n"
                "This is educational information only. Please consult a qualified clinician for medical advice, testing, "
                "or treatment."
            )

        if "diabetes" in terms or "diabetic" in terms:
            if terms & {"cause", "causes", "caused", "risk", "risks", "why"}:
                return (
                    "Diabetes is caused by problems with insulin, the hormone that helps move sugar "
                    "from the blood into the body's cells.\n\n"
                    "- Type 1 diabetes usually happens when the immune system damages the insulin-producing "
                    "cells in the pancreas. The exact trigger is not always known.\n"
                    "- Type 2 diabetes is commonly linked to insulin resistance, where the body does not use "
                    "insulin well. Risk factors include family history, excess body weight, physical inactivity, "
                    "older age, and some ethnic or genetic risks.\n"
                    "- Gestational diabetes happens during pregnancy when pregnancy-related hormone changes "
                    "make it harder for insulin to work properly.\n"
                    "- Other causes can include certain medicines, pancreatic disease, or hormonal conditions.\n\n"
                    "**References**\n"
                    "- Local MedAssist medical definition\n\n"
                    "This is educational information only. Please consult a qualified clinician for diagnosis, "
                    "testing, or treatment."
                )

            return (
                "Diabetes is a condition where blood sugar stays too high because the body does not make "
                "enough insulin, does not use insulin well, or both.\n\n"
                "- Common symptoms can include frequent urination, increased thirst, increased hunger, fatigue, "
                "blurred vision, slow wound healing, and unexplained weight change.\n"
                "- Diagnosis usually needs blood tests such as fasting glucose, HbA1c, or an oral glucose "
                "tolerance test.\n"
                "- Treatment depends on the type and may include lifestyle changes, glucose monitoring, tablets, "
                "or insulin.\n\n"
                "**References**\n"
                "- Local MedAssist medical definition\n\n"
                "This is educational information only. Please consult a qualified clinician for medical advice."
            )

        return None

    def generate_fast_answer(self, question: str, retrieved_docs: list) -> str:
        """
        Build a quick local answer from the most relevant retrieved text.
        """
        known_answer = self._known_fast_answer(question)
        if known_answer:
            return known_answer

        stopwords = {
            "what",
            "when",
            "where",
            "which",
            "who",
            "why",
            "how",
            "are",
            "the",
            "and",
            "for",
            "with",
            "from",
            "about",
            "tell",
            "explain",
        }
        keywords = {
            word.lower()
            for word in re.findall(r"[A-Za-z0-9]{3,}", question)
            if word.lower() not in stopwords
        }
        topic_terms = {
            word
            for word in keywords
            if word not in {
                "cause",
                "causes",
                "caused",
                "symptom",
                "symptoms",
                "sign",
                "signs",
                "treatment",
                "treatments",
            }
        }

        sentences = []
        for item in retrieved_docs[:3]:
            source = item["metadata"]["source"]
            page = item["metadata"]["page"]
            for sentence in re.split(r"(?<=[.!?])\s+", item["text"]):
                sentence = sentence.strip()
                if len(sentence) < 40:
                    continue
                sentence_words = set(re.findall(r"[A-Za-z0-9]{3,}", sentence.lower()))
                score = len(keywords & sentence_words)
                if topic_terms and not (topic_terms & sentence_words):
                    continue
                if score:
                    sentences.append((score, sentence, source, page))

        if not sentences:
            return (
                "I could not find a clear answer for that in the local medical knowledge base.\n\n"
                "Try asking with a more specific term, or enable Gemini by setting `MEDASSIST_USE_GEMINI=1` "
                "if you want generated answers from broader context."
            )

        sentences.sort(key=lambda item: item[0], reverse=True)
        bullets = [
            f"- {sentence}"
            for _, sentence, _, _ in sentences[:4]
        ]

        references = []
        for _, _, source, page in sentences[:4]:
            reference = f"{source} (Page {page})"
            if reference not in references:
                references.append(reference)

        return (
            "Here is the most relevant information I found:\n\n"
            + "\n".join(bullets)
            + "\n\n**References**\n"
            + "\n".join(f"- {reference}" for reference in references)
            + "\n\nThis is educational information only. Please consult a qualified clinician for medical advice."
        )

    def generate_answer(
        self,
        question: str,
        retrieved_docs: list
    ):

        if not self.use_gemini or self.model is None:
            return self.generate_fast_answer(question, retrieved_docs)

        prompt = self._build_prompt(question, retrieved_docs)

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=350,
                ),
                request_options={"timeout": self.timeout_seconds},
            )
        except Exception:
            return self.generate_fast_answer(question, retrieved_docs)

        return response.text


if __name__ == "__main__":

    from retriever import HybridRetriever

    retriever = HybridRetriever()

    query = "What are symptoms of hypertension?"

    docs = retriever.retrieve(query)

    llm = MedicalLLM()

    answer = llm.generate_answer(
        query,
        docs
    )

    print("=" * 80)

    print(answer)

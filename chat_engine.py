"""
chat_engine.py

Main chat engine for MedAssist AI.
Connects:
- Hybrid Retriever
- Gemini LLM
- Confidence Estimation
"""

from modules.retriever import HybridRetriever
from modules.llm import MedicalLLM


class ChatEngine:
    def __init__(self):
        print("Loading Retriever...")
        self.retriever = HybridRetriever()

        print("Loading answer engine...")
        self.llm = MedicalLLM()

        print("ChatEngine initialized successfully.")

    def ask(self, question: str):
        """
        Ask a medical question.
        """

        print("\n" + "=" * 60)
        print("STEP 1 - Question received")
        print(f"Question: {question}")

        print("STEP 2 - Retrieving documents...")
        retrieved_docs = self.retriever.retrieve(
            question,
            top_k=3
        )

        print("STEP 3 - Documents retrieved")
        print(f"Retrieved {len(retrieved_docs)} document(s)")

        if len(retrieved_docs) == 0:
            print("No relevant documents found.")

            return {
                "question": question,
                "answer": "No relevant medical information found.",
                "confidence": 0,
                "sources": []
            }

        print("STEP 4 - Generating answer...")
        answer = self.llm.generate_answer(
            question,
            retrieved_docs
        )

        print("STEP 5 - Answer generated successfully.")

        confidence = round(
            retrieved_docs[0]["semantic_score"] * 100,
            2
        )

        sources = []

        for doc in retrieved_docs:
            source = (
                f'{doc["metadata"]["source"]} '
                f'(Page {doc["metadata"]["page"]})'
            )

            if source not in sources:
                sources.append(source)

        print("STEP 6 - Response ready.")
        print("=" * 60)

        return {
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "sources": sources
        }

    def generate_response(self, question: str) -> str:
        """
        Return a Streamlit-friendly chat response string.
        """

        result = self.ask(question)

        answer = result.get("answer", "")
        sources = result.get("sources", [])

        if not sources:
            return answer

        if "**References**" in answer or "References" in answer:
            return answer

        source_list = "\n".join(
            f"- {source}" for source in sources
        )

        return (
            f"{answer}\n\n"
            f"**Sources**\n"
            f"{source_list}"
        )


if __name__ == "__main__":
    engine = ChatEngine()

    print("=" * 80)
    print("MedAssist AI Chat")
    print("=" * 80)

    while True:
        query = input("\nAsk a medical question (type exit): ")

        if query.lower() == "exit":
            break

        result = engine.ask(query)

        print("\nConfidence :", result["confidence"], "%")

        print("\nSources")

        for source in result["sources"]:
            print("-", source)

        print("\nAnswer\n")
        print(result["answer"])

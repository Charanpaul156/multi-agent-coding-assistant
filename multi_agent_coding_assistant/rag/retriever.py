"""RAG retriever placeholder.

This is prepared for future LangChain + ChromaDB integration.
"""


class Retriever:
    def retrieve(self, query: str) -> list[str]:
        raise NotImplementedError("Retriever not implemented yet")


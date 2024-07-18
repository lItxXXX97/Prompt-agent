from phi.assistant import Assistant
# from phi.llm.openai import OpenAIChat
# from phi.tools.yfinance import YFinanceTools
# from phi.llm.groq import Groq
import os
os.environ["OPENAI_BASE_URL"] = "https://api.aigc369.com/v1"

from phi.knowledge.pdf import PDFKnowledgeBase, PDFReader
from phi.vectordb.pgvector import PgVector2

pdf_knowledge_base = PDFKnowledgeBase(
    path="data/pdfs",
    # Table name:ai.pdf_documents
    vector_db=PgVector2(
        collection="pdf_documents",
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
    ),
    reader=PDFReader(chunk=True),
)


assistant = Assistant(
    knowledge_base=pdf_knowledge_base,
    add_references_to_prompt=True,
)
assistant.knowledge_base.load(recreate=False)

assistant.print_response("请展示一些控制生成的prompts")

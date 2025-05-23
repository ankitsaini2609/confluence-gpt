import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import cacheHuggingFaceModel

# Load environment variables
cacheHuggingFaceModel.main()
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./hugging_chroma")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Load Chroma vector store
print(f"🔍 Loading Chroma vector store from: {CHROMA_PERSIST_DIR}")
embedding_fn = HuggingFaceEmbeddings(
    model_name="bge-base-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}
)
vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embedding_fn)

# Create retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Accept user query
query = input("❓ Ask a question: ")

# Retrieve documents
docs = retriever.invoke(query)
context_blocks = []
used_sources = []

for i, doc in enumerate(docs):
    tag = f"[SOURCE-{i+1}]"
    title = doc.metadata.get("source", f"Document {i+1}")
    url = doc.metadata.get("url", "N/A")
    page_content = doc.page_content.strip()

    block = f"{tag} {title} - {url}\n{page_content}"
    context_blocks.append(block)

    used_sources.append({
        "tag": tag,
        "title": title,
        "url": url
    })

combined = "\n\n---\n\n".join(context_blocks)

# Define prompt
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant for internal documentation.

Answer the question using only the information from the context below.
Reference the relevant [SOURCE-X] tags in your answer if needed.

If the answer is not present, respond with:
"I couldn't find that in the documentation."

Context:
{context}

Question:
{question}

Answer:"""
)

# Initialize LLM (OpenAI)
llm = ChatOpenAI(
    model_name="gpt-4",   # or "gpt-3.5-turbo" if preferred
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)

# Expression-based chaining (LCEL style)
chain = prompt_template | llm
response = chain.invoke({"context": combined, "question": query})


print("\n🤖 Answer:\n")
print(response.content.strip())
print("\n🔗 Referenced Sources:")
printed_sources = set()
found_any = False
for source in used_sources:
    if source["tag"] in response.content and source["url"] not in printed_sources:
        print(f"- {source['title']}: {source['url']}")
        printed_sources.add(source["url"])
        found_any = True

if not found_any:
    print("No specific source referenced.")

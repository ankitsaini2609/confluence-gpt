import os
from atlassian import Confluence
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
import base64
import cacheHuggingFaceModel

class ingestToChroma():
    def __init__(self):
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        print("locally saving the hugging face model")
        cacheHuggingFaceModel.main()
        self.CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")
        self.CONFLUENCE_SPACE_KEY = "Security"
        self.CONFLUENCE_TOKEN = os.getenv("CONFLUENCE_TOKEN")
        if not self.CONFLUENCE_TOKEN:
            raise ValueError("Missing CONFLUENCE_TOKEN environment variable.")
        decoded_bytes = base64.b64decode(self.CONFLUENCE_TOKEN)
        decoded_str = decoded_bytes.decode("utf-8")
        try:
            self.CONFLUENCE_USERNAME, self.CONFLUENCE_API_TOKEN = decoded_str.split(":")
        except ValueError:
            raise ValueError("Invalid CONFLUENCE_TOKEN format. Should be base64(email:token)")

        self.confluence = Confluence(
            url=self.CONFLUENCE_URL,
            username=self.CONFLUENCE_USERNAME,
            password=self.CONFLUENCE_API_TOKEN
        )
        self.CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./hugging_chroma")

        if not all([self.CONFLUENCE_URL, self.CONFLUENCE_USERNAME, self.CONFLUENCE_API_TOKEN, self.CONFLUENCE_SPACE_KEY]):
            raise ValueError("Missing one or more required environment variables.")

    def fetchConfluencePage(self):
        # Fetch all pages in the space
        pages = self.confluence.get_all_pages_from_space(
            space=self.CONFLUENCE_SPACE_KEY,
            start=0,
            limit=200,
            status='current'
        )
        print(f"✅ Retrieved {len(pages)} pages from space '{self.CONFLUENCE_SPACE_KEY}'.")
        documents = []
        for page in pages:
            page_id = page["id"]
            page_url = f"{self.CONFLUENCE_URL}/wiki/pages/viewpage.action?pageId={page_id}"
            title = page["title"]
            try:
                page_content_raw = self.confluence.get_page_by_id(page_id, expand="body.storage")
                html = page_content_raw["body"]["storage"]["value"]
                parsed_body = self.extract_text_preserve_ordered(html)
                text = f"# {title}\n\n{parsed_body}"
                documents.append(Document(page_content=text, metadata={"source": title, "url": page_url, "page_id": page_id}))
            except Exception as e:
                print(f"⚠️ Failed to process page {title}: {e}")
        print(f"✅ Processed {len(documents)} documents.")

        return documents 


    def resolve_user_display_name(self, account_id: str) -> str:
        try:
            url = f"{self.confluence.url}/rest/api/user"
            params = {"accountId": account_id}
            response = self.confluence._session.get(url, params=params)

            if response.status_code == 200:
                return response.json().get("displayName", f"@{account_id}")
            else:
                return f"@{account_id}"
        except Exception:
            return f"@{account_id}"



    def parse_table_cell(self, td):
        # Jira macro inside table cell
        jira_macro = td.find("ac:structured-macro", {"ac:name": "jira"})
        if jira_macro:
            key_param = jira_macro.find("ac:parameter", {"ac:name": "key"})
            if key_param and key_param.string:
                return f"Jira Issue: {key_param.string.strip()}"

        # User mention
        user_tag = td.find("ri:user")
        if user_tag:
            account_id = user_tag.get("ri:account-id")
            if account_id:
                return f"@{self.resolve_user_display_name(account_id)}"

        # Page link
        page_tag = td.find("ri:page")
        if page_tag:
            return f"[{page_tag.get('ri:content-title', 'Linked Page')}]"

        # Attachment link
        attachment_tag = td.find("ri:attachment")
        if attachment_tag:
            return f"Attachment: {attachment_tag.get('ri:filename', 'file')}"

        return td.get_text(strip=True)


    def extract_text_preserve_ordered(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        output = []

        for tag in soup.recursiveChildGenerator():
            if isinstance(tag, str):
                continue

            if tag.name == "ac:structured-macro":
                macro_name = tag.get("ac:name", "")

                if macro_name == "code":
                    cdata = tag.find("ac:plain-text-body")
                    if cdata and cdata.string:
                        code = cdata.string.strip()
                        output.append(f"\n```bash\n{code}\n```\n")

                elif macro_name == "jira":
                    key_param = tag.find("ac:parameter", {"ac:name": "key"})
                    if key_param and key_param.string:
                        output.append(f"Jira Issue: {key_param.string.strip()}")

                elif macro_name == "status":
                    title = tag.find("ac:parameter", {"ac:name": "title"})
                    if title and title.string:
                        output.append(f"[Status: {title.string.strip()}]")

                elif macro_name in {"info", "note", "warning"}:
                    body = tag.find("ac:rich-text-body")
                    if body:
                        panel_text = body.get_text(strip=True)
                        output.append(f"[{macro_name.capitalize()} Panel] {panel_text}")

            elif tag.name == "ac:task":
                task_body = tag.find("ac:task-body")
                if task_body:
                    task_text = task_body.get_text(strip=True)
                    output.append(f"- [ ] {task_text}")

            elif tag.name.startswith("h") and tag.name[1:].isdigit():
                output.append(f"\n# {tag.get_text(strip=True)}\n")

            elif tag.name == "p":
                paragraph_text = ""
                for sub in tag.contents:
                    if getattr(sub, "name", None) == "a":
                        href = sub.get("href")
                        text = sub.get_text(strip=True)
                        paragraph_text += f"[{text}]({href})" if href else text

                    elif getattr(sub, "name", None) == "ac:link":
                        if sub.find("ri:user"):
                            user_tag = sub.find("ri:user")
                            if user_tag:
                                account_id = user_tag.get("ri:account-id")
                                if account_id:
                                    paragraph_text += f"@{self.resolve_user_display_name(account_id)}"
                        elif sub.find("ri:page"):
                            title = sub.find("ri:page").get("ri:content-title", "Linked Page")
                            paragraph_text += f"[{title}]"
                        elif sub.find("ri:attachment"):
                            name = sub.find("ri:attachment").get("ri:filename", "file")
                            paragraph_text += f"Attachment: {name}"

                    elif isinstance(sub, str):
                        paragraph_text += sub
                    else:
                        paragraph_text += sub.get_text(strip=True)

                if paragraph_text.strip():
                    output.append(paragraph_text.strip())

            elif tag.name == "li":
                output.append(f"- {tag.get_text(strip=True)}")

            elif tag.name == "pre":
                code = tag.get_text("\n").strip()
                output.append(f"\n```bash\n{code}\n```\n")

            elif tag.name == "code" and not tag.find_parent("pre"):
                output.append(f"`{tag.get_text(strip=True)}`")

            elif tag.name == "table":
                rows = tag.find_all("tr")
                if not rows:
                    continue

                headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
                for row in rows[1:]:
                    cells = [self.parse_table_cell(td) for td in row.find_all("td")]
                    if len(cells) != len(headers):
                        continue

                    main = f"The {headers[0]} '{cells[0]}'"
                    details = [f"{headers[i]} '{cells[i]}'" for i in range(1, len(headers))]
                    sentence = main + " includes " + ", ".join(details) + "."
                    output.append(sentence)

        return "\n\n".join(output)


    def processConfluencePage(self):
        documents = self.fetchConfluencePage()
        # Chunk documents
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(documents)
        print(f"🔍 Split into {len(chunks)} chunks.")

        # Use HuggingFace embeddings
        embedding_fn = HuggingFaceEmbeddings(
            model_name="bge-base-en-v1.5",
            encode_kwargs={"normalize_embeddings": True}  # Optional but recommended
        )
        vectorstore = Chroma.from_documents(chunks, embedding_fn, persist_directory=self.CHROMA_PERSIST_DIR)

        print(f"✅ Stored embeddings in ChromaDB at: {self.CHROMA_PERSIST_DIR}")



if __name__ == "__main__":
    ic = ingestToChroma()
    ic.processConfluencePage()
    
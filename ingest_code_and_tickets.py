import os, csv, pathlib, re, json, hashlib, time
from dotenv import load_dotenv
from redisvl.index import SearchIndex
from redisvl.utils.vectorize import OpenAITextVectorizer
import tiktoken
from composio import Composio
import numpy as np
load_dotenv()

embedder = OpenAITextVectorizer(model="text-embedding-3-small", api_config={"api_key": os.environ["OPENAI_API_KEY"]})
tickets = SearchIndex.from_existing("tickets_idx", redis_url=os.environ["REDIS_URL"])
code = SearchIndex.from_existing("code_idx", redis_url=os.environ["REDIS_URL"])

SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", "dist", "build", "target", ".venv", "__pycache__", ".txt"}
EXTS = {".py",".ts",".tsx",".js",".jsx",".go",".java",".kt",".rb",".rs",".php",".cs",".scala",".md",".yaml",".yml",".json", ".html"}
MAX_FILE_BYTES = 400_000

def chunk_by_tokens(text, max_tokens=500, overlap=60):
    enc = tiktoken.get_encoding("cl100k_base")
    toks = enc.encode(text)
    chunks = []
    i = 0
    while i < len(toks):
        window = toks[i:i+max_tokens]
        chunks.append(enc.decode(window))
        i += max_tokens - overlap
    return chunks

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def ingest_tickets():
    rows = []
    csv_path = os.getenv("TICKETS_CSV")
    if csv_path:
        with open(csv_path) as f:
            csv_rows = list(csv.DictReader(f))
        # Normalize CSV column names to expected format
        for r in csv_rows:
            rows.append({
                "id": r.get("Issue key", r.get("id", "")),
                "title": r.get("Summary", r.get("title", "")),
                "body": r.get("Description", r.get("body", "")),
                "issuetype": r.get("Issue Type", r.get("issuetype", "Task")),
                "component": r.get("component", ""),
                "assignee": r.get("Assignee", r.get("assignee", "")),
                "sprint": r.get("Sprint", os.getenv("SPRINT_NAME", "")),
            })
    else:
        composio = Composio(api_key=os.environ["COMPOSIO_API_KEY"])
        user_id = os.environ["COMPOSIO_USER_ID"]
        tools = composio.tools.get(user_id=user_id, toolkits=["JIRA"])
        search_slug = next(t["slug"] for t in tools if "SEARCH" in t["slug"] or "ISSUES_SEARCH" in t["slug"])
        res = composio.tools.execute(
            slug=search_slug, user_id=user_id,
            arguments={"jql": os.environ.get("TICKETS_JQL","project = ABC ORDER BY created DESC"),
                       "maxResults": 200}
        )
        for issue in res.get("issues", []):
            comps = [c.get("name","") for c in (issue["fields"].get("components") or [])]
            rows.append({
              "id": issue["key"],
              "title": issue["fields"]["summary"],
              "body": (issue["fields"].get("description") or ""),
              "issuetype": issue["fields"]["issuetype"]["name"],
              "component": "|".join([c for c in comps if c]),   # multiple components via tag separator
              "assignee": (issue["fields"].get("assignee") or {}).get("displayName",""),
              "sprint": os.environ.get("SPRINT_NAME",""),
            })

    if not rows:
        return

    texts = [(r.get("title","") + "\n" + r.get("body","")).strip() for r in rows]
    embs = embedder.embed_many(texts)  # batch!
    payloads = []
    for r, emb in zip(rows, embs):
        # Convert embedding list to numpy array bytes
        emb_bytes = np.array(emb, dtype=np.float32).tobytes()
        payloads.append({
          "id": r["id"],
          "title": r.get("title",""),
          "body": r.get("body",""),
          "issuetype": r.get("issuetype","Task"),
          "component": r.get("component",""),
          "assignee": r.get("assignee",""),
          "sprint": r.get("sprint",""),
          "embedding": emb_bytes,
        })
    tickets.load(payloads, id_field="id")
    print(f"Ingested {len(payloads)} tickets")

# ---- Code repo ----
def ingest_code():
    root = pathlib.Path(os.environ["CODE_DIR"]).resolve()
    repo = os.environ.get("REPO_NAME", root.name)
    payloads, text_batch, meta_batch = [], [], []
    batch_limit = 128

    for p in root.rglob("*"):
        # Skip if any parent directory is in SKIP_DIRS
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        
        if p.is_dir():
            continue
        
        if (p.suffix.lower() not in EXTS) or (p.stat().st_size > MAX_FILE_BYTES):
            continue

        try:
            body = p.read_text(errors="ignore")
        except Exception:
            continue

        for idx, chunk in enumerate(chunk_by_tokens(body, max_tokens=500, overlap=60)):
            text_batch.append(chunk)
            meta_batch.append((f"{repo}:{p.as_posix()}:{idx}", p.as_posix(), (p.suffix or "").lstrip(".").lower()))

            if len(text_batch) >= batch_limit:
                embs = embedder.embed_many(text_batch)
                for (doc_id, path, lang), emb in zip(meta_batch, embs):
                    emb_bytes = np.array(emb, dtype=np.float32).tobytes()
                    payloads.append({
                        "id": doc_id,
                        "repo": repo,
                        "path": path,
                        "lang": lang,
                        "body": text_batch[len(payloads) % batch_limit],  # safe mapping inside window
                        "embedding": emb_bytes,
                    })
                code.load(payloads, id_field="id")
                payloads, text_batch, meta_batch = [], [], []

    if text_batch:
        embs = embedder.embed_many(text_batch)
        for (doc_id, path, lang), emb, chunk in zip(meta_batch, embs, text_batch):
            emb_bytes = np.array(emb, dtype=np.float32).tobytes()
            payloads.append({
                "id": doc_id, "repo": repo, "path": path, "lang": lang, "body": chunk, "embedding": emb_bytes
            })
        code.load(payloads, id_field="id")

    print("Code ingested.")

if __name__ == "__main__":
    print("🚀 Starting ingestion...\n")
    
    print("📋 Ingesting tickets...")
    ingest_tickets()
    
    print("\n💻 Ingesting code...")
    ingest_code()
    
    print("\n✅ Ingestion complete!")

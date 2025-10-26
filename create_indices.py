# create_indices.py (revised)
import os
from dotenv import load_dotenv
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema

load_dotenv()

tickets_schema = IndexSchema.from_dict({
  "index": {
    "name": "tickets_idx",
    "prefix": "ticket",
    "key_separator": ":",
    "storage_type": "hash",
  },
  "fields": [
    {"name": "id", "type": "tag"},
    {"name": "title", "type": "text", "attrs": {"weight": 2.0}},
    {"name": "body", "type": "text"},
    {"name": "issuetype", "type": "tag"},
    {"name": "component", "type": "tag", "attrs": {"separator": "|", "case_sensitive": False}},
    {"name": "assignee", "type": "tag"},
    {"name": "sprint", "type": "tag"},
    {
      "name": "embedding", "type": "vector",
      "attrs": {
        "dims": 1536,                # OpenAI text-embedding-3-small
        "algorithm": "hnsw",
        "distance_metric": "cosine",
        "datatype": "float32",
        "m": 24,                     # good recall/speed tradeoff
        "ef_construction": 200,
        "ef_runtime": 50
      },
    },
  ],
})

code_schema = IndexSchema.from_dict({
  "index": {
    "name": "code_idx",
    "prefix": "code",
    "key_separator": ":",
    "storage_type": "hash",
  },
  "fields": [
    {"name": "id", "type": "tag"},
    {"name": "repo", "type": "tag"},
    {"name": "path", "type": "text", "attrs": {"withsuffixtrie": True}},  # search *.py, filenames, etc.
    {"name": "lang", "type": "tag"},
    {"name": "body", "type": "text"},
    {
      "name": "embedding", "type": "vector",
      "attrs": {
        "dims": 1536,
        "algorithm": "hnsw",
        "distance_metric": "cosine",
        "datatype": "float32",
        "m": 24,
        "ef_construction": 200,
        "ef_runtime": 50
      },
    },
  ],
})

for schema in (tickets_schema, code_schema):
    idx = SearchIndex(schema=schema, redis_url=os.environ["REDIS_URL"])
    idx.create(overwrite=True)
    print("OK:", schema.index.name)

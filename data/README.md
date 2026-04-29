# `data/` directory

This folder holds the **persisted Chroma vector store** that the eval reads against. Everything in here except this README is gitignored.

## Setup

Copy the `chroma_db/` folder from your RAG project into here:

```bash
cp -R /path/to/your/rag-project/data/chroma_db ./chroma_db
```

After copying, the layout should look like:

```
data/
├── README.md
└── chroma_db/
    ├── chroma.sqlite3
    └── <some-uuid>/
        ├── data_level0.bin
        ├── header.bin
        ├── length.bin
        └── link_lists.bin
```

## Important: model and collection names must match

The eval uses `text-embedding-3-small` to embed queries and `collection_name="vendors"` to read the right collection. If your RAG project used different settings, override them in `.env`:

```
MODEL_EMBED=text-embedding-3-small
COLLECTION_NAME=vendors
```

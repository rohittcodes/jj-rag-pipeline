# CLI Quick Reference

## Main Commands

```bash
python cli.py <command> [options]
```

### Available Commands

| Command | Description |
|---------|-------------|
| `setup` | Database setup operations |
| `ingest` | Content ingestion & embeddings |
| `sync` | External data syncing |
| `youtube` | YouTube operations |
| `api` | Start API server |

---

## Setup

```bash
# Setup everything
python cli.py setup --all

# Individual operations
python cli.py setup --database    # Main schema
python cli.py setup --youtube     # YouTube schema
python cli.py setup --indexes     # Vector indexes
```

---

## Ingest

```bash
# Ingest everything
python cli.py ingest --all

# Individual operations
python cli.py ingest --blogs      # Ingest blogs
python cli.py ingest --youtube    # Ingest YouTube
python cli.py ingest --embeddings # Generate embeddings

# Targeted embedding generation
python cli.py ingest --embeddings --blogs-only
python cli.py ingest --embeddings --youtube-only
python cli.py ingest --embeddings --regenerate
```

---

## Sync

```bash
# Sync everything
python cli.py sync --all

# Individual operations
python cli.py sync --sanity --all         # All Sanity articles
python cli.py sync --sanity --id=<ID>     # Specific article
python cli.py sync --products             # Product configs
python cli.py sync --test-data            # Test data PDFs from S3

# Test data options
python cli.py sync --test-data --config-id=15  # Specific config
python cli.py sync --test-data --limit=10      # Limit number of files
```

---

## YouTube

```bash
# Test with 1 video
python cli.py youtube --test

# Fetch transcripts
python cli.py youtube --fetch --count=20
```

---

## API Server

```bash
# Development (with auto-reload)
python cli.py api

# Production mode
python cli.py api --no-reload

# Custom host/port
python cli.py api --host 127.0.0.1 --port 3000
```

---

## Common Workflows

### Initial Setup
```bash
python cli.py setup --all
python cli.py sync --all
python cli.py ingest --all
python cli.py setup --indexes
python cli.py api
```

### Update Content
```bash
python cli.py sync --sanity --all
```

### Add YouTube Videos
```bash
python cli.py youtube --fetch --count=20
python cli.py ingest --youtube
python cli.py ingest --embeddings --youtube-only
```

### Update Product Data
```bash
python cli.py sync --products
```

### Sync Test Data (Performance Benchmarks)
```bash
# Sync all available test data PDFs
python cli.py sync --test-data

# Sync specific config
python cli.py sync --test-data --config-id=15

# Sync with limit
python cli.py sync --test-data --limit=10
```

### Regenerate All Embeddings
```bash
python cli.py ingest --embeddings --regenerate
```

---

## Help

Get help for any command:
```bash
python cli.py setup --help
python cli.py ingest --help
python cli.py sync --help
python cli.py youtube --help
```

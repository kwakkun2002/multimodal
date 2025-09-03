# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multimodal AI project that implements a CLIP-based image-text retrieval system using Milvus vector database and MinIO object storage. The system can embed images and text using OpenAI's CLIP model, store images in MinIO, and perform similarity searches using Milvus.

## Development Environment

### Dependencies Management
- Uses `uv` as the Python package manager (pyproject.toml + uv.lock)
- Python 3.12+ required
- Key dependencies: transformers[torch], pymilvus, datasets, pillow, minio

### Infrastructure Setup
```bash
# Start Milvus, MinIO, and Attu (web UI) services
docker compose up -d

# Check services status
docker compose ps
```

### Python Environment
```bash
# Install dependencies (if uv is available)
uv sync

# Alternative with pip
pip install -e .
```

## Core Architecture

### MultiModalStore Class (`MultiModalStore.py`)
The main service class that orchestrates:
- **CLIP Model**: OpenAI CLIP-ViT-Large-Patch14 (768-dim embeddings)
- **MinIO**: Object storage for images (JPEG format)
- **Milvus**: Vector database with single vector field schema

**Key Methods**:
- `embed_text()`: Convert text to CLIP embeddings
- `embed_images()`: Convert PIL images to CLIP embeddings  
- `add_images_with_captions()`: Upload images + insert vectors
- `search_by_text()`: Text-to-image similarity search

### Data Pipeline
1. **Dataset**: Flickr30K from HuggingFace datasets
2. **Processing**: PIL-based image handling with batch collation
3. **Storage**: Images → MinIO, Vectors → Milvus, Metadata → Milvus
4. **Search**: Text queries → CLIP embedding → Vector similarity

## Common Development Tasks

### Running Data Ingestion
```bash
python addFlickrToMilvus.py
```

### Dataset Operations
```bash
# Split dataset (if needed)
python split_dataset.py

# View dataset samples
python view_dataset.py

# View CLIP embeddings
python view_clip_embed.py
```

### Services Management
```bash
# Access Attu web interface
# http://localhost:8000

# Access MinIO console  
# http://localhost:9001
# Credentials: minioadmin/minioadmin
```

## Configuration

### Service Endpoints
- Milvus: localhost:19530
- MinIO: localhost:9000  
- Attu UI: localhost:8000
- MinIO Console: localhost:9001

### Default Settings
- Collection: "openclip_multimodal"
- Vector dimension: 768
- Metric: Inner Product (IP)
- Index: HNSW (M=16, efConstruction=200)

## Important Notes

- All image processing uses PIL.Image format
- CLIP embeddings are L2-normalized  
- Batch processing recommended for large datasets
- Use `DRY_RUN=True` for testing without actual insertion
- Call `flush()` after batch insertions for consistency
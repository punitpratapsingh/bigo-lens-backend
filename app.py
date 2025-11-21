from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
import io
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import uuid
import json
import numpy as np


from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance



qdrant_client_app = QdrantClient(
    url="https://25dee531-7266-42c7-9ef7-55a6adc469c6.eu-central-1-0.aws.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.oPNVAu1YmZpM6mnGTHgHjj3P8HtR7TOvZHIk4HxEfKI",
)

# qdrant_client_app.create_collection(
#     collection_name="test",
#     vectors_config=VectorParams(size=768, distance=Distance.COSINE)
# )



#eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.oPNVAu1YmZpM6mnGTHgHjj3P8HtR7TOvZHIk4HxEfKI

app = FastAPI(title="SigLIP Image Embedding Service")

# Load SigLIP model once
model_name = "google/siglip-base-patch16-224"
processor = AutoProcessor.from_pretrained(model_name,use_fast=False)
model = AutoModel.from_pretrained(model_name)


# qdrant = QdrantClient("http://localhost:6333")  # change to your Qdrant host if remote
# collection_name = "image_vectors"

# # Create collection if not exists
# qdrant.recreate_collection(
#     collection_name=collection_name,
#     vectors_config=VectorParams(size=768, distance=Distance.COSINE)
# )


# api for search image
@app.post("/image-search")
async def embed_image(file: UploadFile = File(...)):
    # Read image

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess
    inputs = processor(images=image, return_tensors="pt")

    # Get embeddings
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        # Normalize
        embedding = features / features.norm(p=2, dim=-1, keepdim=True)

        embedding_vector = embedding.cpu().numpy()[0].tolist()

 # 🔥 Qdrant Search
    search_result = qdrant_client_app.search(
        collection_name="test",
        query_vector=embedding_vector,
        limit=10  # number of similar results
    )

    # Convert results
    formatted_results = []
    for r in search_result:
        formatted_results.append({
            "id": r.id,
            "score": r.score,
            "payload": r.payload
        })

    return JSONResponse({
       # "query_vector": embedding_vector,
        "status": "success",
        "code": 200,
        "results": formatted_results,

    })


# -----------------------------
# 1️⃣ BULK IMAGE UPLOAD + STORE
# -----------------------------
@app.post("/bulk-image-upload")
async def embed_bulk_images(
    files: list[UploadFile] = File(...),
    category: str = Form(None)
):
    images = []
    file_names = []

    for file in files:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        images.append(image)
        file_names.append(file.filename)

    # Preprocess batch
    inputs = processor(images=images, return_tensors="pt", padding=True)

    with torch.no_grad():
        features = model.get_image_features(**inputs)
        embeddings = features / features.norm(p=2, dim=-1, keepdim=True)
        embeddings = embeddings.cpu().numpy()
#    # Store vectors in Qdrant
    points = []
    for idx, vector in enumerate(embeddings):
        point_id = str(uuid.uuid4())
        metadata = {
            "filename": file_names[idx],
            "category": category if category else "unknown",
            "title": "Gold Earring",
            "price": 1499,
            "material": "18K Gold",
            "image_url": "https://cdn.com/earring1.jpg",
            "tags": ["jewelry", "earring", "gold"],
            "category": "earrings",
            "brand": "Aza",
            "color": "gold",
            "weight": 2.5, 
            "sku": "ER-4921"
        }

        points.append(
            PointStruct(id=point_id, vector=vector.tolist(), payload=metadata)
        )




    print('tetst--------',points)
    collection_name = "test"
    qdrant_client_app.upsert(collection_name=collection_name, points=points)



    response = [
        {"filename": file_names[i], "vector_id": points[i].id}
        for i in range(len(points))
    ]

    return JSONResponse({
        "status": "success",
        "code": 201,
        "message": f"{len(files)} images processed and stored successfully.",
        "data": response,
    })


# -----------------------------
# 2️⃣ IMAGE SIMILARITY SEARCH
# -----------------------------
@app.post("/similarity-search")
async def search_image(
    file: UploadFile = File(...),
    top_k: int = Form(5)
):
    # Read uploaded image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess
    inputs = processor(images=image, return_tensors="pt")

    # Compute vector
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        embedding = features / features.norm(p=2, dim=-1, keepdim=True)
        vector = embedding.cpu().numpy()[0].tolist()

    # Search in Qdrant
    search_results = qdrant.search(
        collection_name=collection_name,
        query_vector=vector,
        limit=top_k
    )

    # Format response
    matches = []
    for result in search_results:
        matches.append({
            "score": result.score,
            "id": result.id,
            "filename": result.payload.get("filename"),
            "category": result.payload.get("category")
        })

    return JSONResponse({
        "query_image": file.filename,
        "top_results": matches
    })

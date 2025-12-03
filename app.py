from download_model import download_model
download_model()
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import uuid
import onnxruntime as ort
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# ----------------------------
# LOAD ONNX SIGLIP-BASE MODEL
# ----------------------------
#ort_session = ort.InferenceSession("models/siglip_base.onnx")
ort_session = ort.InferenceSession("models/siglip_base.onnx")


# ----------------------------
# PREPROCESS FUNCTION (SigLIP) for image
# ----------------------------
def preprocess(image: Image.Image):
    image = image.resize((224, 224))
    array = np.array(image).astype(np.float32) / 255.0
    array = (array - 0.5) / 0.5  # Normalize
    array = np.transpose(array, (2, 0, 1))  # HWC → CHW
    return array[np.newaxis, :]  # Add batch dim


# ----------------------------
# QDRANT CONFIG
# ----------------------------
qdrant_client_app = QdrantClient(
    url="https://25dee531-7266-42c7-9ef7-55a6adc469c6.eu-central-1-0.aws.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.oPNVAu1YmZpM6mnGTHgHjj3P8HtR7TOvZHIk4HxEfKI",
)


print(qdrant_client_app.get_collections())   # should print {"status": "ok"}

# from qdrant_client.models import VectorParams, Distance

# qdrant_client_app.create_collection(
#    collection_name="test",
#    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
# )


COLLECTION = "test"


app = FastAPI(title="SigLIP ONNX Image Embedding Service")


# ----------------------------
# 1️⃣ IMAGE SEARCH
# ----------------------------
@app.post("/image-search")
async def embed_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess
    pixel_values = preprocess(image)

    # ONNX inference
    outputs = ort_session.run(None, {"pixel_values": pixel_values})
    print("ONNX output shape:", outputs[0].shape)   # should be (1, 197, 768)

    # Get CLS token embedding
    embedding = outputs[0][0][0]   # shape → (768,)
    print("Embedding shape BEFORE norm:", embedding.shape)
    print("Sample raw:", embedding[:10])

    # Normalize
    embedding = embedding / np.linalg.norm(embedding)
    print("Embedding shape AFTER norm:", embedding.shape)
    print("Sample normalized:", embedding)

    # Qdrant requires 1D list
    query_vector = embedding.tolist()
    print("Query vector length:", len(query_vector))

    # Qdrant search
    search_result = qdrant_client_app.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=10
    )
    print("Search result:", search_result)

    formatted_results = [
        {"id": r.id, "score": r.score, "payload": r.payload}
        for r in search_result
    ]

    return JSONResponse({
        "status": "success",
        "code": 200,
        "results": formatted_results
    })
   


# ----------------------------
# 2️⃣ BULK IMAGE UPLOAD + STORE
# ----------------------------
@app.post("/bulk-image-upload")
async def embed_bulk_images(
    files: list[UploadFile] = File(...),
    category: str = Form(None)
):
    points = []
    stored_info = []

    for file in files:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        x = preprocess(image)
        out = ort_session.run(None, {"pixel_values": x})

        emb = out[0].squeeze()          # 1D
        emb = emb / np.linalg.norm(emb)
        point_id = str(uuid.uuid4())
        emb = emb[0]
        print('emb shape----',emb.shape,emb)

        metadata = {
            "filename": file.filename,
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
            PointStruct(
                id=point_id,
                vector=emb.tolist(),
                payload=metadata
            )
        )

        stored_info.append({"filename": file.filename, "vector_id": point_id})
    
    # ---------- FIXED: BATCH UPSERT ----------
    BATCH_SIZE = 20
    for i in range(0, len(points), BATCH_SIZE):
        qdrant_client_app.upsert(
            collection_name=COLLECTION,
            points=points[i:i + BATCH_SIZE]
        )
    # -----------------------------------------

    return JSONResponse({
        "status": "success",
        "code": 201,
        "message": f"{len(files)} images processed and stored successfully.",
        "data": stored_info
    })


# ----------------------------
# 3️⃣ SIMILARITY SEARCH
# ----------------------------
@app.post("/similarity-search")
async def search_image(
    file: UploadFile = File(...),
    top_k: int = Form(5)
):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    pixel_values = preprocess(image)

    out = ort_session.run(None, {"pixel_values": pixel_values})
    emb = out[0].squeeze()              # <-- FIX flatten
    emb = emb / np.linalg.norm(emb)

    results = qdrant_client_app.search(
        collection_name=COLLECTION,
        query_vector=emb.tolist(),
        limit=top_k
    )

    formatted = [
        {
            "score": r.score,
            "id": r.id,
            "filename": r.payload.get("filename"),
            "category": r.payload.get("category")
        }
        for r in results
    ]

    return JSONResponse({
        "query_image": file.filename,
        "top_results": formatted
    })

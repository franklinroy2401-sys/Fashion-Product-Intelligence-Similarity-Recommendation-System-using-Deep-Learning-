import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model, Model
from sklearn.metrics.pairwise import cosine_similarity

IMG_SIZE = (224, 224)

# ==========================================================
# 1️. LOAD ALL ARTIFACTS
# ==========================================================
def load_artifacts():

    # Load datasets
    df_men = pd.read_csv("men_products.csv")
    df_full = pd.read_csv("fashion_products_clean.csv")

    # Load trained classification model
    model = load_model("men_fashion_classifier.keras")

    # Create feature extractor (remove softmax layer)
    feature_extractor = Model(
        inputs=model.inputs,
        outputs=model.layers[-2].output
    )

    # Load embeddings
    embeddings = np.load("men_embeddings.npy")

    # Load image paths
    with open("image_paths.pkl", "rb") as f:
        image_paths = pickle.load(f)

    # Load class names (IMPORTANT: exact training order)
    with open("class_names.pkl", "rb") as f:
        class_names = pickle.load(f)

    # Safety check
    if len(embeddings) != len(image_paths):
        raise ValueError("Mismatch between embeddings and image paths!")

    print("All artifacts loaded successfully ✅")

    return df_men, df_full, model, class_names, feature_extractor, embeddings, image_paths


# ==========================================================
# 2️. PREPROCESS UPLOADED IMAGE
# ==========================================================
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize(IMG_SIZE)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# ==========================================================
# 3️. PREDICT CATEGORY
# ==========================================================
def predict_category(img_array, model, class_names):

    prediction = model.predict(img_array, verbose=0)
    idx = np.argmax(prediction)

    predicted_class = class_names[idx]

    return predicted_class, idx


# ==========================================================
# 4️. EXTRACT FEATURE EMBEDDING
# ==========================================================
def extract_feature(img_array, feature_extractor):

    feature = feature_extractor.predict(img_array, verbose=0)
    feature = feature.flatten()

    # Normalize (mandatory for cosine similarity)
    norm = np.linalg.norm(feature)
    if norm != 0:
        feature = feature / norm

    return feature.reshape(1, -1)


# ==========================================================
# 5️. RECOMMEND SIMILAR PRODUCTS
# ==========================================================
def recommend_similar(feature, embeddings, image_paths, df_men, predicted_class, top_n=5):

    # Compute cosine similarity
    similarities = cosine_similarity(feature, embeddings)[0]

    # Sort highest similarity first
    sorted_indices = np.argsort(similarities)[::-1]

    results = []
    count = 0

    for idx in sorted_indices:

        image_path = image_paths[idx]

        # Extract product ID from filename
        try:
            product_id = int(os.path.splitext(os.path.basename(image_path))[0])
        except:
            continue

        # Fetch metadata
        product_rows = df_men[df_men["id"] == product_id]

        if product_rows.empty:
            continue

        row = product_rows.iloc[0]

        # Keep only same predicted category
        if row["articleType"] == predicted_class:

            results.append({
                "image_path": image_path,
                "articleType": row["articleType"],
                "brandName": row["brandName"],
                "baseColour": row["baseColour"],
                "season": row["season"]
            })

            count += 1

        if count == top_n:
            break

    return results
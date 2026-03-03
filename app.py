# ===========================================
# Fashion Product Intelligence & Similarity
# Streamlit Application (Final Version)
# ===========================================

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from utils import (
    load_artifacts,
    preprocess_image,
    predict_category,
    extract_feature,
    recommend_similar
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Fashion Product Intelligence",
    layout="wide"
)

# --------------------------------------------------
# LOAD ALL ARTIFACTS (Cached)
# --------------------------------------------------
@st.cache_resource
def load_all():
    return load_artifacts()

df_men, df, model, class_names, feature_extractor, embeddings, image_paths = load_all()

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Home", "EDA Insights", "Image-Based Recommendation"]
)

# ==========================================================
# 1️⃣ HOME PAGE
# ==========================================================
if page == "Home":

    st.title("👕 Fashion Product Intelligence & Similarity Recommendation System")

    # ------------------------------------------------------
    # PROJECT OVERVIEW
    # ------------------------------------------------------
    st.header("PROJECT OVERVIEW")
    st.markdown("""
This project builds an end-to-end **Fashion Product Intelligence and Similarity Recommendation System** using Deep Learning.

The pipeline begins with flattening complex nested JSON product data into a structured dataset, followed by detailed Exploratory Data Analysis to understand category distribution and trends.

A Convolutional Neural Network (**MobileNetV2**) with Transfer Learning is used to classify fashion products by article type. The trained model is further utilized as a feature extractor to generate image embeddings.

Cosine similarity is applied on these embeddings to recommend visually similar products within the predicted category, creating a real-world e-commerce style recommendation system.
""")

    # ------------------------------------------------------
    # DATASET DESCRIPTION
    # ------------------------------------------------------
    st.header("Dataset Description")
    st.markdown("""
The dataset consists of fashion product metadata extracted from nested JSON files.  
Each record represents a single product in the catalog and contains the following attributes:

• **id** – Unique identifier for each product  
• **productDisplayName** – Name displayed to customers  
• **brandName** – Brand associated with the product  
• **gender** – Target audience category (Men / Women / Kids)  
• **baseColour** – Primary color of the product  
• **season** –  Season for which the product is designed
• **year** – Year of product release  
• **usage** – Intended usage (Casual, Sports, Formal, etc.)  
• **articleType** – Specific product type (e.g., T-Shirts, Shirts, Shoes)  
• **masterCategory** – Broad product category (e.g., Apparel, Footwear)  
• **subCategory** – Sub-level classification  
• **imageURL** – URL used to download the product image  

The original data was stored in a complex nested JSON structure and was flattened into a structured tabular format using Pandas. Missing values were handled, categories were standardized, and duplicate or invalid records were removed to ensure data quality.
""")

    # ------------------------------------------------------
    # HIGH-LEVEL ARCHITECTURE
    # ------------------------------------------------------
    st.header("High-Level Architecture (Textual)")
    st.markdown("""
The system follows a three-stage architecture:

### 1. Data Processing Layer
- Raw fashion product data stored in nested JSON format  
- JSON flattened into structured CSV using Pandas  
- Data cleaning and preprocessing  
- Product images downloaded using imageURL  
- Images organized into category-based folder structure  

### 2. Model Development Layer
- Image dataset prepared using TensorFlow data pipeline  
- Transfer Learning applied using MobileNetV2 (pretrained on ImageNet)  
- Custom classification head added for articleType prediction  
- Model trained and saved as a reusable classifier  
- Feature extractor created by removing final softmax layer  
- Image embeddings generated and stored for similarity computation  

### 3. Recommendation & Deployment Layer
- User uploads a fashion product image  
- Image resized and preprocessed  
- Model predicts product category (articleType)  
- Feature embedding extracted  
- Cosine similarity computed against stored embeddings  
- Top-N visually similar products retrieved  
- Results displayed via Streamlit web interface  
""")

# ==========================================================
# 2️⃣ EDA INSIGHTS PAGE
# ==========================================================
elif page == "EDA Insights":

    st.title("📊 Exploratory Data Analysis (EDA) Insights")

    st.header("🔹 Key Charts")

    # 1. Product Count by Gender
    st.subheader("1. Product Count by Gender")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="gender", order=df["gender"].value_counts().index, ax=ax)
    for container in ax.containers:
        ax.bar_label(container)
    st.pyplot(fig)

    # 2. Master Category Distribution
    st.subheader("2. Master Category Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df, y="masterCategory",
                  order=df["masterCategory"].value_counts().index, ax=ax)
    for container in ax.containers:
        ax.bar_label(container)
    st.pyplot(fig)

    # 3. Top 10 Article Types
    st.subheader("3. Top 10 Article Types")
    top_articles = df["articleType"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(
        data=df[df["articleType"].isin(top_articles.index)],
        y="articleType",
        order=top_articles.index,
        ax=ax
    )
    for container in ax.containers:
        ax.bar_label(container)
    st.pyplot(fig)

    # 4. Top 10 Colors
    st.subheader("4. Top 10 Product Colors")
    top_colors = df["baseColour"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(
        data=df[df["baseColour"].isin(top_colors.index)],
        y="baseColour",
        order=top_colors.index,
        ax=ax
    )
    for container in ax.containers:
        ax.bar_label(container)
    st.pyplot(fig)

    # 5. Seasonal Trends
    st.subheader("5. Seasonal Product Trends")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(
        data=df,
        x="season",
        order=df["season"].value_counts().index,
        ax=ax
    )
    for container in ax.containers:
        ax.bar_label(container)
    st.pyplot(fig)

    # 6. Year-wise Distribution
    st.subheader("6. Year-wise Product Distribution")
    year_counts = df["year"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(year_counts.index, year_counts.values, marker="o")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Products")
    ax.grid(True)
    st.pyplot(fig)

    # 7. Top Brands
    st.subheader("7. Top 10 Brands by Product Count")
    top_brands = df["brandName"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_brands.values, y=top_brands.index, ax=ax)
    for container in ax.containers:
        ax.bar_label(container)
    st.pyplot(fig)

    # Category Insights
    st.header("🔹 Category-Wise Insights (Men)")
    st.markdown("""
### Article Type by Gender
- Clear differences are observed in article distribution across genders
- Men category is dominated by shirts, t-shirts, and trousers
- Women category shows higher variation in fashion-oriented products
- The Men category demonstrates a more balanced spread among core apparel types, making it stable for modeling


### Casual vs Sports vs Formal Usage
- Casual wear dominates across both Men and Women categories
- Sports and Formal segments contribute comparatively fewer products
- Men category shows a relatively balanced distribution between Casual and Formal usage
- The dominance of Casual wear makes it ideal for similarity-based recommendation modeling 
                
### Top 10 Subcategory 
- Apparel-based subcategories contribute the highest product volume
- Clear clustering of products is observed within specific subcategories
- The hierarchical structure (MasterCategory → SubCategory → ArticleType) enables structured classification
- Strong subcategory concentration supports focused deep learning training
             

### Top Brands in Men Category
- A few major brands dominate the Men category.
- Brand distribution appears relatively balanced among top players.
- Strong brand presence ensures visual diversity in product design.
- This diversity improves feature learning in CNN models.
""")

    st.header("🎯 Chosen Category Explanation")
    st.markdown("""
**Chosen Category: Men**
- High product count  
- Balanced article type distribution  
- Strong brand presence  
- Adequate data for deep learning  
Ideal for classification & similarity-based recommendation.
""")

# ==========================================================
# 3️⃣ IMAGE-BASED RECOMMENDATION PAGE
# ==========================================================
elif page == "Image-Based Recommendation":

    st.title("🖼 Image-Based Fashion Recommendation (Men Category)")

    st.markdown("""
**Instructions:**  
- Upload a product image  
- System will predict its **product category**  
- Display the **uploaded image**  
- Show **Top-N visually similar products**
""")

    uploaded_file = st.file_uploader(
        "Upload a product image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

        with st.spinner("Analyzing image and finding similar products..."):

            img_array = preprocess_image(uploaded_file)

            predicted_class, idx = predict_category(
                img_array, model, class_names
            )

            feature = extract_feature(
                img_array, feature_extractor
            )

            recommendations = recommend_similar(
                feature,
                embeddings,
                image_paths,
                df_men,
                predicted_class,
                top_n=5
            )

        st.subheader("Predicted Product Category")
        st.markdown(f"**{predicted_class}**")

        st.subheader("Top 5 Recommended Similar Products")

        if recommendations:
            cols = st.columns(len(recommendations))
            for rec, col in zip(recommendations, cols):
                with col:
                    sim_image = Image.open(rec["image_path"])
                    st.image(sim_image, use_container_width=True)
                    st.markdown(
                        f"**{rec['articleType']}**  \n"
                        f"{rec['brandName']} | "
                        f"{rec['baseColour']} | "
                        f"{rec['season']}"
                    )
        else:
            st.warning("No similar products found.")
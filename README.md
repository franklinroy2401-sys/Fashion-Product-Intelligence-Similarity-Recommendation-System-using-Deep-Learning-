**Fashion Product Intelligence \& Similarity Recommendation System**



An end-to-end Deep Learning–based Fashion Product Classification and Image Similarity Recommendation System built using Transfer Learning and deployed with Streamlit.



**Project Overview**



Modern e-commerce platforms store product data in complex nested JSON structures. Transforming such raw, unstructured data into an intelligent recommendation system requires strong data engineering and machine learning capabilities.



This project builds a complete Fashion Product Intelligence System that:



* Parses and flattens nested JSON product data



* Performs structured data cleaning and validation



* Conducts Exploratory Data Analysis (EDA)



* Trains a Deep Learning image classifier using Transfer Learning



* Extracts CNN feature embeddings



* Implements cosine similarity–based product recommendation



* Deploys an interactive Streamlit application



**Tech Stack**



* Python 3.11



* TensorFlow / Keras



* NumPy



* Pandas



* Scikit-learn



* Pillow



* Streamlit



* Matplotlib / Seaborn



The project was developed and tested using Python 3.11 for compatibility with deep learning libraries



**Project Workflow**



Raw JSON Files

&nbsp;       ↓

JSON Flattening \& Cleaning

&nbsp;       ↓

Structured CSV Dataset

&nbsp;       ↓

Men Category Selection

&nbsp;       ↓

Image Download \& Validation

&nbsp;       ↓

Transfer Learning (MobileNetV2)

&nbsp;       ↓

Feature Extraction (Embeddings)

&nbsp;       ↓

Cosine Similarity Engine

&nbsp;       ↓

Streamlit Deployment



**Phase 1: Data Engineering \& Exploratory Analysis**



**JSON Flattening**



\-Loaded nested product JSON files



\-Extracted key attributes including:



&nbsp;  -id



&nbsp;  -productDisplayName



&nbsp;  -brandName



&nbsp;  -gender



&nbsp;  -masterCategory



&nbsp;  -subCategory



&nbsp;  -articleType



&nbsp;  -baseColour



&nbsp;  -season



&nbsp;  -year



&nbsp;  -usage



&nbsp;  -imageURL



* Converted structured data into a Pandas DataFrame



* Saved as **fashion\_products.csv**



**Data Cleaning \& Validation**



Performed:



* Removal of rows missing critical fields



* Standardization of categorical values



* Duplicate image URL removal



* Metadata consistency checks



* Programmatic image verification using Pillow



* Removal of corrupted and unreadable images



* Visual dataset validation to remove cross-category inconsistencies



Saved cleaned dataset as:



"fashion\_products\_clean.csv"



Final modeling dataset:



"men\_products.csv"



**Dataset Statistics**



* Original Men dataset: 16,484 images



* After validation and cleaning: 15,726 images



* Final dataset used for modeling: 15,726 validated samples



**Phase 2: Deep Learning \& Similarity Recommendation Engine**



**Image Classification Model**



**Model Used:** MobileNetV2 (Transfer Learning)



Configuration:



* Input size: 224 × 224



* Data augmentation applied



* Base model initially frozen



* Fine-tuning performed



* EarlyStopping used



Saved trained model:



"men\_classifier.keras" (**tracked with Git LFS**)



**Feature Extraction (Embeddings)**



* Removed final classification layer



* Extracted feature vectors from the penultimate layer



* Normalized embeddings



* Generated embeddings for all dataset images



Saved artifacts:



* men\_embeddings.npy



* image\_paths.pkl



* class\_names.pkl



Each image is represented as a fixed-length numerical feature vector.



**Similarity Recommendation Engine**



* Implemented cosine similarity search



* For a given query image:



&nbsp;    -Compute embedding



&nbsp;    -Compare with dataset embeddings



&nbsp;    -Retrieve Top-N most similar products



* Map similarity results back to product metadata



**Streamlit Application**



Application Flow:



1\.User uploads an image



2\.System predicts product category



3\.Extracts image embedding



4\.Retrieves visually similar products



5\.Displays recommendations



Run locally:

pip install -r requirements.txt

streamlit run app.py



**Project Structure:**



Fashion-Product-Intelligence/



* men\_products.csv
* men\_classifier.keras
* men\_embeddings.npy
* image\_paths.pkl
* class\_names.pkl
* JSON\_to\_csv.py
* app.py
* utils.py
* requirements.txt
* README.md



**Conclusion:**

This project shows a full machine learning workflow from start to finish: cleaning data, training a model, extracting features, finding similar items, and deploying the system. It is organized step by step, like how real-world ML projects are built.








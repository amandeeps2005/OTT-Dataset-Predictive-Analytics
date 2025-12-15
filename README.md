# 🎬 Predictive Analysis of Student OTT Consumption Patterns

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📌 Project Overview
This project analyzes the streaming habits of university students to understand the impact of **Over-The-Top (OTT)** platforms on financial behavior, sleep health, and content consumption. 

Using a primary dataset of **500 student responses**, the project implements the full Data Science lifecycle—from data cleaning and EDA to building predictive models using Supervised and Unsupervised Learning.

## 📊 Key Analyses & Algorithms
The project is divided into four main analytical units:

| Analysis Type | Algorithm Used | Objective | Key Finding |
| :--- | :--- | :--- | :--- |
| **Regression** | Linear Regression | Predict Monthly Spending | Spending is linearly dependent on the **Number of Apps** installed ($R^2 = 0.85$). |
| **Classification** | Decision Tree | Identify "Binge-Watchers" | **Sleep Disruption** is the strongest predictor of binge-watching behavior (91% Recall). |
| **Clustering** | K-Means | User Segmentation | Identified 3 profiles: **Casual Viewers**, **Weekend Bingers**, and **Heavy Users**. |
| **Association** | Apriori Algorithm | Market Basket Analysis | Strong correlation found between **Anime** and **Action** genres (100% Confidence). |

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** * `Pandas` & `NumPy` (Data Manipulation)
    * `Matplotlib` & `Seaborn` (Visualization)
    * `Scikit-Learn` (Regression, Classification, Clustering)
    * `Mlxtend` (Association Rule Mining)

## 📂 Dataset
The primary dataset used for this analysis contains 500 rows of student response data. You can access and download the full CSV file from the link below:

👉 **[Download Dataset (Google Drive)](https://drive.google.com/drive/folders/1k3S-LHsUP1xNp_zgT3TehVyGTLo8pTno?usp=sharing)**

## 🚀 How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/OTT-Consumption-Analysis.git](https://github.com/your-username/OTT-Consumption-Analysis.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn mlxtend
    ```
3.  **Run the script:**
    Ensure the CSV path is correct in the script, then run:
    ```bash
    python main_analysis.py
    ```

## 📈 Visualizations
The script generates the following insights:
1.  **Correlation Heatmap** (checking feature relationships)
2.  **Regression Line** (Actual vs. Predicted Spending)
3.  **Confusion Matrix** (Binge-Watcher Classification accuracy)
4.  **Cluster Scatter Plot** (Weekday vs. Weekend consumption segments)

## 👤 Author
**Amandeep Singh** *Lovely Professional University*

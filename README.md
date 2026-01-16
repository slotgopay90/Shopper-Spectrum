# 🛒 Shopper Spectrum: Customer Segmentation & Product Recommendation System

An end-to-end **E-Commerce Analytics** project that performs:
- ✅ **Customer Segmentation** using **RFM Analysis + KMeans Clustering**
- ✅ **Product Recommendations** using **Item-Based Collaborative Filtering (Cosine Similarity)**
- ✅ Deployment using an interactive **Streamlit Web App**

---

## 🎯 Objective
Analyze e-commerce transaction data to:
1. Segment customers into meaningful groups (High-Value, Regular, Occasional, At-Risk)
2. Recommend similar products based on purchase behavior
3. Provide a real-time web interface for both features

---

## 📂 Dataset
**File:** `online_retail.csv`  
**Columns Used:**
- InvoiceNo  
- StockCode  
- Description  
- Quantity  
- InvoiceDate  
- UnitPrice  
- CustomerID  
- Country  

---

## 🧹 Data Cleaning
Steps applied:
- Removed rows with missing `CustomerID`
- Removed rows with missing `Description`
- Removed cancelled invoices (`InvoiceNo` starting with **C**)
- Removed invalid records (`Quantity <= 0` or `UnitPrice <= 0`)
- Converted `InvoiceDate` to datetime
- Removed duplicates
- Created `TotalPrice = Quantity × UnitPrice`

---

## 📊 Exploratory Data Analysis (EDA)
Visual insights include:
- Transaction volume by country
- Top 10 selling products
- Monthly and daily sales trends
- Monetary distribution per transaction
- Customer spending distribution
- RFM distributions
- Elbow curve for clustering
- Product similarity heatmap

---

## 🧠 Customer Segmentation (RFM + KMeans)
**RFM Definition**
- **Recency:** Days since last purchase  
- **Frequency:** Number of unique invoices  
- **Monetary:** Total spending  

**Modeling**
- Standardized RFM features using `StandardScaler`
- Applied **KMeans clustering**
- Selected optimal clusters using **Elbow Method + Silhouette Score**
- Assigned business segment labels:
  - 💎 High-Value
  - 🙂 Regular
  - 🛒 Occasional
  - ⚠️ At-Risk

---

## 🛍 Recommendation System (Item-Based Collaborative Filtering)
Approach:
- Built a **Product × Customer** pivot table using Quantity
- Computed **Cosine Similarity** between products
- Given a product name, recommends **Top 5 similar products**
- Handles invalid product names gracefully

---

## 🌐 Streamlit Web Application
The Streamlit app contains two modules:

### 🛍 Module 1: Product Recommendation
- Input: Product name  
- Output: Top 5 similar products  

### 🎯 Module 2: Customer Segmentation
- Inputs: Recency, Frequency, Monetary  
- Output: Predicted customer segment  

---

## 🛠 Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Streamlit
- Pickle (Model Saving)

---

## 📦 Project Structure

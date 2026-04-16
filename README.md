# 🛒 Smart Cart — Customer Segmentation System
### Unsupervised ML · PCA · K-Means · Agglomerative Clustering · 4-Segment Profiling

---

## 📌 Project Overview

**Smart Cart** is a customer segmentation system built on top of a retail marketing dataset. Using unsupervised machine learning, the system groups customers into **4 distinct behavioral segments** based on their income, spending patterns, demographics, and purchase history.

The insights from this segmentation can power:
- 🎯 **Targeted marketing campaigns** for each customer group
- 🛍️ **Personalized product recommendations** inside the smart cart
- 💡 **Customer lifetime value** analysis and churn prevention

The pipeline covers preprocessing, feature engineering, PCA-based dimensionality reduction, optimal cluster detection, and cluster profiling.

---

## 📁 Project Structure

```
```
├── 📁 data
│   ├── 📁 processed
│   │   └── 📄 processed_data.csv
│   └── 📁 raw
│       └── 📄 smartcart_customers.csv
├── 📁 notebooks
│   ├── 📄 01_Data_preprocessing & EDA.ipynb
│   └── 📄 02_Model.ipynb
├── ⚙️ .gitignore
├── 📝 README.md
└── 📄 requirements.txt
```

---

## 📊 Dataset

**Source:** Customer Personality Analysis (retail marketing dataset)

| Property | Value |
|---|---|
| File | `smartcart_customers.csv` |
| Task Type | **Unsupervised — Clustering** |
| Key Raw Features | Income, Year_Birth, Education, Marital_Status, Spending per category, Campaign responses |
| Spending Categories | Wines, Fruits, Meat Products, Fish Products, Sweet Products, Gold Products |

---

## ⚙️ Pipeline

### 1 · Data Preprocessing & EDA (`01_Data_preprocessing___EDA.ipynb`)

#### Missing Values
- `Income` → filled with **median** (robust to outliers)

#### Date & Age Engineering
| Derived Feature | Source | Logic |
|---|---|---|
| `Age` | `Year_Birth` | `1996 − Year_Birth` |
| `Customer_Tenure_Days` | `Dt_Customer` | Days from join date to latest date in dataset |

#### Aggregate Feature Engineering
| Feature | Formula | Purpose |
|---|---|---|
| `Total_Spending` | Sum of all 6 product category spends | Single composite spend signal |
| `Total_Children` | `Kidhome + Teenhome` | Household family load |

#### Categorical Consolidation

**Education** — simplified into 3 tiers:

| Original | Mapped To |
|---|---|
| Basic, 2n Cycle | `Undergraduate` |
| Graduation | `Graduate` |
| Master, PhD | `Postgraduate` |

**Marital Status** — mapped to household living arrangement:

| Original | Mapped To |
|---|---|
| Married, Together | `Partner` |
| Single, Divorced, Widow, Absurd, YOLO | `Alone` |

#### Outlier Removal
| Column | Condition Applied |
|---|---|
| `Age` | Kept `Age < 90` (removed implausible ages) |
| `Income` | Kept `Income < 600,000` (removed extreme outliers) |

#### Columns Dropped
- Identifiers: `ID`, `Year_Birth`, `Marital_Status`, `Kidhome`, `Teenhome`, `Dt_Customer`
- Individual spend cols replaced by `Total_Spending`: `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`

#### Encoding
- `OneHotEncoder` applied to `Education` and `Living_With` (encoded in-place, index-aligned)

#### EDA Visualizations
- Pairplot across `Income`, `Recency`, `Response`, `Age`, `Total_Spending`, `Total_Children`
- Correlation heatmap (annotated, coolwarm palette)

---

### 2 · Modelling (`02_Model.ipynb`)

#### Scaling
- `StandardScaler` applied to all features before PCA

#### Dimensionality Reduction — PCA
| Config | Purpose |
|---|---|
| `n_components=2` | 2D scatter plot to spot cluster shapes |
| `n_components=3` | 3D scatter for richer structure visualization |

PCA reduces high-dimensional encoded features into principal components that capture maximum variance — making clustering more effective and visually interpretable.

#### Optimal K Selection

Two methods were used in parallel:

**1 · Elbow Method** (WCSS minimization via `KneeLocator`):
```python
knee = KneeLocator(range(1, 11), wcss, curve="convex", direction="decreasing")
optimal_k = knee.elbow   # → 4
```

**2 · Silhouette Score** (cohesion vs. separation):
```python
for k in range(2, 11):
    score = silhouette_score(X_pca, labels)
```

Both methods were plotted on a **dual-axis combined chart** (WCSS on left, Silhouette Score on right) to confirm the optimal cluster count.

> ✅ **Optimal K = 4** confirmed by both methods.

#### Clustering Algorithms

| Algorithm | Config | Notes |
|---|---|---|
| **K-Means** | `n_clusters=4`, `random_state=42` | Fast, centroid-based; fit on PCA-reduced data |
| **Agglomerative Clustering** | `n_clusters=4`, `linkage='ward'` | Hierarchical; Ward minimizes within-cluster variance |

Both algorithms were visualized in **3D PCA space** with color-coded scatter plots.

#### Final Labels
Agglomerative Clustering labels were used for the final cluster assignment:
```python
X["cluster"] = labels_agg
```

---

## 👥 Customer Segments (4 Clusters)

Clusters were profiled using **Income vs. Total Spending scatter plots** and a **cluster-wise mean summary table**:

| Cluster | Label (Inferred) | Typical Profile |
|---|---|---|
| 0 | 💰 High-Value Shoppers | High income · High spending |
| 1 | 🏠 Family Savers | Moderate income · Children · Lower spending |
| 2 | 🌱 Young Budget Buyers | Lower income · Low spend · High recency |
| 3 | 🎯 Engaged Mid-Tier | Mid income · Active campaign responders |

> ℹ️ Exact cluster labels and characteristics should be confirmed from the `cluster_summary` mean table printed during notebook execution.

---

## 📈 Visual Outputs

| Plot | What It Shows |
|---|---|
| Pairplot | Bivariate relationships between key features |
| Correlation Heatmap | Feature correlation matrix (annotated) |
| 2D PCA Scatter | Raw cluster shape in 2 components |
| 3D PCA Scatter | Cluster separation across 3 components |
| Elbow Curve | WCSS vs. K — inflection at optimal K |
| Silhouette Plot | Cluster quality score vs. K |
| Dual-Axis Combined Plot | WCSS + Silhouette overlaid for K selection |
| Cluster Count Bar | Distribution of customers across clusters |
| Income vs. Spending Scatter | Cluster characterization by spend behaviour |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.x |
| Environment | JupyterLab + Miniconda |
| Data Manipulation | pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML — Preprocessing | scikit-learn (`StandardScaler`, `OneHotEncoder`) |
| ML — Reduction | scikit-learn (`PCA`) |
| ML — Clustering | scikit-learn (`KMeans`, `AgglomerativeClustering`, `silhouette_score`) |
| Elbow Detection | `kneed` (`KneeLocator`) |

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/smart-cart-segmentation.git
cd smart-cart-segmentation
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kneed
```

### 3. Run the notebooks in order

```
01_Data_preprocessing___EDA.ipynb   →  Cleans data, engineers features, saves processed_data.csv
02_Model.ipynb                       →  PCA, K selection, K-Means, Agglomerative, cluster profiling
```

---

## 🔮 Future Improvements

- [ ] Add **DBSCAN** as a density-based alternative for non-globular cluster shapes
- [ ] Build a **Streamlit dashboard** to visualize customer segments interactively
- [ ] Integrate a **product recommendation engine** based on cluster spending patterns
- [ ] Use **t-SNE or UMAP** for richer non-linear dimensionality visualization
- [ ] Automate cluster labeling using cluster-mean threshold rules
- [ ] Track segment migration over time (cohort analysis)

---

## 👤 Author

**Kiran Metri** 

---

## 📄 License

This project is intended for educational and portfolio purposes.
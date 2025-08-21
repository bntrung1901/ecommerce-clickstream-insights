# E-commerce Customer Insights & Clickstream Analytics

End-to-end analysis of a multi-category e-commerce clickstream (10–11/2019) using **Dask** for big data,
with a **Power BI** dashboard for business storytelling and a **RFM** customer segmentation.  
This repo is recruiter-friendly and reproducible.

## 🔎 What’s inside
- **Traffic analysis** by day/hour/weekday and a 11.11 sale spike.
- **Funnel & conversion** (View→Cart→Purchase) with pre/sale/post breakdown.
- **Category performance** (Top categories and conversion by stage).
- **RFM segmentation** with extended segments and revenue contribution.

> Dashboard key metrics (from the PDF in `docs/`): **64M Total Views**, **3M Carts**, **917K Purchases**, overall **1.44% conversion**.  
> Daily funnel averages: **3.13% View→Cart**, **72.13% Cart→Purchase**, **1.67% View→Purchase**.  
> RFM overview: **~697.5K customers**, avg **Recency = 25**, **Frequency = 3**, **Avg TotalRevenue ≈ 63.14M**.

## 🗂 Repository structure
```
Ecommerce-clickstream-dashboard/
├─ Power BI/
│  ├─ Ecommerce Customer Insights Dashboard.pbix
│  ├─ Ecommerce Customer Insights Dashboard.pbit
│  └─ Ecommerce Customer Insights Dashboard.pdf              
├─ Data/
│  └─ Processed/
│     ├─ traffic_overview.csv
│     ├─ funnel.csv
│     ├─ category.csv
│     ├─ rfm_extended.csv
│     └─ rfm_summary.csv
├─ Notebooks/
│  └─ Ecommerce_clickstream_analysis.ipynb
├─ Src/
│  └─ ecommerce_clickstream.py
├─ Assets/
│  └─ Screenshots/
│     ├─ Traffic Overview.png
│     ├─ Funnel & Conversion.png
│     ├─ Category Performance.png
│     └─ Customer Segmentation (RFM).png
├─ README.md
├─ requirements.txt
├─ LICENSE
├─ .gitignore
└─ .gitattributes
```

## ⚙️ Setup
```bash
# 1) Create & activate a virtual environment (optional)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt
```

## 🚀 Run the analysis
1. Download the original dataset locally (Kaggle: https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store?resource=download).
2. Place the CSV files in `data/`.
3. **Open `src/Ecommerce_clickstream.py` and update the `path` glob** to your machine, e.g.:
   ```python
   # before (example Windows path)
   path = r"D:\...\*.csv"
   # after (relative to repo)
   path = r"./data/*.csv"
   ```
4. Run:
   ```bash
   python src/Ecommerce_clickstream.py
   ```
5. The script will export 5 CSVs used by the dashboard to the repo root (or working dir):
   - `traffic.csv`
   - `funnel.csv`
   - `category.csv`
   - `rfm_extended.csv`
   - `rfm_summary.csv`

## 📊 Dashboard
- Open the Power BI report (not included due to size) or use the provided **PDF** in `docs/` as a visual reference.
- If you want to re-build the dashboard, import the 5 CSVs above and connect visuals according to the field names.

## 🧠 Techniques
- **Big data with Dask**: lazy compute, chunked CSV reading, groupby aggregations.
- **Funnel analysis** with pre/sale/post 11.11 split.
- **Category funnel** with safeguards for divide-by-zero and anomaly checks.
- **RFM** segmentation + extended business-friendly labels.

## 📌 Results highlights
- 11.11 creates a pronounced traffic spike and improved **View→Cart** & **View→Purchase**; **Cart→Purchase** stabilizes post-sale.
- Electronics & Appliances lead **View→Purchase**; Unknown needs labeling cleanup.
- Top RFM segments include **Loyal Customers, Need Attention, Hibernating, New Customers**.

## 📝 Notes
- The dataset is large; Dask helps run on a laptop but expect minutes for first computations.
- If your `.pbix` exceeds 100 MB, consider Git LFS (`git lfs track "*.pbix"`).

## 📄 License
MIT — see `LICENSE`.

---

**Tác giả:** Leo — MIT License.
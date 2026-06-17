# Mini-Project-Bola

Prediksi hasil pertandingan FIFA World Cup menggunakan Machine Learning, dilengkapi analisis korelasi ekonomi negara (GDP per capita) terhadap performa tim.

Mini-project untuk trial class / webinar 2 jam — topik AI & Data Science untuk pemula.

---

## Datasets

### 1. International Football Results (1872–2024)
**Sumber:** [Kaggle — martj42](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)

Dataset utama berisi hasil pertandingan sepak bola internasional dari tahun 1872 sampai sekarang (~45.000+ pertandingan). Dari dataset ini kita filter hanya pertandingan **FIFA World Cup**.

File yang dipakai:
| File | Keterangan |
|---|---|
| `results.csv` | Data utama — tanggal, tim home/away, skor, jenis turnamen, lokasi |
| `former_names.csv` | Mapping nama negara yang sudah berganti nama (misal: tidak dipakai langsung di pipeline saat ini, tersedia sebagai referensi) |
| `goalscorers.csv` | Detail pencetak gol per pertandingan (tidak dipakai di pipeline utama) |
| `shootouts.csv` | Hasil adu penalti (tidak dipakai di pipeline utama) |

### 2. GDP per Capita (current US$)
**Sumber:** [World Bank](https://data.worldbank.org/indicator/NY.GDP.PCAP.CD)

Data pendapatan rata-rata per kapita tiap negara dari tahun 1960–2025. Dipakai untuk menjawab pertanyaan: **apakah negara yang lebih kaya cenderung lebih sering menang di World Cup?**

File yang dipakai:
| File | Keterangan |
|---|---|
| `gdp_per_capita.csv` | Data GDP per capita tiap negara per tahun (format World Bank, 4 baris header) |
| `gdp_country_metadata.csv` | Info tambahan per negara: region, income group — berguna untuk mapping nama negara |

---

## Struktur Project

```
Mini-Project-Bola/
├── data/
│   ├── raw/                        # Dataset asli dari Kaggle & World Bank, tidak diubah
│   └── processed/                  # Data hasil cleaning, siap dipakai EDA & modeling
│       ├── wc_clean.csv            # Data World Cup bersih + fitur hasil feature engineering
│       └── win_rate_per_team.csv   # Win rate historis tiap tim
├── notebooks/
│   ├── 01_data_understanding.ipynb # Eksplorasi awal: cek shape, kolom, missing values
│   ├── 02_data_cleaning.ipynb      # Filter World Cup, buat target W/D/L, feature engineering
│   ├── 03_eda.ipynb                # Visualisasi & analisis korelasi (win rate, GDP, dst)
│   └── 04_modeling.ipynb           # Train & evaluasi Logistic Regression + Random Forest
├── models/
│   └── model.pkl                   # Model terbaik (otomatis terpilih berdasar akurasi) siap dipakai app
├── app/
│   └── app.py                      # Streamlit app — antarmuka prediksi interaktif
├── requirements.txt                # Daftar dependency Python
├── README.md                       # Dokumentasi project (file ini)
└── CLAUDE.md                       # Panduan kerja untuk Claude Code di repo ini (di-gitignore)
```

---

## Tech Stack

- Python, Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn (Logistic Regression, Random Forest)
- Streamlit (deployment)

---

## Alur Project

1. **Data Understanding** — eksplorasi dataset awal
2. **Data Cleaning** — filter, missing values, feature engineering
3. **EDA** — visualisasi & analisis korelasi GDP vs win rate
4. **Modeling** — train Logistic Regression & Random Forest, evaluasi model, simpan model dengan akurasi tertinggi
5. **Streamlit App** — deploy prediksi interaktif

---

## Cara Menjalankan

```bash
pip install -r requirements.txt
jupyter notebook
streamlit run app/app.py
```

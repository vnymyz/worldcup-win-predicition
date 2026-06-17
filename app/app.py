import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Prediksi World Cup",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — desain UI/UX
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #0f1c2e 0%, #15233a 100%);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 980px;
        }

        /* Hero header */
        .hero {
            text-align: center;
            padding: 1.2rem 1rem 1.6rem 1rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            margin-bottom: 1.2rem;
            box-shadow: 0 8px 24px rgba(0,0,0,0.25);
        }
        .hero h1 {
            font-size: 2.1rem;
            color: #ffffff;
            margin-bottom: 0.3rem;
        }
        .hero p {
            color: #cfe0ff;
            font-size: 1rem;
            margin: 0;
        }
        .badge {
            display: inline-block;
            background: rgba(255,255,255,0.15);
            color: #ffffff;
            padding: 0.25rem 0.9rem;
            border-radius: 999px;
            font-size: 0.85rem;
            margin-top: 0.6rem;
            border: 1px solid rgba(255,255,255,0.25);
        }

        /* Card */
        .card {
            background: #1b2a40;
            border-radius: 16px;
            padding: 1.4rem;
            border: 1px solid #2a3c57;
            margin-bottom: 1rem;
        }

        /* VS matchup */
        .vs-circle {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
        }
        .vs-text {
            font-size: 1.6rem;
            font-weight: 800;
            color: #ffd166;
            text-align: center;
        }

        /* Result banner */
        .result-banner {
            text-align: center;
            padding: 1.3rem;
            border-radius: 16px;
            font-size: 1.4rem;
            font-weight: 800;
            color: white;
            margin-bottom: 1rem;
        }
        .result-win   { background: linear-gradient(135deg, #2ecc71, #27ae60); }
        .result-draw  { background: linear-gradient(135deg, #f6c343, #f4b400); color:#3a2d00; }
        .result-lose  { background: linear-gradient(135deg, #e74c3c, #c0392b); }

        .reason-box {
            background: #1b2a40;
            border: 1px solid #2a3c57;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            color: #dce8ff;
            margin-bottom: 1.2rem;
            font-size: 0.95rem;
            line-height: 1.5;
        }

        /* Tombol prediksi — soft, tidak terlalu mencolok */
        div[data-testid="stButton"] button[kind="primary"] {
            background: linear-gradient(135deg, #4d8af0, #3a6fd8) !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            padding: 0.55rem 1.4rem !important;
            box-shadow: 0 4px 14px rgba(77,138,240,0.35) !important;
            transition: all 0.15s ease-in-out;
        }
        div[data-testid="stButton"] button[kind="primary"]:hover {
            background: linear-gradient(135deg, #5d97f3, #4878dd) !important;
            box-shadow: 0 6px 18px rgba(77,138,240,0.45) !important;
        }

        /* Step guide */
        .step-box {
            background: #1b2a40;
            border-left: 4px solid #4d8af0;
            padding: 0.8rem 1.1rem;
            border-radius: 10px;
            margin-bottom: 0.6rem;
            color: #dce8ff;
        }
        .step-num {
            display: inline-block;
            background: #4d8af0;
            color: white;
            width: 26px;
            height: 26px;
            border-radius: 50%;
            text-align: center;
            line-height: 26px;
            font-weight: 700;
            margin-right: 0.5rem;
        }

        /* Expander "Cara Menggunakan" jadi lebih jelas seperti tombol */
        .stExpander {
            border: 1.5px solid #4d8af0 !important;
            border-radius: 12px !important;
            background: #1b2a40 !important;
        }
        .stExpander summary {
            font-size: 1.05rem !important;
            font-weight: 700 !important;
            color: #ffffff !important;
            padding: 0.7rem 1rem !important;
        }
        .stExpander summary:hover {
            color: #ffd166 !important;
        }

        /* Footer */
        .footer-note {
            text-align: center;
            color: #7c92b3;
            font-size: 0.82rem;
            margin-top: 1.5rem;
        }

        /* Responsive tweaks */
        @media (max-width: 640px) {
            .hero h1 { font-size: 1.5rem; }
            .hero p  { font-size: 0.88rem; }
            .vs-text { font-size: 1.2rem; }
            .result-banner { font-size: 1.1rem; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open("models/model.pkl", "rb") as f:
        return pickle.load(f)

payload = load_model()
model = payload["model"]
model_name = payload["model_name"]
features = payload["features"]
win_rate_dict = payload["win_rate_dict"]
label_map = payload["label_map"]
accuracy = payload["accuracy"]

teams = sorted(win_rate_dict.keys())

FLAGS = {
    "Brazil": "🇧🇷", "Germany": "🇩🇪", "Argentina": "🇦🇷", "France": "🇫🇷",
    "Spain": "🇪🇸", "Portugal": "🇵🇹", "England": "🏴", "Italy": "🇮🇹",
    "Netherlands": "🇳🇱", "Belgium": "🇧🇪", "Uruguay": "🇺🇾", "Croatia": "🇭🇷",
    "Mexico": "🇲🇽", "United States": "🇺🇸", "Japan": "🇯🇵", "South Korea": "🇰🇷",
    "Indonesia": "🇮🇩", "Qatar": "🇶🇦", "Morocco": "🇲🇦", "Senegal": "🇸🇳",
    "Switzerland": "🇨🇭", "Poland": "🇵🇱", "Denmark": "🇩🇰", "Sweden": "🇸🇪",
    "Russia": "🇷🇺", "Serbia": "🇷🇸", "Australia": "🇦🇺", "Canada": "🇨🇦",
    "Ghana": "🇬🇭", "Cameroon": "🇨🇲", "Tunisia": "🇹🇳", "Saudi Arabia": "🇸🇦",
    "Iran": "🇮🇷", "Costa Rica": "🇨🇷", "Ecuador": "🇪🇨", "Wales": "🏴",
    "Scotland": "🏴", "Colombia": "🇨🇴", "Chile": "🇨🇱", "Peru": "🇵🇪",
}

def flag(team):
    return FLAGS.get(team, "⚽")

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <h1>⚽ Prediksi Hasil FIFA World Cup</h1>
        <p>Pilih dua tim dan lihat prediksi hasil pertandingan berbasis data historis World Cup</p>
        <span class="badge">Model: {model_name} &nbsp;·&nbsp; Akurasi: {accuracy*100:.1f}%</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Panduan penggunaan
# ---------------------------------------------------------------------------
with st.expander("Klik di sini — Cara Menggunakan App Ini", expanded=False):
    st.markdown(
        """
        <div class="step-box"><span class="step-num">1</span> Pilih <b>Tim A</b> dan <b>Tim B</b> yang ingin dipertandingkan.</div>
        <div class="step-box"><span class="step-num">2</span> Atur apakah pertandingan dimainkan di <b>lapangan netral</b> atau <b>kandang Tim A</b>.</div>
        <div class="step-box"><span class="step-num">3</span> Klik tombol <b>Prediksi Hasil</b>.</div>
        <div class="step-box"><span class="step-num">4</span> Lihat hasil prediksi, win rate kedua tim, dan probabilitas tiap kemungkinan hasil.</div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Model memprediksi berdasarkan **win rate historis** tiap tim di FIFA World Cup — "
        "bukan berdasarkan pemain, pelatih, atau kondisi tim saat ini."
    )

st.write("")

# ---------------------------------------------------------------------------
# Input — pemilihan tim
# ---------------------------------------------------------------------------
col1, col_vs, col2 = st.columns([5, 1, 5])

with col1:
    st.markdown("**Tim A (Home)**")
    tim_a = st.selectbox(
        "Tim A", teams,
        index=teams.index("Brazil") if "Brazil" in teams else 0,
        label_visibility="collapsed",
    )
    st.markdown(f"<h1 style='text-align:center'>{flag(tim_a)}</h1>", unsafe_allow_html=True)

with col_vs:
    st.markdown('<div class="vs-circle"><div class="vs-text">VS</div></div>', unsafe_allow_html=True)

with col2:
    st.markdown("**Tim B (Away)**")
    default_b = "Germany" if "Germany" in teams else teams[1]
    tim_b = st.selectbox(
        "Tim B", teams,
        index=teams.index(default_b),
        label_visibility="collapsed",
    )
    st.markdown(f"<h1 style='text-align:center'>{flag(tim_b)}</h1>", unsafe_allow_html=True)

st.write("")
lapangan_netral = st.toggle(
    "Lapangan netral (bukan kandang salah satu tim)",
    value=True,
    help="ON = pertandingan di negara ketiga. OFF = dianggap dimainkan di kandang Tim A.",
)

st.write("")
btn_col1, btn_col2, btn_col3 = st.columns([1, 1.2, 1])
with btn_col2:
    prediksi_clicked = st.button("Prediksi Hasil", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Hasil Prediksi
# ---------------------------------------------------------------------------
if prediksi_clicked:
    if tim_a == tim_b:
        st.warning("Pilih dua tim yang berbeda.")
    else:
        wr_a = win_rate_dict.get(tim_a, 0.33)
        wr_b = win_rate_dict.get(tim_b, 0.33)
        diff = wr_a - wr_b
        netral = 1 if lapangan_netral else 0

        fitur = pd.DataFrame([[wr_a, wr_b, diff, netral]], columns=features)
        hasil = model.predict(fitur)[0]
        proba = model.predict_proba(fitur)[0]
        kelas = model.classes_

        banner_class = {"W": "result-win", "D": "result-draw", "L": "result-lose"}[hasil]
        banner_text = {
            "W": f"{tim_a} DIPREDIKSI MENANG",
            "D": "PERTANDINGAN DIPREDIKSI SERI",
            "L": f"{tim_b} DIPREDIKSI MENANG",
        }[hasil]

        st.markdown(
            f'<div class="result-banner {banner_class}">{banner_text}</div>',
            unsafe_allow_html=True,
        )

        # Penjelasan alasan prediksi
        selisih_persen = abs(diff) * 100
        if hasil == "W":
            alasan = (
                f"<b>{tim_a}</b> punya win rate historis <b>{wr_a*100:.1f}%</b>, "
                f"lebih tinggi {selisih_persen:.1f} poin dibanding <b>{tim_b}</b> ({wr_b*100:.1f}%). "
                f"Tim dengan riwayat kemenangan lebih banyak di World Cup cenderung diprediksi menang lagi."
            )
        elif hasil == "L":
            alasan = (
                f"<b>{tim_b}</b> punya win rate historis <b>{wr_b*100:.1f}%</b>, "
                f"lebih tinggi {selisih_persen:.1f} poin dibanding <b>{tim_a}</b> ({wr_a*100:.1f}%). "
                f"Tim dengan riwayat kemenangan lebih banyak di World Cup cenderung diprediksi menang."
            )
        else:
            alasan = (
                f"Win rate kedua tim cukup berimbang — <b>{tim_a}</b> {wr_a*100:.1f}% vs "
                f"<b>{tim_b}</b> {wr_b*100:.1f}% (selisih hanya {selisih_persen:.1f} poin). "
                f"Karena kekuatan historisnya mirip, model memprediksi hasil seri."
            )
        if netral == 0:
            alasan += f" Faktor tambahan: <b>{tim_a}</b> bermain di kandang sendiri, memberi sedikit keuntungan ekstra."

        st.markdown(f'<div class="reason-box">{alasan}</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"Win Rate {tim_a}", f"{wr_a*100:.1f}%")
        with col2:
            st.metric(f"Win Rate {tim_b}", f"{wr_b*100:.1f}%")

        st.markdown("#### Probabilitas Setiap Hasil")

        urut = sorted(zip(kelas, proba), key=lambda x: -x[1])
        label_tampil = {
            "W": f"{tim_a} Menang",
            "D": "Seri",
            "L": f"{tim_b} Menang",
        }

        for k, p in urut:
            st.write(f"**{label_tampil[k]}** — {p*100:.1f}%")
            st.progress(float(p))

        fig, ax = plt.subplots(figsize=(7, 3.5))
        labels_chart = [label_tampil[k] for k, _ in urut]
        values_chart = [p * 100 for _, p in urut]
        colors = ["#2ecc71" if k == "W" else "#95a5a6" if k == "D" else "#e74c3c" for k, _ in urut]

        bars = ax.bar(labels_chart, values_chart, color=colors, edgecolor="white")
        for bar, val in zip(bars, values_chart):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:.1f}%", ha="center", fontweight="bold")
        ax.set_ylabel("Probabilitas (%)")
        ax.set_ylim(0, 100)
        ax.set_title(f"{tim_a} vs {tim_b}")
        fig.tight_layout()
        st.pyplot(fig)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer-note">
        Model dilatih dari data historis FIFA World Cup (1930–2022).
        Prediksi hanya berdasarkan win rate historis — tidak memperhitungkan pemain, pelatih, atau kondisi terkini.
    </div>
    """,
    unsafe_allow_html=True,
)

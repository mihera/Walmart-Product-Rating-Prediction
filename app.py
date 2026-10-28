
import warnings
warnings.filterwarnings("ignore")

from collections import UserList

# ---------- Compatibility shim ----------
# Preserve the compatibility fix for older scikit-learn pickles.
import sklearn.compose._column_transformer as _ct

if not hasattr(_ct, "_RemainderColsList"):
    class _RemainderColsList(UserList):
        def __init__(
            self,
            lst=None,
            *,
            future_dtype=None,
            warning_was_emitted=False,
            warning_enabled=True
        ):
            super().__init__(lst or [])
            self.future_dtype = future_dtype

    _ct._RemainderColsList = _RemainderColsList

# ----------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ---------- Page configuration ----------
st.set_page_config(
    page_title="Walmart Rating Predictor",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ---------- Custom styling ----------
st.markdown("""
<style>
/* Main background */
.stApp {
    background: linear-gradient(145deg, #f3f8ff 0%, #ffffff 50%, #fffaf0 100%);
}

.block-container {
    max-width: 1180px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Header */
.hero {
    background: linear-gradient(120deg, #003b75 0%, #0071ce 65%, #1689e8 100%);
    padding: 34px 36px;
    border-radius: 22px;
    color: white;
    box-shadow: 0 10px 30px rgba(0, 74, 145, 0.18);
    border-bottom: 6px solid #ffc220;
    margin-bottom: 25px;
}

.hero h1 {
    color: white !important;
    font-size: 2.35rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.8px;
}

.hero p {
    color: #e5f2ff;
    font-size: 1.05rem;
    margin: 10px 0 0 0;
}

.hero-tag {
    display: inline-block;
    background: #ffc220;
    color: #17324d;
    font-size: 0.8rem;
    font-weight: 800;
    padding: 6px 12px;
    border-radius: 20px;
    margin-bottom: 12px;
}

/* Section headings */
.section-heading {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #064b8e;
    font-size: 1.15rem;
    font-weight: 800;
    padding: 12px 0 10px 0;
    margin-top: 10px;
    border-bottom: 2px solid #e2edfa;
    margin-bottom: 14px;
}

.section-icon {
    background: #fff0bf;
    border-radius: 10px;
    padding: 7px 10px;
}

/* Input labels */
.stTextInput label,
.stNumberInput label,
.stSelectbox label,
.stSlider label,
.stCheckbox label {
    color: #29445e !important;
    font-weight: 600 !important;
}

/* Input fields */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    border-radius: 10px;
    border-color: #d4e3f4;
    background-color: #ffffff;
}

div[data-baseweb="input"] input {
    color: #183b5b;
}

/* Slider colors */
div[data-testid="stSlider"] [data-baseweb="slider"] {
    padding-top: 4px;
    padding-bottom: 8px;
}

div[data-testid="stSlider"] [role="slider"] {
    background-color: #0071ce;
    border-color: #0071ce;
}

/* Cards */
.panel {
    background: rgba(255, 255, 255, 0.92);
    padding: 22px 24px;
    border: 1px solid #dfebf8;
    border-radius: 18px;
    box-shadow: 0 4px 16px rgba(20, 70, 120, 0.045);
    margin-bottom: 18px;
}

.metric-card {
    background: white;
    border: 1px solid #e0eaf5;
    border-radius: 14px;
    padding: 17px;
    text-align: center;
    box-shadow: 0 3px 12px rgba(0, 60, 120, 0.04);
}

.metric-title {
    font-size: 0.84rem;
    color: #647b91;
    font-weight: 600;
    margin-bottom: 7px;
}

.metric-value {
    font-size: 1.55rem;
    color: #07549a;
    font-weight: 800;
}

.metric-note {
    font-size: 0.8rem;
    color: #8092a4;
    margin-top: 4px;
}

/* Prediction button */
div.stButton > button {
    width: 100%;
    min-height: 54px;
    background: linear-gradient(90deg, #0071ce, #004f9a);
    color: white;
    font-size: 1.08rem;
    font-weight: 800;
    border: none;
    border-radius: 13px;
    transition: all 0.2s ease;
    box-shadow: 0 5px 15px rgba(0, 90, 180, 0.2);
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #005eaf, #003b75);
    color: white;
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0, 90, 180, 0.25);
}

/* Prediction result */
.result-card {
    background: linear-gradient(145deg, #ffffff, #f0f7ff);
    border: 1px solid #d5e8fa;
    border-top: 6px solid #ffc220;
    border-radius: 22px;
    padding: 30px 20px;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0 8px 28px rgba(0, 70, 140, 0.09);
}

.result-label {
    color: #647b91;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 700;
}

.result-score {
    color: #0066b8;
    font-size: 4rem;
    font-weight: 900;
    line-height: 1.25;
    margin-top: 8px;
}

.result-score span {
    color: #91a2b3;
    font-size: 1.35rem;
    font-weight: 600;
}

.stars {
    position: relative;
    display: inline-block;
    font-size: 2.5rem;
    letter-spacing: 5px;
    line-height: 1.5;
    color: #dce3ea;
}

.stars .fill {
    position: absolute;
    left: 0;
    top: 0;
    overflow: hidden;
    white-space: nowrap;
    color: #ffbd00;
}

.result-badge {
    display: inline-block;
    margin-top: 8px;
    padding: 8px 22px;
    border-radius: 25px;
    font-size: 1rem;
    font-weight: 800;
    color: white;
}

.result-caption {
    margin-top: 15px;
    color: #6e8194;
    font-size: 0.92rem;
}

/* Small informational note */
.info-strip {
    background: #fff8e6;
    border-left: 4px solid #ffc220;
    border-radius: 8px;
    padding: 12px 15px;
    color: #604c1c;
    font-size: 0.9rem;
    margin: 12px 0 18px 0;
}

.footer {
    text-align: center;
    color: #8394a5;
    font-size: 0.83rem;
    padding: 25px 0 5px 0;
    border-top: 1px solid #e5edf5;
    margin-top: 35px;
}

/* Responsive sizing */
@media (max-width: 640px) {
    .hero {
        padding: 24px 20px;
    }
    .hero h1 {
        font-size: 1.7rem;
    }
    .panel {
        padding: 16px;
    }
    .result-score {
        font-size: 3rem;
    }
}
</style>
""", unsafe_allow_html=True)


# ---------- Load model ----------
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")


try:
    model = load_model()
except Exception as e:
    st.error(
        "Could not load best_model.pkl. Check that the file exists "
        "beside app.py and that the required Python/scikit-learn "
        "versions match the model's training environment."
    )
    st.code(f"{type(e).__name__}: {e}")
    st.stop()


# Read category choices from the fitted model.
try:
    prep = model.steps[0][1]
    enc = prep.named_transformers_["cat"]

    dept_opts, l1_opts, l2_opts, ful_opts = [
        list(c) for c in enc.categories_
    ]
except Exception as e:
    st.error("The loaded model does not have the expected preprocessing structure.")
    st.code(f"{type(e).__name__}: {e}")
    st.stop()


# ---------- Header ----------
st.markdown("""
<div class="hero">
    <div class="hero-tag">STATISTICAL INFERENCE PROJECT</div>
    <h1>⭐ Walmart Rating Predictor</h1>
    <p>Predict customer ratings using product pricing, reviews, categories
    and product listing details.</p>
</div>
""", unsafe_allow_html=True)


# ---------- Product information ----------
st.markdown("""
<div class="section-heading">
    <span class="section-icon">🛍️</span>
    Product Information
</div>
""", unsafe_allow_html=True)

name = st.text_input(
    "Product name",
    value="Cordless Stick Vacuum Cleaner for Carpet and Hard Floor",
    help="Enter the product name. Its length and words are used by the model."
)


# ---------- Pricing and reviews ----------
st.markdown("""
<div class="section-heading">
    <span class="section-icon">💰</span>
    Pricing & Customer Reviews
</div>
""", unsafe_allow_html=True)

price_col, original_col = st.columns(2)

with price_col:
    price = st.number_input(
        "Current price ($)",
        min_value=0.01,
        max_value=10000000.0,
        value=30.0,
        step=1.0,
        format="%.2f"
    )

    st.caption("Adjust the current selling price.")

with original_col:
    was_price = st.number_input(
        "Original price ($)",
        min_value=0.01,
        max_value=10000000.0,
        value=50.0,
        step=1.0,
        format="%.2f"
    )

    st.caption("Enter the price before the discount.")

# Sliders offer quick adjustment; number inputs preserve wider value ranges.
price_slider_col, original_slider_col = st.columns(2)

with price_slider_col:
    price_slider = st.slider(
        "Quick adjust current price ($)",
        min_value=0.01,
        max_value=500.0,
        value=float(min(price, 500.0)),
        step=0.5,
        key="price_quick_slider"
    )

    if price <= 500:
        price = float(price_slider)
    else:
        st.caption("Using the current price entered above.")

with original_slider_col:
    original_slider = st.slider(
        "Quick adjust original price ($)",
        min_value=0.01,
        max_value=500.0,
        value=float(min(was_price, 500.0)),
        step=0.5,
        key="original_quick_slider"
    )

    if was_price <= 500:
        was_price = float(original_slider)
    else:
        st.caption("Using the original price entered above.")

reviews_col, fulfillment_col = st.columns(2)

with reviews_col:
    num_reviews = st.number_input(
        "Number of reviews",
        min_value=0,
        max_value=100000000,
        value=100,
        step=10
    )

with fulfillment_col:
    fulfillment = st.selectbox(
        "Fulfillment method",
        ful_opts
    )

st.slider(
    "Quick adjust number of reviews",
    min_value=0,
    max_value=10000,
    value=int(min(num_reviews, 10000)),
    step=10,
    key="reviews_quick_slider"
)

# Use the slider for quick adjustments within its range.
if num_reviews <= 10000:
    num_reviews = st.session_state["reviews_quick_slider"]
else:
    st.caption("Using the number of reviews entered above.")


# ---------- Live price summary ----------
savings_preview = max(was_price - price, 0)
discount_preview = min(
    max((was_price - price) / was_price, 0),
    1
) * 100

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Current Price</div>
        <div class="metric-value">${price:,.2f}</div>
        <div class="metric-note">Selling price</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">You Save</div>
        <div class="metric-value">${savings_preview:,.2f}</div>
        <div class="metric-note">Compared with original price</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Discount</div>
        <div class="metric-value">{discount_preview:.1f}%</div>
        <div class="metric-note">Calculated automatically</div>
    </div>
    """, unsafe_allow_html=True)


# ---------- Category ----------
st.markdown("""
<div class="section-heading">
    <span class="section-icon">🗂️</span>
    Product Category
</div>
""", unsafe_allow_html=True)

cat_col1, cat_col2, cat_col3 = st.columns(3)

with cat_col1:
    dept = st.selectbox("Department", dept_opts)

with cat_col2:
    cat_l1 = st.selectbox("Category level 1", l1_opts)

with cat_col3:
    cat_l2 = st.selectbox("Category level 2", l2_opts)


# ---------- Badges and seller information ----------
st.markdown("""
<div class="section-heading">
    <span class="section-icon">🏷️</span>
    Badges & Seller Details
</div>
""", unsafe_allow_html=True)

badge_col1, badge_col2, badge_col3 = st.columns(3)

with badge_col1:
    bestseller = st.checkbox("🏆 Best seller")

with badge_col2:
    sponsored = st.checkbox("📣 Sponsored listing")

with badge_col3:
    walmart_seller = st.checkbox("🔵 Sold by Walmart")


with st.expander("⚙️ Advanced product details", expanded=False):
    st.caption("These details provide additional features to the prediction model.")

    advanced_col1, advanced_col2 = st.columns(2)

    with advanced_col1:
        n_flags = st.number_input(
            "Number of badge flags",
            min_value=0,
            max_value=10000,
            value=1 if bestseller else 0,
            step=1,
            key="n_flags"
        )

        n_tags = st.number_input(
            "Number of badge tags",
            min_value=0,
            max_value=10000,
            value=1,
            step=1
        )

    with advanced_col2:
        seller_freq = st.number_input(
            "Products from this seller in the deals list",
            min_value=1,
            max_value=10000000,
            value=1,
            step=1
        )

        desc_len = st.number_input(
            "Short description length (characters)",
            min_value=0,
            max_value=10000000,
            value=0,
            step=10
        )


# ---------- Prediction ----------
st.markdown("""
<div class="info-strip">
    <b>Ready to predict?</b> Review the product details above and click
    the button below to estimate its customer rating.
</div>
""", unsafe_allow_html=True)

if st.button("⭐ Predict Product Rating", use_container_width=True):

    # Keep the original feature calculations unchanged.
    savings = max(was_price - price, 0)

    discount_pct = min(
        max((was_price - price) / was_price, 0),
        1
    )

    row = pd.DataFrame([{
        "log_price": np.log1p(price),
        "log_reviews": np.log1p(num_reviews),
        "discount_pct": discount_pct,
        "savings": savings,
        "name_len": len(name),
        "name_words": len(name.split()),
        "desc_len": desc_len,
        "has_clearance": int("clearance" in name.lower()),
        "is_walmart_seller": int(walmart_seller),
        "seller_freq": seller_freq,
        "sponsored": int(sponsored),
        "bestseller": int(bestseller),
        "n_flags": n_flags,
        "n_tags": n_tags,
        "dept": dept,
        "cat_l1": cat_l1,
        "cat_l2": cat_l2,
        "fulfillment": fulfillment,
        "name": name,
    }])

    try:
        pred = float(model.predict(row)[0])
        pred = min(max(pred, 1.0), 5.0)

    except Exception as e:
        st.error("Prediction failed. Check the model and input feature compatibility.")
        st.code(f"{type(e).__name__}: {e}")
        st.stop()

    # Keep the original rating categories.
    if pred >= 4.5:
        label, color = "Excellent", "#239653"
        message = "This product has a very high predicted customer rating."
    elif pred >= 4.0:
        label, color = "Good", "#63a82e"
        message = "This product has a good predicted customer rating."
    elif pred >= 3.5:
        label, color = "Average", "#e7a000"
        message = "This product has an average predicted customer rating."
    else:
        label, color = "Below average", "#df5948"
        message = "This product has a lower predicted customer rating."

    # Visual rating out of five.
    pct = pred / 5 * 100

    st.markdown("""
    <div class="section-heading">
        <span class="section-icon">📊</span>
        Prediction Result
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Predicted Customer Rating</div>
        <div class="result-score">{pred:.2f}<span> / 5.00</span></div>

        <div class="stars">
            ★★★★★
            <div class="fill" style="width:{pct}%">★★★★★</div>
        </div>

        <br>

        <div class="result-badge" style="background:{color}">
            {label}
        </div>

        <div class="result-caption">{message}</div>
    </div>
    """, unsafe_allow_html=True)

    # Additional summary
    r1, r2, r3 = st.columns(3)

    with r1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Product</div>
            <div style="font-size:1rem;font-weight:700;color:#07549a;">
                {name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Reviews Entered</div>
            <div class="metric-value">{num_reviews:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with r3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Discount Entered</div>
            <div class="metric-value">{discount_pct * 100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)


# ---------- Footer ----------
st.markdown("""
<div class="footer">
    ⭐ Walmart Product Rating Predictor
    <br>
    Statistical Inference Mini Project
</div>
""", unsafe_allow_html=True)

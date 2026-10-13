
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime

st.set_page_config(
    page_title="CorroGuard AI-X",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ CorroGuard AI-X")
st.subheader("AI-Assisted Corrosion Detection & Sustainable Protection")

st.write(
    "Analyze metal samples, compare protective coatings, "
    "and track estimated rust-colored areas."
)

st.warning(
    "Research prototype only. Image colors provide preliminary estimates, "
    "not validated corrosion measurements."
)

if "history" not in st.session_state:
    st.session_state.history = []

def analyze_image(file):
    image = Image.open(file).convert("RGB")
    rgb = np.array(image)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    lower1 = np.array([0, 40, 20])
    upper1 = np.array([25, 255, 255])
    lower2 = np.array([160, 40, 20])
    upper2 = np.array([180, 255, 255])

    mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower1, upper1),
        cv2.inRange(hsv, lower2, upper2)
    )

    percentage = np.count_nonzero(mask) / mask.size * 100

    overlay = rgb.copy()
    overlay[mask > 0] = [255, 40, 40]

    return rgb, overlay, percentage

analysis_tab, compare_tab, history_tab = st.tabs([
    "Image Analysis",
    "Sample Comparison",
    "History & Trends"
])

with analysis_tab:
    uploaded = st.file_uploader(
        "Upload a metal sample",
        type=["jpg", "jpeg", "png"],
        key="single"
    )

    sample_name = st.text_input(
        "Sample name",
        value="Uncoated Steel"
    )

    if uploaded:
        original, overlay, percentage = analyze_image(uploaded)

        col1, col2 = st.columns(2)

        with col1:
            st.image(original, caption="Original Sample",
                     use_container_width=True)

        with col2:
            st.image(overlay, caption="Estimated Rust Regions",
                     use_container_width=True)

        st.metric("Rust-Colored Area", f"{percentage:.2f}%")

        if st.button("Save Analysis"):
            st.session_state.history.append({
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Sample": sample_name,
                "Rust Coverage (%)": round(percentage, 2)
            })
            st.success("Analysis saved.")

with compare_tab:
    col1, col2 = st.columns(2)

    with col1:
        file_a = st.file_uploader(
            "Upload Sample A",
            type=["jpg", "jpeg", "png"],
            key="a"
        )
        name_a = st.text_input("Sample A", "Uncoated Steel")

    with col2:
        file_b = st.file_uploader(
            "Upload Sample B",
            type=["jpg", "jpeg", "png"],
            key="b"
        )
        name_b = st.text_input("Sample B", "Natural Coating")

    if st.button("Compare Samples"):
        if file_a is None or file_b is None:
            st.warning("Upload both sample images first.")
        else:
            rgb_a, overlay_a, rust_a = analyze_image(file_a)
            rgb_b, overlay_b, rust_b = analyze_image(file_b)

            c1, c2 = st.columns(2)

            with c1:
                st.image(overlay_a, caption=name_a,
                         use_container_width=True)
                st.metric("Rust-Colored Area", f"{rust_a:.2f}%")

            with c2:
                st.image(overlay_b, caption=name_b,
                         use_container_width=True)
                st.metric("Rust-Colored Area", f"{rust_b:.2f}%")

            difference = rust_a - rust_b

            if difference > 0:
                st.write(
                    f"Sample B has {difference:.2f} percentage points "
                    "less rust-colored area than Sample A."
                )
            elif difference < 0:
                st.write(
                    f"Sample B has {abs(difference):.2f} percentage points "
                    "more rust-colored area than Sample A."
                )
            else:
                st.write("Both samples have equal estimated coverage.")

            st.caption(
                "Use consistent lighting and repeated measurements. "
                "Color differences alone do not prove coating effectiveness."
            )

with history_tab:
    st.header("History & Trends")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(9, 4))

        for sample, group in df.groupby("Sample"):
            ax.plot(
                range(1, len(group) + 1),
                group["Rust Coverage (%)"],
                marker="o",
                label=sample
            )

        ax.set_xlabel("Measurement Number")
        ax.set_ylabel("Rust-Colored Area (%)")
        ax.set_title("Saved Sample Measurements")
        ax.grid(True, alpha=0.3)
        ax.legend()

        st.pyplot(fig)
        plt.close(fig)

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            file_name="corroguard_history.csv",
            mime="text/csv"
        )

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No saved measurements yet.")

st.divider()
st.caption("CorroGuard AI-X | Preliminary research prototype")

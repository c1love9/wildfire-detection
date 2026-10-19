import streamlit as st
from pathlib import Path

st.markdown(
        """
        <style>
        .container {
            max-width: 800px;
        }
        .title {
            text-align: center;
            font-size: 35px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .description {
            margin-bottom: 30px;
        }
        .instructions {
            margin-bottom: 20px;
            padding: 10px;
            background-color: #f5f5f5;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

st.markdown("<div class='title'>Demo Images</div>", unsafe_allow_html=True)

images = sorted((Path(__file__).resolve().parents[1] / 'demo-images').glob('*.jpeg'))

for image in images:
    st.image(str(image), width='stretch')


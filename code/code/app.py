import streamlit as st
from config import PROJECT_TILE, PROJECT_SUBTITLE, logo_html

import pandas as pd
import pytesseract
import glob
import cv2
from utils import encode_and_prepare_image, process_text, get_regex
from image_processing import prepare_composition, compose_functions, resize_img
from texts import (
    PROCESS_DEFINITIONS,
    IMAGE_SELECTION,
    IMAGE_LOAD,
    APPLY_ORC,
    RESIZE_OPTIONS,
    BLUR_1,
    THRESHOLDING_TEXT,
    HISTOGRAM_TEXT,
    BLUR_2,
    TEXT_PROCESSING_TEXT,
    WORDS_TEXT,
)

words_water = [
    "bicarbonato",
    "sodio",
    "calcio",
    "magnesio",
    "cloreto",
    "sulfato",
    "potassio",
    "nitrato",
    "fluoreto",
    "estroncio",
    "bario",
]
words = [
    "carboidratos",
    "proteinas",
    "gorduras totais",
    "gorduras saturadas",
    "gorduras trans",
    "fibra alimentar",
    "sodio",
    "valor energetico",
]
languages = {"engUS": "English", "ptBR": "Português"}
words_dict = {"Water": words_water, "Food": words}


st.markdown(
    logo_html, unsafe_allow_html=True,
)
st.title(PROJECT_TILE)

lang = st.sidebar.selectbox(
    "", ["engUS", "ptBR"], format_func=lambda x: languages.get(x),
)
st.header(PROJECT_SUBTITLE[lang])
st.sidebar.title(PROCESS_DEFINITIONS[lang])
filenames = list(sorted(glob.glob("./images/good_examples/*")))
file_to_show = st.sidebar.selectbox(IMAGE_SELECTION[lang], filenames)

load_image = st.sidebar.checkbox(IMAGE_LOAD[lang], value=False)
st.sidebar.subheader(APPLY_ORC[lang])
ch_o = st.sidebar.checkbox("Original", value=False)


st.sidebar.subheader("Process image with...")
resize_options = st.sidebar.multiselect(
    RESIZE_OPTIONS[lang],
    options=[0.1, 0.3, 0.5, 0.7, 0.9, 1],
    default=[0.1, 0.3, 0.5, 0.7, 0.9, 1],
)
blurry = st.sidebar.checkbox(BLUR_1[lang], value=False)
blurry_value = None
bilateral_filtering_sigma = None
if blurry:
    blurry_value = st.sidebar.slider(
        "Kernel", min_value=1, value=3, max_value=21, step=2
    )
thresholding = st.sidebar.checkbox(THRESHOLDING_TEXT[lang], value=False)
contrast = st.sidebar.checkbox(HISTOGRAM_TEXT[lang], value=False)
filtering = st.sidebar.checkbox(BLUR_2[lang], value=False)
if filtering:
    bilateral_filtering_sigma = st.sidebar.slider(
        "Sigma", min_value=5, value=50, max_value=100, step=5
    )

functions_to_compose = prepare_composition(
    blurry, blurry_value, thresholding, contrast, filtering, bilateral_filtering_sigma
)

st.sidebar.write("-------")
st.sidebar.subheader(TEXT_PROCESSING_TEXT[lang])
words_database = st.sidebar.selectbox("Select dataset", ["Food", "Water"])
words_to_consider = st.sidebar.multiselect(
    WORDS_TEXT[lang],
    options=words_dict[words_database],
    default=words_dict[words_database],
)


if not load_image:
    st.write("Waiting for image...")
else:
    img = cv2.imread(file_to_show, cv2.IMREAD_GRAYSCALE)
    st.markdown(
        encode_and_prepare_image("Original", img), unsafe_allow_html=True,
    )
    result_dict = {}
    result_cols = [
        "scale_factor",
        "recovered_text",
        "terms",
        "terms_to_search",
        "acc_terms",
        "acc_tts",
    ]
    result_df = pd.DataFrame(columns=result_cols)
    if ch_o:
        for scale_factor in resize_options:
            st.text("-----------------")
            resized = resize_img(img, scale_factor)
            composed_image = compose_functions(resized, functions_to_compose)
            st.markdown(
                encode_and_prepare_image(str(scale_factor), composed_image),
                unsafe_allow_html=True,
            )

            txt = pytesseract.image_to_string(composed_image, lang="por")
            try:
                terms, terms_to_search, distance_acc, df2 = process_text(
                    txt, words_to_consider
                )
                result = get_regex(terms_to_search, txt)
                result_dict = {}
                result_dict = dict(
                    zip(
                        result_cols,
                        [
                            str(scale_factor),
                            txt,
                            terms,
                            result,
                            len(terms),
                            len(result),
                        ],
                    )
                )
                result_df = result_df.append(result_dict, ignore_index=True)
            except:
                print(f"Unable to process image to {scale_factor}!")

        st.write("-------------------")
        st.header("Results:")
        result_df = result_df.sort_values(by=["acc_terms", "acc_tts"], ascending=False)
        terms_set = list(
            set([item for sublist in result_df["terms"] for item in sublist])
        )
        st.write(result_df)
        best_result = result_df.iloc[0]
        st.subheader("Accuracy")
        st.write(f"Combined result acc = {len(terms_set)/len(words_to_consider)}")
        st.write(
            f'Best result for scale_factor = {best_result["scale_factor"]}, with acc = {best_result["acc_terms"]/len(words_to_consider)}!'
        )
        st.subheader("Terms recovered (combined)")
        st.write(terms_set)
        st.subheader("Terms recovered (best result)")
        st.write(best_result["terms"])
        st.write("-------------------")
        st.subheader("Recovered text...")
        st.write(best_result["recovered_text"])

import cv2
import jiwer
import Levenshtein as lev
import pandas as pd
import base64


def get_regex(terms, text):
    ttt = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemoveEmptyStrings(),
        ]
    )
    transformed_text = ttt(text).split("\n")
    import re

    result = {}
    rerere = re.compile(r"\d+(,\d+)*")
    for line in transformed_text:
        for term in terms:
            if term in line:
                # print(f"Found {term} in {line}!")
                possible_number = rerere.search(line)
                if possible_number:
                    result[term] = possible_number.group(0)

    return result


def process_text(text, words):
    df = pd.DataFrame(columns=["word", *words])
    default_transformation = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemoveWhiteSpace(replace_by_space=True),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.SentencesToListOfWords(word_delimiter=" "),
            jiwer.RemoveEmptyStrings(),
        ]
    )
    transformed_text = default_transformation(text)
    dd = {}
    for word in words:
        gram_len = len(word.split(" "))
        for i in range(0, len(transformed_text) - gram_len + 1):
            instance = " ".join(transformed_text[i : i + gram_len])
            distance = lev.distance(instance, word)
            aux_dd = (word, distance)
            if instance not in dd:
                dd[instance] = [aux_dd]
            else:
                dd[instance].append(aux_dd)

    for key, values in dd.items():
        bla = {}
        bla["word"] = key
        for w, d in values:
            bla[w] = d
        df = df.append(bla, ignore_index=True)

    df = df.fillna(666)
    df[words] = df[words].apply(pd.to_numeric)
    terms_df = df.iloc[df[words].idxmin().values]
    col_idxs = terms_df[words].idxmin(axis=1).values
    terms = []
    terms_to_search = []
    distance_acc = 0
    for term, probable_word in dict(zip(terms_df.word, col_idxs)).items():
        value = terms_df.query("word == @term")[probable_word].values[0]
        if value <= 2:
            # print(f"distance between {term} and {probable_word} == {value}")
            terms.append(probable_word)
            terms_to_search.append(term)
            distance_acc += value
        # else:
        # print(f"for {term}={probable_word}, not enough={value}")
    return (terms, terms_to_search, len(terms), df)


def html_details_session(title, image_str):
    return f"""
        <details>
            <summary>{title}</summary>
            <img src='data:image/png;base64,{image_str}' width='500' height='600'>.
        </details>
        """


def encode_and_prepare_image(title, image):
    _, im_png = cv2.imencode(".png", image)
    byte_img = base64.b64encode(im_png.tobytes()).decode()
    return html_details_session(title, byte_img)

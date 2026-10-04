import streamlit as st, json, io, re
import fitz
from PIL import Image
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="Document Investigator", layout="wide")
st.title("🔎 Intelligent Document Investigator")


def split(text, name, loc):
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 30]
    return [{"file": name, "loc": loc, "text": p[:1200]} for p in paras]

def ocr(img):
    return pytesseract.image_to_string(img)

def extract(f):
    name, data, chunks = f.name, f.read(), []
    low = name.lower()
    if low.endswith(".pdf"):
        doc = fitz.open(stream=data, filetype="pdf")
        for i, page in enumerate(doc):
            t = page.get_text()
            if len(t.strip()) < 20:
                pix = page.get_pixmap(dpi=150)
                t = ocr(Image.open(io.BytesIO(pix.tobytes("png"))))
            chunks += split(t, name, f"page {i+1}")
    elif low.endswith((".png", ".jpg", ".jpeg")):
        chunks += split(ocr(Image.open(io.BytesIO(data))), name, "image")
    else:
        chunks += split(data.decode("utf-8", errors="ignore"), name, "text")
    return chunks

files = st.file_uploader("Upload PDFs, images or text files", accept_multiple_files=True,
                         type=["pdf", "png", "jpg", "jpeg", "txt"])

if files:
    if "chunks" not in st.session_state or st.session_state.get("names") != [f.name for f in files]:
        with st.spinner("Extracting and indexing..."):
            chunks = [c for f in files for c in extract(f)]
            vec = TfidfVectorizer(stop_words="english").fit([c["text"] for c in chunks])
            st.session_state.update(chunks=chunks, vec=vec,
                                    mat=vec.transform([c["text"] for c in chunks]),
                                    names=[f.name for f in files])
    st.success(f"Indexed {len(st.session_state.chunks)} passages from {len(files)} documents")

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

q = st.text_input("Ask a question about your documents")

def fields(text):
    return {m.group(1).strip().lower(): m.group(2).strip()
            for m in re.finditer(r"([A-Za-z ]{3,25}):\s*([^.:\n]+)", text)}

def missing_terms(q, text):
    words = [w for w in re.findall(r"[a-z0-9]+", q.lower())
             if w not in ENGLISH_STOP_WORDS and len(w) > 2]
    t = text.lower()
    return [w for w in words if w[:5] not in t]

if q and files:
    S = st.session_state
    sims = cosine_similarity(S.vec.transform([q]), S.mat)[0]
    top = [i for i in sims.argsort()[::-1][:5] if sims[i] > 0]
    best = sims[top[0]] if top else 0
    conf = "HIGH" if best > 0.35 else "MEDIUM" if best > 0.15 else "LOW"

    st.subheader("Answer")
    miss = []
    if not top or best < 0.05:
        st.warning("No relevant information found in the uploaded documents.")
    else:
        c0 = S.chunks[top[0]]
        miss = missing_terms(q, c0["text"])
        if miss:
            conf = "LOW"
            st.warning("I couldn't find information about: " + ", ".join(miss) + ". Showing the closest passage instead.")
        st.write(c0["text"])
        st.caption(f"Best source: {c0['file']} — {c0['loc']}")
        st.metric("Confidence", conf)
        if conf == "LOW":
            st.warning("Low confidence: the match is weak. Please verify in the sources below.")

        vals = {}
        for i in top:
            c = S.chunks[i]
            for k, v in fields(c["text"]).items():
                vals.setdefault(k, {}).setdefault(v, set()).add(c["file"])
        conflicts = [(k, d) for k, d in vals.items()
                     if len(d) > 1 and len({f for fs in d.values() for f in fs}) > 1]

        if conflicts and not miss:
            st.error("⚠️ Conflicting information detected across documents")
            for k, d in conflicts:
                st.write(f"**{k.title()}**: " + " vs ".join(f"'{v}' ({', '.join(sorted(fs))})" for v, fs in d.items()))
            st.info("The documents disagree, so no single answer is certain.")

        st.subheader("Supporting sources")
        for i in top:
            c = S.chunks[i]
            with st.expander(f"{c['file']} — {c['loc']}  (match {sims[i]:.2f})"):
                st.write(c["text"])
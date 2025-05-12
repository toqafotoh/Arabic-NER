import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pandas as pd

# Initialize session state
if "selected_sentence" not in st.session_state:
    st.session_state.selected_sentence = None
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# Load the custom NER model (CPU) from local files
model_path = r"D:\Level 4 2nd term\FinalModel\kaggle\working\FinalModel"
tokenizer_path = model_path
config_path = model_path

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

# Create NER pipeline using the custom model and tokenizer
ner_model = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=False, device=-1)

# Example sentences from the model's training data
example_sentences = [
    "سافر أحمد إلى القاهرة لحضور مؤتمر التكنولوجيا.",
    "مُنح كتاب 'الخيميائي' جائزة أفضل رواية.",
    "ستقام المباراة النهائية في ملعب الملك فهد الدولي.",
    "تعمل ليلى في شركة مايكروسوفت منذ خمس سنوات.",
    "ولد العالم إسحاق نيوتن في إنجلترا.",
    "أقيمت فعاليات معرض الكتاب الدولي في الرياض.",
    "يبدأ العام الدراسي الجديد في سبتمبر.",
    "يتحدث سامي اللغة الفرنسية بطلاقة.",
    "زار السائحون مدينة البتراء الأثرية في الأردن.",
    "يقدم المستشفى الوطني خدمات طبية عالية الجودة.",
    "تم عرض الفيلم الوثائقي الجديد على قناة الجزيرة.",
    "افتتحت شركة سامسونج فرعاً جديداً في دبي.",
    "سيتم تسليم الجوائز يوم الخميس القادم.",
    "أحب قراءة كتاب 'مئة عام من العزلة'.",
    "تعيّن الدكتور يوسف عميداً لكلية الهندسة.",
    "اللغة الإسبانية منتشرة في قارة أمريكا الجنوبية.",
    "زار الفريق الرئاسي قصر قرطاج الرئاسي في تونس.",
    "يبدأ مهرجان كان السينمائي في مايو من كل عام.",
    "تعد جبال الهيمالايا من أعلى سلاسل الجبال في العالم.",
    "تم إعلان الفائز بجائزة نوبل للسلام هذا الأسبوع."
]

# Dropdown for example sentences
st.sidebar.title("Select an Example Sentence")
selected_sentence = st.sidebar.selectbox(
    "Choose a sentence:", [""] + example_sentences
)

# Abbreviations and their full forms
abbreviations = {
    "ANG": "Anger",
    "DUC": "Document or Discussion",
    "EVE": "Event",
    "FAC": "Facility",
    "GPE": "Geopolitical Entity",
    "INFORMAL": "Informal Expression",
    "LOC": "Location",
    "MISC": "Miscellaneous",
    "ORG": "Organization",
    "PER": "Person",
    "TIMEX": "Time Expression",
    "TTL": "Title",
    "WOA": "Work of Art",
    "O": "Outside any entity"
}

# Sidebar: Abbreviations Table
st.sidebar.title("Abbreviations & Full Forms")
abbs_df = pd.DataFrame(list(abbreviations.items()), columns=["Abbreviation", "Full Form"])
st.sidebar.write(abbs_df)

# Update session state when a sentence is selected
if selected_sentence:
    st.session_state.selected_sentence = selected_sentence
    st.session_state.input_text = selected_sentence  # Pre-fill the input box
else:
    st.session_state.selected_sentence = None
    st.session_state.input_text = ""  # Clear the input box if no sentence is selected

# Main section for text input
st.title("NER (Named Entity Recognition) Tool")

# Input box for text (either user input or pre-filled with selected sentence)
input_text = st.text_area(
    "Enter arabic text for NER",
    value=st.session_state.input_text,
    height=150,
    key="input_box"
)

# دالة للتحقق من أن النص كله عربي فقط
def is_arabic(text):
    return all(
        char == ' ' or
        '\u0600' <= char <= '\u06FF' or
        '\u0750' <= char <= '\u077F' or
        '\u08A0' <= char <= '\u08FF' or
        '\uFB50' <= char <= '\uFDFF' or
        '\uFE70' <= char <= '\uFEFF' or
        not char.isalpha()
        for char in text
    )
# Run NER when the button is pressed
if st.button("Run NER"):
    if not input_text.strip():  # If no input is provided
        st.warning("Please enter arabic text or select an example sentence.")
    elif not is_arabic(input_text) and not selected_sentence:
        st.error("Please enter arabic text or select an example sentence!")
    else:
        ner_results = ner_model(input_text)

        # Build char-index → label map
        span_to_ent = {}
        for ent in ner_results:
            for i in range(ent["start"], ent["end"]):
                span_to_ent[i] = ent["entity"]  # Use 'entity' instead of 'entity_group'

        # Tokenize and assign labels
        tokens, labels = [], []
        offset = 0
        for word in input_text.split():
            start = input_text.find(word, offset)
            end = start + len(word)
            offset = end

            char_labels = {span_to_ent.get(i) for i in range(start, end)}
            char_labels.discard(None)

            if len(char_labels) == 1:
                label = char_labels.pop()  # Directly use the label without conversion to int
            elif len(char_labels) > 1:
                label = next(iter(char_labels))
            else:
                label = "O"  # Outside any entity

            tokens.append(word)
            labels.append(label)

        # Create DataFrame and display as table
        df = pd.DataFrame({"Token": tokens, "Entity": labels})
        st.subheader("Token‑level Entities")
        st.table(df)

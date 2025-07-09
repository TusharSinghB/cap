import re
from collections import Counter
from textblob import TextBlob
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk

# Download required resources once
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

from nltk.corpus import stopwords

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))


# ----------------------------- #
#       TOP KEYWORDS            #
# ----------------------------- #

def get_top_keywords(text, n=5):
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    filtered_words = [word for word in words if word not in stop_words]
    return Counter(filtered_words).most_common(n)


# ----------------------------- #
#    NAMED ENTITY EXTRACTION    #
# ----------------------------- #

def get_named_entities(text, entity_types=["PERSON", "ORG", "GPE"]):
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        if ent.label_ in entity_types:
            entities.setdefault(ent.label_, []).append(ent.text)

    # Count most common per entity type
    entity_counts = {
        label: Counter(vals).most_common(5) for label, vals in entities.items()
    }
    return entity_counts


# ----------------------------- #
#        HEADING DETECTION      #
# ----------------------------- #

def detect_headings(text):
    lines = text.split('\n')
    headings = []
    for line in lines:
        line = line.strip()
        if (
            re.match(r'^\d+[\.\)]\s+\w+', line)  # numbered heading: 1. Introduction
            or re.match(r'^[A-Z][A-Za-z\s]+:$', line)  # ends with colon: Objective:
            or line.isupper() and len(line.split()) < 10  # all caps line with <10 words
        ):
            headings.append(line)
    return headings


# ----------------------------- #
#     SENTENCE-WISE SENTIMENT   #
# ----------------------------- #

def sentence_sentiments(text):
    blob = TextBlob(text)
    return [(str(sentence), round(sentence.sentiment.polarity, 3)) for sentence in blob.sentences]


# ----------------------------- #
#    TABLE-LIKE TEXT BLOCKS     #
# ----------------------------- #

def detect_table_like_sections(text):
    lines = text.split('\n')
    tables = []
    current_table = []

    for line in lines:
        if len(re.findall(r'\s{2,}|\t+', line)) >= 1:
            current_table.append(line)
        else:
            if current_table:
                tables.append('\n'.join(current_table))
                current_table = []

    if current_table:
        tables.append('\n'.join(current_table))

    return tables[:3]  # return top 3


# ----------------------------- #
#        WORD CLOUD             #
# ----------------------------- #

def generate_wordcloud(text):
    wc = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout()
    return plt

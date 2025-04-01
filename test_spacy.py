# python -m spacy download el_core_news_sm
import spacy
nlp = spacy.load("el_core_news_sm")
import el_core_news_sm
nlp = el_core_news_sm.load()
doc = nlp("Αυτή είναι μια πρόταση.")
print([(w.text, w.pos_) for w in doc])
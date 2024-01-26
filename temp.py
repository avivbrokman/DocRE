import spacy

nlp = spacy.load('en_core_sci_lg')


#%%
text = "Your text goes here. It can be multiple sentences."

doc = nlp(text)

#%%
for sentence in doc.sents:
    print("Sentence:", sentence.text)
    for token in sentence:
        print("  Token:", token.text)

#%%
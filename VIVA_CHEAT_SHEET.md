# Viva Cheat Sheet: Resume–JD Matcher

## 1. Why preprocessing?
Raw text is noisy. Without cleaning, `"Python"`, `"python,"` and `"PYTHON"` are
three different tokens, and the most frequent words are filler like
*the, and, of*. Each step removes one kind of noise:
- **Lowercasing**: so case differences don't create separate tokens
- **Removing punctuation/numbers**: they rarely say anything about skills
- **Tokenizing**: splits text into units we can count
- **Removing stopwords**: drops common words with little meaning. I also added
  domain stopwords like *experience* and *requirement*, which appear in every JD.
- **Lemmatizing**: maps word forms to one base form

*Result:* the vectors represent **content words**, so the similarity score
reflects real overlap and not shared grammar. (The "How it works" panel shows
the text after each stage.)

## 2. Why lemmatization over stemming?
- **Stemming** cuts off word endings using rules. It often produces non-words:
  *studies → studi*, *analysis → analysi*.
- **Lemmatization** looks words up in a dictionary (WordNet) and returns the real
  base form: *studies → study*, *models → model*, *analyses → analysis*.

For keyword matching this matters because the missing/matched keywords are
**shown to the user**, so they need to be readable words. Lemmatization is also
less likely to merge unrelated words. Stemming would turn *university* and
*universe* into the same stem, *univers*.
*Trade-off:* lemmatization is slower and works best with POS tags. I lemmatize
everything as a noun, so *built* stays *built* and doesn't become *build* (see
limitations).

## 3. Why TF-IDF over Bag-of-Words?
**Bag-of-Words** only counts words, so frequent words dominate even when they
are generic. **TF-IDF** = Term Frequency × Inverse Document Frequency:
- **TF**: how often the word appears in *this* document (it matters here)
- **IDF**: `log(N / df)`, which is lower for words that appear in *many* documents
  (they don't help tell documents apart)

So distinctive terms like *pytorch* or *docker* get more weight than common
ones. In this project I also use TF-IDF across the **lines of the JD** to pick its
top keywords. Words that appear in only a few specific lines get higher weight.

## 4. What does cosine similarity measure?
The **cosine of the angle** between two vectors:

```
cos(θ) = (A · B) / (‖A‖ × ‖B‖)
```

- 1 → the vectors point in the same direction (same word distribution)
- 0 → no words in common (TF-IDF values are never negative, so the minimum is 0)

It measures **orientation, not length**. A 2-page resume and a half-page JD can
still score high if they use the same terms in similar proportions. Euclidean
distance would penalise the length difference. ×100 gives a percentage.

## 5. Limitations
1. **No understanding of synonyms or meaning.** *ML* ≠ *machine learning*,
   *JS* ≠ *JavaScript*. TF-IDF only matches identical words. That is why the
   bonus semantic score (sentence embeddings) is usually higher.
2. **Word order and context are lost.** In Bag-of-Words/TF-IDF, "no Python
   experience" still matches "Python".
3. **Preprocessing side-effects.** Removing punctuation breaks *C++*, *C#*,
   *.NET* and *TF-IDF*. Noun-only lemmatization leaves *built/build* and
   *analysed/analyse* as different words.
4. **The scores are relative, not absolute.** A real resume and JD usually score
   only about 20–40% with TF-IDF. A "good" threshold has to be chosen by
   experience. Short texts also give unstable scores.
5. **No idea of importance or level.** "5 years of Java" and "Java basics" look
   the same. A must-have skill counts the same as a nice-to-have, and there is
   no experience/education weighting. PDF text extraction can also be messy for
   multi-column resumes.

**Possible improvements:** POS-aware lemmatization, n-grams for phrases like
"machine learning", a skills dictionary or NER to extract only real skills,
embeddings (already in the bonus), and weighting must-have vs. nice-to-have
sections.

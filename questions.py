import nltk
import sys
import string
import os
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 3


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus-en")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    while True:
        query = set(tokenize(input("Query: ")))
        if query == "exit":
            return
        # Determine top file matches according to TF-IDF
        filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

        # Extract sentences from top files
        sentences = dict()
        for filename in filenames:
            for passage in files[filename].split("\n"):
                for sentence in nltk.sent_tokenize(passage):
                    tokens = tokenize(sentence)
                    if tokens:
                        sentences[sentence] = tokens

        # Compute IDF values across sentences
        idfs = compute_idfs(sentences)

        # Determine top sentence matches
        matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
        for match in matches:
            print(match)


def load_files(directory):
    res = {}
    for (dirpath, _, filenames) in os.walk(directory):
        for filename in filenames:
            with open(os.path.join(dirpath, filename)) as f:
                res[filename] = f.read()
    return res


def tokenize(document):
    stopwords = set(nltk.corpus.stopwords.words("english"))

    tokens = nltk.word_tokenize(document.lower())
    tokens = [t for t in tokens if not (t in string.punctuation or t in stopwords)]
    return tokens


def compute_idfs(documents):
    dn = len(documents)
    res = {}
    for doc in documents.values():
        appeared = set()
        for token in doc:
            if token in appeared:
                continue
            res[token] = res.get(token, 0) + 1
            appeared.add(token)

    for token in res:
        res[token] = math.log(dn / res[token])
    return res


def top_files(query, files, idfs, n):
    scores = []
    for f in files:
        score = 0
        for word in query:
            tf = files[f].count(word)
            score += tf * idfs.get(word, 0)
        scores.append((f, score))

    scores.sort(key=lambda item: item[1], reverse=True)
    return [item[0] for item in scores][:n]


def top_sentences(query, sentences, idfs, n):
    scores = []
    for s in sentences:
        score = 0
        density = 0
        for word in query:
            if word not in sentences[s]:
                continue
            density += sentences[s].count(word)
            score += idfs.get(word, 0)

        density /= len(sentences[s])
        scores.append((s, score, density))

    scores.sort(key=lambda item: (item[1], item[2]), reverse=True)
    return [item[0] for item in scores][:n]


def dld(s1, s2):
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in range(-1, lenstr1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, lenstr2 + 1):
        d[(-1, j)] = j + 1

    for i in range(lenstr1):
        for j in range(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,  # deletion
                d[(i, j - 1)] + 1,  # insertion
                d[(i - 1, j - 1)] + cost,  # substitution
            )
            if i and j and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[i - 2, j - 2] + 1)  # transposition

    return d[lenstr1 - 1, lenstr2 - 1]


if __name__ == "__main__":
    main()

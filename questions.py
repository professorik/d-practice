import nltk
import sys
import string
import os
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1
DELTA = 2
LANGUAGE = "english"


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python questions.py corpus-uk-lite uk")
    if len(sys.argv) == 3:
        global LANGUAGE
        LANGUAGE = {"en": "english", "uk": "ukrainian"}.get(sys.argv[2], LANGUAGE)

    file_words = tokenize_dir(sys.argv[1])
    file_idfs = compute_idfs(file_words)

    while True:
        instr = input("Query: ")
        if instr == "exit":
            return
        query = set(tokenize(instr))
        filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

        sentences = dict()
        for filename in filenames:
            for passage in get_passages(os.path.join(sys.argv[1], filename)):
                for sentence in nltk.sent_tokenize(passage):
                    tokens = tokenize(sentence)
                    if tokens:
                        sentences[sentence] = tokens

        idfs = compute_idfs(sentences)
        matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
        for match in matches:
            print(match)


def tokenize(document, stopwords=None):
    if stopwords is None:
        stopwords = set(nltk.corpus.stopwords.words(LANGUAGE))

    tokens = nltk.word_tokenize(document.lower())
    tokens = [t for t in tokens if not (t in string.punctuation or t in stopwords)]
    return tokens


def tokenize_dir(directory):
    stopwords = set(nltk.corpus.stopwords.words(LANGUAGE))
    res = {}
    for (dirpath, _, filenames) in os.walk(directory):
        for filename in filenames:
            with open(os.path.join(dirpath, filename), encoding="utf-8") as f:
                res[filename] = []
                for line in f:
                    res[filename].extend(tokenize(line.lower(), stopwords))
    return res


def get_passages(filepath):
    res = [""]
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            lines = line.split("\n")
            res[-1] = res[-1].join(lines[0])
            if len(lines) == 0:
                continue
            for i in range(1, len(lines)):
                res.append(lines[i])
    return res


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
            tf = files[f].count(word) / len(files[f])
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
            fl = False
            for s_word in sentences[s]:
                if dld(word, s_word) >= DELTA:
                    continue
                fl = True
                density += 1
            if fl:
                score += idfs.get(word, 0)

        density /= len(sentences[s])
        scores.append((s, score, density))

    scores.sort(key=lambda item: (item[1], item[2]), reverse=True)
    return [item[0] for item in scores][:n]


def dld(s1, s2):
    d = {}
    for i in range(-1, len(s1) + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, len(s2) + 1):
        d[(-1, j)] = j + 1
    for i in range(len(s1)):
        for j in range(len(s2)):
            cost = 0 if s1[i] == s2[j] else 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,
                d[(i, j - 1)] + 1,
                d[(i - 1, j - 1)] + cost,
            )
            if i and j and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[i - 2, j - 2] + 1)
    return d[len(s1) - 1, len(s2) - 1]


if __name__ == "__main__":
    main()

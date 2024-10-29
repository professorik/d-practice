import nltk
import sys
import pandas as pd
import numpy as np
import os
import re

FILE_MATCHES = 2

damping_fun = np.sqrt


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python main.py corpus-en")

    print_dir_info(sys.argv[1])

    file_words = tokenize_dir(sys.argv[1])
    print_file_tockens_info(file_words)

    tokens, freq = get_frequencies(file_words)
    tf = pd.DataFrame(freq, index=tokens)
    print(tf)
    tf = tf.apply(damping_fun)
    tfidf = tf_idf_transform(tf)
    print(tfidf)
    print(doc_cos_dist(tfidf))
    print(doc_binary(tfidf))


def tf_idf_transform(tf):
    N = len(tf.columns)
    idf = np.log(N / (tf > 0).sum(axis = 1))
    return tf.multiply(idf, axis="index")


def print_file_tockens_info(file_words):
    for file in file_words:
        print(f"Number of tokens ({file}):\t\t {len(file_words[file])}")


def print_dir_info(directory):
    for (dirpath, _, filenames) in os.walk(directory):
        print(f"Docs in corpus:\t\t {len(filenames)}")
        for filename in filenames:
            with open(os.path.join(dirpath, filename), encoding="utf-8") as f:
                print(f"{filename} size:\t\t {len(f.read())}")


def tokenize(document):
    stopwords = set(get_lines("stopwords-en.txt"))
    tokens = nltk.word_tokenize(document.lower())
    tokens = [t for t in tokens if not t in stopwords and re.search(f"^([a-z])([a-z\-\']*)$", t)]
    return tokens


def tokenize_dir(directory):
    res = {}
    for (dirpath, _, filenames) in os.walk(directory):
        for filename in filenames:
            with open(os.path.join(dirpath, filename), encoding="utf-8") as f:
                res[filename] = []
                for line in f:
                    res[filename].extend(tokenize(line.lower()))
    return res


def get_lines(filepath):
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


def get_frequencies(documents):
    res = {}
    appeared = set()
    for doc in documents.values():
        appeared.update(doc)
    tokens: list = list(appeared)
    tokens.sort()
    for doc in documents:
        res[doc] = []
        for token in tokens:
            res[doc].append(documents[doc].count(token))
    return tokens, res


def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def doc_cos_dist(tfidf):
    N = len(tfidf.columns)
    res = np.zeros((N, N), dtype=float)
    for i in range(N - 1):
        for j in range(i + 1, N):
            res[i][j] = res[j][i] = cosine_distance(tfidf.iloc[:, i].tolist(), tfidf.iloc[:, j].tolist())
    return pd.DataFrame(res, index=tfidf.columns, columns=tfidf.columns)


def doc_binary(tfidf):
    res = tfidf.copy()
    res[res != 0] = 1
    res = res.astype(int)
    return res


def test():
    freq = {}
    N = 8
    for i in range(1, N + 1):
        freq[f"Doc {i}"] = []
    tokens: list = []
    with open("test.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("\n")[0]
            lines = line.split(" ")
            if len(lines) != N + 1: continue
            tokens.append(lines[0])
            for i in range(1, len(lines)):
                freq[f"Doc {i}"].append(int(lines[i]))
    tf = pd.DataFrame(freq, index=tokens)
    print(tf)
    tf = tf.apply(damping_fun)
    tfidf = tf_idf_transform(tf)
    print(tfidf)
    print(doc_cos_dist(tfidf))
    print(doc_binary(tfidf))


if __name__ == "__main__":
    main()
    #test()
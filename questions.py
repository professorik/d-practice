import nltk
import sys
import string
import os
import math

FILE_MATCHES = 2
LANGUAGE = "english"
stopwords = set()


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python questions.py corpus-uk-lite uk")

    global stopwords
    stopwords = set(get_lines("stopwords-en.txt"))
    file_words = tokenize_dir(sys.argv[1])
    file_idfs = compute_idfs(file_words)
    #print(file_idfs)

    instr = "space rocket industry"
    query = set(tokenize(instr))
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)
    print(filenames)

    # sentences = dict()
    # for filename in filenames:
    #     for passage in get_lines(os.path.join(sys.argv[1], filename)):
    #         for sentence in nltk.sent_tokenize(passage):
    #             tokens = tokenize(sentence)
    #             if tokens:
    #                 sentences[sentence] = tokens
    #
    # idfs = compute_idfs(sentences)
    # print(idfs)


def tokenize(document):
    tokens = nltk.word_tokenize(document.lower())
    tokens = [t for t in tokens if not (t in string.punctuation or t in stopwords)]
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


if __name__ == "__main__":
    main()

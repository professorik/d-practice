_id = 0
smallfile = open("test/chunks/chunk_0.txt", "w")
with open("wiki_dump.tokenized.txt", encoding="utf-8") as corpus:
    for line in corpus:
        if line.startswith("Література"):
            if smallfile:
                smallfile.close()
                _id += 1
            small_filename = "test/chunks/chunk_{}.txt".format(_id)
            smallfile = open(small_filename, "w", encoding="utf-8")
        smallfile.write(line)
    if smallfile:
        smallfile.close()
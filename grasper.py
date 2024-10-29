import wikipedia

articles = [
    {"title": "NASA", "page_id": 18426568},
    {"title": "SpaceX", "page_id": 832774},
    {"title": "Cat", "page_id": 6678},
    {"title": "Lion", "page_id": 36896},
    {"title": "Cheetah", "page_id": 45609}
]


def write(name, text, url):
    f = open(f"corpus-en/{name}.txt", "w", encoding="utf-8")
    f.write(url + "\n\n")
    f.write(text)
    f.close()


wikipedia.set_lang("en")

for article in articles:
    page = wikipedia.page(pageid=article["page_id"])
    write(article["title"], page.content, page.url)
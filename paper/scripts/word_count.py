import re

text = open("main.tex", encoding="utf-8").read()
body = text.split(r"\begin{document}")[1].split(r"\end{document}")[0]

body2 = re.sub(
    r"\\begin\{(table|figure\*?|equation|tabular|thebibliography|itemize|enumerate)\}.*?\\end\{\1\}",
    " ", body, flags=re.S,
)
body3 = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", body2)
body3 = re.sub(r"[{}$]", " ", body3)
words = re.findall(r"[A-Za-z][A-Za-z'-]*", body3)
print("prose word count (tables/figs/eqs/lists stripped):", len(words))

# also count including itemize (since it now carries real content)
body2b = re.sub(
    r"\\begin\{(table|figure\*?|equation|tabular|thebibliography)\}.*?\\end\{\1\}",
    " ", body, flags=re.S,
)
body3b = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", body2b)
body3b = re.sub(r"[{}$]", " ", body3b)
words_b = re.findall(r"[A-Za-z][A-Za-z'-]*", body3b)
print("prose word count incl. itemize/list content:", len(words_b))

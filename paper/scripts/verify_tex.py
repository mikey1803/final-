import re
import os

text = open("main.tex", encoding="utf-8").read()

labels = set(re.findall(r"\\label\{([^}]+)\}", text))
refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", text))
print("labels not referenced:", labels - refs)
print("refs without label:", refs - labels)

figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)
print("includegraphics:", figs)
for f in figs:
    print(f, os.path.exists(os.path.join("figures", f)))

cites = set(re.findall(r"\\cite\{([^}]+)\}", text))
cite_keys = set()
for c in cites:
    cite_keys.update(k.strip() for k in c.split(","))
bibitems = set(re.findall(r"\\bibitem\{([^}]+)\}", text))
print("cite keys without bibitem:", cite_keys - bibitems)
print("bibitem unused:", bibitems - cite_keys)

begins = re.findall(r"\\begin\{([a-zA-Z*]+)\}", text)
ends = re.findall(r"\\end\{([a-zA-Z*]+)\}", text)
from collections import Counter
print("begin/end counts:", len(begins), len(ends))
print("type diff:", Counter(begins) - Counter(ends), Counter(ends) - Counter(begins))
print("brace balance:", text.count("{"), text.count("}"))

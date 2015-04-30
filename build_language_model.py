#!/usr/bin/env python
from collections import defaultdict
from math import log
import codecs

def utf8read(file): 
  return codecs.open(file, 'r', 'utf-8')

form = utf8read("data/train.lemma")

count = defaultdict(float)
# p = p(w2|w1)
p = defaultdict(float)

for line in form:
  spl = line.strip().replace(".", "").replace(",", "").split(" ")
  spl.insert(0, "<s>")
  spl.append("</s>")
  for i, word in enumerate(spl[:-1]):
    p[(spl[i+1], spl[i])] += 1.0
    count[spl[i]] += 1.0
  count[spl[-1]] += 1.0


language_model = codecs.open("lm_lemmas", 'w', 'utf-8')
for key in p:
  p[key] = p[key]/count[key[1]]
  #Use log-probabilities to avoid numerical underflow
  if p[key] == 0.0:
    p[key] = -1e300
  else:
    p[key] = log(p[key])
  
  language_model.write(key[0])
  language_model.write(" ")
  language_model.write(key[1])
  language_model.write(" ||| ")
  language_model.write(str(p[key]))
  language_model.write("\n")

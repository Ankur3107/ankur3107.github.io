---
title: "Spacy: Speeding Up Tokenization Processing Time"
last_modified_at: 2020-08-02T07:30:02-05:00
categories:
  - Blogs
tags:
  - NLP
  - Tokenization
excerpt: Tokenization is the process of turning a meaningful piece of data, such as an account number, into a random string of characters called a token
toc: true
toc_label: "Contents"
toc_icon: "cog"
---

![Cover Page](https://course.spacy.io/pipeline.png)

# Introduction

This weekend,I had been thinking about how can I optimize tokenization processing time. I have 6 Core CPUs Laptop. I wanted to utilize it.

# What is Tokenization?

![Tokenization](https://blog.floydhub.com/content/images/2020/02/tokenize.png)


Tokenization is the process of breaking text into pieces, called tokens. 

`Explanation: `How we humans understand language? We first try to *segment para into sentences* followed by *segment each sentence into words*.  After that, we try to link the words and make sense out it followed by link sentences to make overall sense.

There are two types of tokenizations:
- Sentence Tokenization
- Word Tokenization

`Disclaimer:` Here we discuss how we can process word tokenization faster.


## Experiment with Code

Let's load required packages first:

```python
import spacy
from fastprogress import *
from tqdm import tqdm_notebook, tqdm
from concurrent.futures import ProcessPoolExecutor
import re
nlp = spacy.load('en_core_web_sm')
```

`1. Multi-threaded approach:` Spacy has **.pipe** generator. It has n_threads and n_process parameters.

```python
def multi_thread_based_tokenizations(nlp, text_list, batch_size=1000, n_threads=4, n_process=1):
    docs = nlp.pipe(text_list,batch_size=batch_size, n_threads = n_threads, n_process=n_process)
    word_sequences = []

    for doc in tqdm(docs):
        word_seq = []
        for token in doc:
            word_seq.append(token.text)
        word_sequences.append(word_seq)
    return word_sequences
```
### Time Statistics: Multi-threaded approach

I took *10000 sentences* and ran experiments with a combination of batch_size, threads, process(CPU). These are the stats:

batch_size|n_threads|n_process|Time(s)
|---|---|---|---|
1000|2|4|17.773555
1500|2|4|18.986561
2000|2|4|20.790800
2500|2|4|21.243879
3000|2|4|21.849630
3500|2|4|22.805901
1000|2|2|25.049998
4000|2|4|26.027611
2500|2|2|26.379638
1500|2|2|26.804577
2000|2|2|27.956490
4000|2|2|29.257992
3000|2|2|29.973307
3500|2|2|31.415203

`2. Multi-processing approach:` I have used *concurrent.futures* package to parallelize tokenization code.

```python

def parallel(func, arr, max_workers=4):
    if max_workers<2: results = list(progress_bar(map(func, enumerate(arr)), total=len(arr)))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            return list(progress_bar(ex.map(func, enumerate(arr)), total=len(arr)))
    if any([o is not None for o in results]): return results

class TokenizeProcessor():
    def __init__(self, nlp, chunksize=2000, max_workers=4): 
        self.chunksize,self.max_workers = chunksize,max_workers
        self.tokenizer = nlp.tokenizer

    def proc_chunk(self, args):
        i,chunk = args
        docs = [[d.text for d in doc] for doc in self.tokenizer.pipe(chunk)]
        return docs

    def __call__(self, items): 
        toks = []
        chunks = [items[i: i+self.chunksize] for i in (range(0, len(items), self.chunksize))]
        toks = parallel(self.proc_chunk, chunks, max_workers=self.max_workers)
        return sum(toks, [])

```

### Time Statistics: Multi-processing approach

I took *10000 sentences* and ran experiments with a combination of batch_size, process(CPU). These are the stats:

batch_size|n_process|Time(s)
|---|---|---|
4000|2|3.468632
4000|4|3.487540
3500|2|4.040751
3000|2|4.352866
3000|4|4.422525
2500|4|4.445214
2500|2|4.512357
3500|4|4.737216
2000|4|5.303331
2000|2|5.378104
1500|2|7.097535
1500|4|7.247377
1000|4|9.837863
1000|2|9.938355

# Observation

- Multiprocessing with *concurrent.futures* gave the best result, took 3.4sec per 10000 sentences.
- Multithreading with Spacy also gave better results as compared to the native approach, when I increased processes(CPUs), it took 17.7sec per 10000 sentences.

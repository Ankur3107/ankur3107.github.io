---
title: "BERTology: How Bert Works"
last_modified_at: 2020-03-03T10:30:02-05:00
categories:
  - Blogs
tags:
  - NLP
  - BERT
excerpt: BERT (Bidirectional Encoder Representations from Transformers)
toc: true
toc_label: "Contents"
toc_icon: "cog"
---

![Cover Page](/assets/images/bertology.png)

<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-2118670497450280"
     crossorigin="anonymous"></script>

# Overview of BERT architecture

Fundamentally, BERT is a stack of Transformer
encoder layers (Vaswani et al., 2017) which consist
of multiple “heads”, i.e., fully-connected neural
networks augmented with a self-attention mechanism. For every input token in a sequence, each
head computes key, value and query vectors, which
are used to create a weighted representation. The
outputs of all heads in the same layer are combined
and run through a fully-connected layer. Each layer
is wrapped with a skip connection and layer normalization is applied after it.

# BERT embeddings

  1.  BERT
embeddings occupy a narrow cone in the vector
space, and this effect increases from lower to higher
layers. That is, two random words will on average have a much higher cosine similarity than expected if embeddings were directionally uniform
(isotropic).

# Syntactic knowledge

  1.  BERT representations are hierarchical rather than linear
  2.  BERT embeddings encode information about parts of speech, syntactic chunks
and roles
  3.  Syntactic structure is not
directly encoded in self-attention weights, but
they can be transformed to reflect it.
  4.   Able to learn transformation matrices
that would successfully recover much of the Stanford Dependencies formalism for PennTreebank
data 

  5.  BERT takes
subject-predicate agreement into account when
performing the cloze task

  6.  BERT is better able to
detect the presence of NPIs (e.g. ”ever”) and the
words that allow their use (e.g. ”whether”) than
scope violations.

#  Semantic knowledge

  1.  BERT encodes information about entity types, relations,
semantic roles, and proto-roles, since this information can be detected with probing classifiers.

  2.  BERT struggles with representations of numbers. Addition and number decoding tasks showed
that BERT does not form good representations for
floating point numbers and fails to generalize away
from the training data

#   World knowledge

  1.  BERT cannot reason based on its
world knowledge. Forbes et al. (2019) show that
BERT can “guess” the affordances and properties
of many objects, but does not have the information
about their interactions (e.g. it “knows” that people
can walk into houses, and that houses are big, but
it cannot infer that houses are bigger than people.)

  2.  At the
same time, Poerner et al. (2019) show that some
of BERT’s success in factoid knowledge retrieval
comes from learning stereotypical character combinations, e.g. it would predict that a person with
an Italian-sounding name is Italian, even when it is
factually incorrect.

#  Self-attention heads

![Self-attention heads](/assets/images/attention_layer_vis.png)

1.  Most selfattention heads do not directly encode any nontrivial linguistic information, since less than half
of them had the “heterogeneous” pattern

2.  Some BERT heads seem to specialize in certain types of syntactic relations.

3.  No single head has the complete
syntactic tree information, in line with evidence
of partial knowledge of syntax 

4.  Even when
attention heads specialize in tracking semantic
relations, they do not necessarily contribute to
BERT’s performance on relevant tasks.


# BERT layers

![BERT layers](/assets/images/bert_layer.png)


1.  The lower layers have the most linear word order
information

2.  syntactic information is the most prominent in the middle BERT layers



****Checkout the [Offical Paper](https://arxiv.org/pdf/2002.12327v1.pdf) ***

<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-2118670497450280"
     crossorigin="anonymous"></script>

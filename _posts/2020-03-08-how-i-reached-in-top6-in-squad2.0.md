---
title: "SQuAD 2.0: How I reached in Top 6"
last_modified_at: 2020-03-08T10:30:02-05:00
categories:
  - Blogs
tags:
  - NLP
  - ALBERT
  - TRANSFORMER
---

![Cover Page](/assets/images/SQuAD_Result.png)

## What is SQuAD

* Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.

* SQuAD2.0 combines the 100,000 questions in SQuAD1.1 with over 50,000 unanswerable questions written adversarially by crowdworkers to look similar to answerable ones. To do well on SQuAD2.0, systems must not only answer questions when possible, but also determine when no answer is supported by the paragraph and abstain from answering.

*   Types of examples in SQuAD 2.0

    ![Cover Page](/assets/images/question_type.png)

*   Dataset statistics of SQuAD 2.0

    ![Cover Page](/assets/images/datasets.png)


## Strategy

*   Data Pre-Processing:

    Data pre-processing was the most time consuming process and it taken around 6-8h of computation time. It includes:

            1. Text Cleaning
            2. Tokenization (Sentencepiece)
            3. Data-format prepration for model input
            4. Serialization into TFRecords

* Tuned Albert:

    Tuned Abert model Architecture build using albert transformer network. 

        1.  Base Network - Transformer
        2.  Albert Architecture (Multi Headed Self Attention)
        3.  QA Layer Architecture
        4.  Parameter Tuning - Hidden layer size, attention in QA Layers etc

*   Training Strategy:

    



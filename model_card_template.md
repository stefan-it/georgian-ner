---
language:
- en
- ka
license: mit
tags:
- flair
- token-classification
- sequence-tagger-model
base_model: {{ base_model }}
widget:
- text: {{ widget_text }}
---

# Fine-tuned English-Georgian NER Model with Flair

This Flair NER model was fine-tuned on the WikiANN dataset
([Rahimi et al.](https://www.aclweb.org/anthology/P19-1015) splits)
using {{ base_model_short }} as backbone LM.

**Notice**: The dataset is very problematic, because it was automatically constructed.

We did manually inspect the development split of the Georgian data and found
a lot of bad labeled examples, e.g. DVD ( 💿 ) as `ORG`.

## Fine-Tuning

The latest
[Flair version](https://github.com/flairNLP/flair/tree/f30f5801df3f9e105ed078ec058b4e1152dd9159)
is used for fine-tuning.

We use English and Georgian training splits for fine-tuning and the
development set of Georgian for evaluation.

A hyper-parameter search over the following parameters with 5 different seeds per configuration is performed:

* Batch Sizes: {{ batch_sizes }}
* Learning Rates: {{ learning_rates }}

More details can be found in this [repository](https://github.com/stefan-it/georgian-ner).

## Results

A hyper-parameter search with 5 different seeds per configuration is performed and micro F1-score on development set
is reported:

{{ results }}

The result in bold shows the performance of this model.

Additionally, the Flair [training log](training.log) and [TensorBoard logs](tensorboard) are also uploaded to the model
hub.

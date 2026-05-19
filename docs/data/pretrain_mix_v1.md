# TinyLLM Pretraining Data Mix v1

This document defines the default pretraining data mix for the first TinyLLM base-model runs. The project is Chinese-priority bilingual, while preserving English, code, math, long-document, and limited multilingual coverage.

Percentages are measured by token count after cleaning, deduplication, quality filtering, and tokenization. Do not measure the mix by file size, raw document count, or source dataset size.

## Default Mix

| Bucket | Weight | Content |
|---|---:|---|
| Chinese general web | 24% | News, encyclopedia-like web pages, blogs, long forum posts, and general knowledge pages. |
| Chinese education, knowledge, and encyclopedia | 16% | Textbook-style text, popular science, encyclopedic writing, course materials, and explanatory content. |
| Chinese books and long-form documents | 8% | Long articles, public books, reports, and coherent long documents. |
| Chinese professional, QA, and community | 4% | Technical QA, vertical-domain articles, and high-quality experience-based answers. |
| English general web | 12% | High-quality web text similar to FineWeb and FineWeb-Edu sources. |
| English education, knowledge, and encyclopedia | 8% | Educational web pages, Wikipedia/reference-style text, tutorials, and explanatory content. |
| English books and long-form documents | 4% | Books, long-form documents, essays, and coherent long text. |
| English academic and technical documents | 4% | arXiv/PDF-like documents, technical reports, and paper body text. |
| Code and code documentation | 10% | Python, JS/TS, Java, C/C++, Go, Rust, Shell, SQL, and Markdown documentation. |
| Math, STEM, and reasoning text | 7% | Math solutions, proofs, STEM textbooks, and verifiable reasoning text in pretraining format rather than chat format. |
| Other multilingual data | 3% | Japanese, Korean, French, Spanish, German, and other small generalization data. |

Total: 100%.

## Data Rules

- Apply these percentages only after cleaning, deduplication, quality filtering, and tokenization.
- Do not mix SFT, chat, or instruction data into base pretraining. Keep those for post-training.
- Every bucket must record source, license, cleaning version, token count, filtering ratio, deduplication ratio, and final shard manifest.
- Chinese and English web buckets must run language identification, duplicate-ratio filtering, boilerplate/ad filtering, low-information-density filtering, PII filtering, and benchmark-contamination filtering.
- Code data must run license filtering, long-file splitting, duplicate repository/file filtering, generated-file filtering, secret filtering, and benchmark-contamination filtering.

## Training Stages

- S1 broad pretrain: use this default mix for the 64M trial model first, then for the 120M-130M main MVP model.
- S2 knowledge/reasoning continue pretrain: after the base run is stable, optionally raise code, math, STEM, and academic/technical data to 25%-30% combined and reduce general web data.
- S3 long-context continue pretrain: when extending from 4K to 8K context, raise books, papers, reports, and long-web data to train long-document continuity.

## Rationale

Public model teams rarely publish exact pretraining mixture ratios, but the common pattern is stable: broad web data provides the base distribution; books, encyclopedia, academic, and technical documents raise knowledge density; code and math need explicit weighting because they are underrepresented in ordinary web data.

Qwen3 describes pretraining over web, PDF-like documents, math, code, and synthetic educational/question-answer/code data, with a later phase that increases STEM, coding, and reasoning data. Early LLaMA used a web-heavy mixture with GitHub, Wikipedia, books, arXiv, and StackExchange. Dolma and DCLM also reinforce the same broad categories: web, academic, code, books, and encyclopedic/reference text.

Reference links:

- [Qwen3 blog](https://qwenlm.github.io/blog/qwen3/)
- [LLaMA paper page](https://ai.meta.com/research/publications/llama-open-and-efficient-foundation-language-models/)
- [Dolma](https://allenai.github.io/dolma/)
- [DCLM](https://github.com/mlfoundations/dclm)
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2)
- [ChineseWebText2.0](https://huggingface.co/datasets/CASIA-LM/ChineseWebText2.0)

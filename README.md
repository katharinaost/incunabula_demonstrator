# incunabula_demonstrator

Demonstrates the processing of a digitized incunable to prepare it for NLP tasks.

## Components

### 01_download_images
Downloads digitized images of the incunable from the digital collections of ULB Düsseldorf.

### 02_text_recognition
Output from performing optical character recognition (OCR) on the downloaded images. The [Transkribus Latin Incunabula Reichenau model](https://app.transkribus.org/models/public/text/latin-incunabula-reichenau) has been used for OCR.

### 03_resolve_abbreviations
Resolves medieval Latin abbreviations in the transcribed text. Includes fine-tuning scripts for ByT5-small and Michael Schonhardt's [BDD-LoRA adapter](https://huggingface.co/mschonhardt/byt5-base-bdd-expansion-lora-v4-l40s) for ByT5-base, trained on annotated data where abbreviated lines from the [Training Data Incunabula Reichenau dataset](https://doi.org/10.5281/zenodo.11046062) have been labeled with their resolutions.

### 04_rejoin_lines
Rejoins hyphenated words that were split across lines. Implements both BiLSTM and logistic regression approaches to identify and merge split words, reconstructing a continuous text.

### 05_fix_punctuation
Corrects and standardizes punctuation in the processed text. Includes scripts for stripping the training data of the [LatinCy pipeline](https://huggingface.co/latincy) from punctuation.

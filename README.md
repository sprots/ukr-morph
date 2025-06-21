# Ukrainian Morpheme Segmentation Dataset
A dataset for Ukrainian morpheme segmentation and labeling, designed for linguistic research, NLP tasks, and morphological analysis.

## Dataset Description

This dataset contains Ukrainian lemmas surface-segmented into morphemes (roots, prefixes, suffixes, interfixes, postfixes) with morpheme type labels. Each entry follows the format:
morph:type(/morph:type)*,tier

### Morpheme Types
- `R` - Root
- `P` - Prefix
- `S` - Suffix
- `I` - Interfix
- `F` - Flexion (ending)
- `H` - Hyphen
- `X` - Reflexive suffix (-ся/-сь), some other particles like -би, -б, -то, -но, etc.

### Tiers explanation
1. Segmented and tagged data from Клименко, Н. Ф., et al. "Словник афіксальних морфем української мови." К.: Ін-т мовознавства ім. ОО Потебні НАН України (1998) (~11k words) and transferred data from Тихонов А. Н. Словообразовательный словарь русского языка. М. : Астрель, 2008. Т. 1–2 (~37k words).
2. Segmented data from Яценко, Іван Тимофійович. Морфемний аналіз: Словник-довідник. Вища Школа, 1981 (~77k new words). Tagged by rule-based scipt. Manually corrected.
3. Unsegmented words from Словник української мови : в 11 т. / І. К. Білодід (гол. ред.) та ін. Київ, 1970–1980 (~25k new words). Segmented and tagged by CNN model trained on previous data. Manually corrected.
4. VESUM base, geo-ukr-koatuu (4g), geo-ukr-hydro (4h).
Some words were resegmented according to Етимологічний словник української мови: В 7 т. К.: Наукова думка, 1982. – 632 с.

## Example Data
- у:P/кра:R/їн:S/а:F,3
- у:P/кра:R/їн:S/ець:S,1
- у:P/кра:R/їн:S/к:S/а:F,1
- у:P/кра:R/їн:S/оч:S/к:S/а:F,2
- у:P/кра:R/їн:S/ств:S/о:F,2
- у:P/кра:R/їн:S/ськ:S/ий:F,1
- у:P/кра:R/їн:S/із:S/аці:S/я:F,1

## Statistics
- v0.1 Total words: 154k
- v0.2 Total words: 286k

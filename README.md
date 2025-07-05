# Ukrainian Morpheme Segmentation Dataset
A dataset for Ukrainian morpheme segmentation and labeling, designed for linguistic research, NLP tasks, and morphological analysis.

## Dataset Description

This dataset contains Ukrainian lemmas surface-segmented into morphemes (roots, prefixes, suffixes, interfixes, postfixes) with morpheme type labels. Each entry follows the format:
morph:type(/morph:type)*,tier

v0.3 data structure update:
- Multext East full tag (~20k entries with UNK tag to be addressed);
- Multex East POS tag predicted by Spacy and Flair, single if they agree and both if not;
- ambiguous segmentation/tagging flag;
- root(s), to be depreived of alternation, palatalisations, etc.
v0.4 data structure update:
- Ubertext corpus frequency in format 10+exp, mantissa, Multext East category code;
- inflectional paradigm in dict_uk notation;
- morpheme tags sequence pattern extracted from morph_tagged_lemma


### Morpheme Types
- `R` - Root
- `P` - Prefix
- `S` - Suffix
- `I` - Interfix
- `F` - Flexion (ending)
- `H` - Hyphen
- `X` - Reflexive postfix (-ся/-сь), some other particles like -би, -б, -то, -но, etc.

### Tiers explanation
1. Segmented and tagged data from Клименко, Н. Ф., et al. "Словник афіксальних морфем української мови." К.: Ін-т мовознавства ім. ОО Потебні НАН України (1998) (~11k words) and transferred data from Тихонов А. Н. Словообразовательный словарь русского языка. М. : Астрель, 2008. Т. 1–2 (~37k words).
2. Segmented data from Яценко, Іван Тимофійович. Морфемний аналіз: Словник-довідник. Вища Школа, 1981 (~77k new words). Tagged by rule-based scipt. Manually corrected.
3. Unsegmented words from Словник української мови : в 11 т. / І. К. Білодід (гол. ред.) та ін. Київ, 1970–1980 (~25k new words). Segmented and tagged by CNN model trained on previous data.
4. VESUM base, geo-ukr-koatuu (4g), geo-ukr-hydro (4h) (~132k). Segmented and tagged by CNN model trained on previous data.

Some words were resegmented according to Етимологічний словник української мови: В 7 т. К.: Наукова думка, 1982. – 632 с.

## Example Data
- multext,pos,freq,paradigm,morph_tagged_lemma,root,tier,ambiguity,lemma,reversed_lemma,Pattern
- Npfsnn,Np;N,75Np,n10,у:P/кра:R/їн:S/а:F,кра,3,,україна,анїарку,PRSF
- Ncmsny,N,64N,n22.p.a.<,у:P/кра:R/їн:S/ець:S,кра,1,,українець,ьценїарку,PRSS
- Ncfsny,N,54N,n10.p2.ko.<,у:P/кра:R/їн:S/к:S/а:F,кра,1,,українка,акнїарку,PRSSF


## Statistics
- v0.1 Total words: 154k
- v0.2 Total words: 286k

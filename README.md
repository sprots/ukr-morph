# Ukrainian Morpheme Segmentation Dataset
A dataset for Ukrainian morpheme segmentation and labeling, designed for linguistic research, NLP tasks, and morphological analysis.

## Dataset Description

This dataset contains Ukrainian lemmas surface-segmented into morphemes (roots, prefixes, suffixes, interfixes, postfixes) with morpheme type labels. Each entry follows the format:
morph:type(/morph:type)*,tier

v0.3 data structure update:
- Multext East full tag;
- Multex East POS tag predicted by Spacy and Flair, single if they agree and both if not;
- ambiguous segmentation/tagging flag;
- root(s), to be depreived of alternation, palatalisations, etc.

v0.4 data structure update:
- Ubertext corpus frequency in format 10+exp, mantissa, Multext East category code;
- inflectional paradigm in dict_uk notation;
- morpheme tags sequence pattern extracted from morph_tagged_lemma.

v0.5 data structure update:
- cluster number (5k clusters grouped by feature (pos, paradigm, lemma length, unigram, bigram, affix, etc similarity);
- ambiguity field now features lemma tag (c - core dataset lemma; r - rear morpheme; u - rear root; a - rear affix);
- predicted lemma by model trained on a core dataset;
- model confidence score;
- error type (0 = No error, 1-6 error by frequency from more to less frequent, 9 core dataset error; Recommendations: n = No action needed, r = Review, a = Add to core dataset, e = Exclude from core dataset)

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
5. UberTExt 2.0 1M most frequent lemmas added that resulted in 582k new noisy lemmas to be further reduced/corrected. Segmented and tagged by CNN+CRF model (P/R/F1=0.975, 90% word accuracy) trained on 80% of 286k dataset.

Some words were resegmented according to Етимологічний словник української мови: В 7 т. К.: Наукова думка, 1982. – 632 с.

## Example Data
- morph_tagged_lemma,root,ambiguity,cluster,multext,mhv_pos,pos,freq,paradigm,mhv_class,tier,lemma,reversed_lemma,Pattern,predicted,confidence,error
- у:P/кра:R/їн:S/а:F,кра,0r,319,Npfsnn,5,Np;N,75Np,n10,1283,3,україна,анїарку,PRSF,у:P/кра:R/їн:S/а:F,0.875,0
- у:P/кра:R/їн:S/ець:S,кра,0,2547,Ncmsny,7,N,64N,n22.p.a.<,1540,1,українець,ьценїарку,PRSS,у:P/кра:R/їн:S/ець:S,0.91,0
- у:P/кра:R/їн:S/із:S/ат:S/ор:S,кра,0,1843,Ncmsny,8,N,25N,n20.a.p.ke.<,1607,4,українізатор,ротазінїарку,PRSSSS,у:P/кра:R/їн:S/із:S/ат:S/ор:S,0.762,0

## Statistics
- v0.1 Total lemmas: 154k
- v0.2 Total lemmas: 286k
- v0.6 Total lemmas: 868k

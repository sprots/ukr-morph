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

v0.7 data structure update:
- mhv_pos, mhv_class from mphdict that duplicate pos/multext, paradigm.
- uk_confidence,ru_confidence,predicted_lang to address tier 5 data contamination.

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
4. VESUM base, geo-ukr-koatuu (4g), geo-ukr-hydro (4h) (~132k), names (4n), etc. Segmented and tagged by CNN model trained on previous data.
5. UberTExt 2.0 1M most frequent lemmas added that resulted in 582k new noisy lemmas to be further reduced/corrected. Segmented and tagged by CNN+CRF model (P/R/F1=0.975, 90% word accuracy) trained on 80% of 286k dataset.
6. Missing ~30k lemmas from unsegmented mphdict.
7. Missing ~9k lemmas from resegmented Карпіловська Є. А. Кореневий гніздовий словник української мови: Гнізда слів з вершинами – омографічними коренями. – К.: Укр. енциклопедія, 2002. – 912 с. 
Some words were resegmented according to Етимологічний словник української мови: В 7 т. К.: Наукова думка, 1982. – 632 с.

## Example Data
- lemma,morph_tagged_lemma,predicted,pos,multext,freq,tier,confidence,error,uk_confidence,ru_confidence,predicted_lang,ambiguity,mhv_pos,mhv_class,paradigm,cluster,reversed_lemma,pattern
- українолюбство,у:P/кра:R/їн:S/о:I/люб:R/ств:S/о:F,у:P/кра:R/їн:S/о:I/люб:R/ств:S/о:F,N,Ncnsnn,17N,4,0.9,0,0.99,0.01,uk,0,13,1992,n2n,2526,овтсбюлонїарку,PRSIRSF
- україномислячий,у:P/кра:R/їн:S/о:I/мисл:R/яч:S/ий:F,,A,Afpmsn,17A,4x,0.81,,0.99,0.01,uk,0,11,2321,adj,2526,йичялсимонїарку,PRSIRSF

## Statistics
- v0.1 Total lemmas: 154k
- v0.2 Total lemmas: 286k
- v0.6 Total lemmas: 868k
- v0.7 Total lemmas: 889k

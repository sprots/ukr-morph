import sys
import argparse
import re
from typing import List, Set

def is_roman(s: str) -> bool:
    """Check if string is a Roman numeral."""
    # Normalize to uppercase
    s = s.upper()

    # Map Cyrillic letters to Latin equivalents (visually similar ones)
    translation_table = str.maketrans("ІХСМ", "IXCM")
    s_translated = s.translate(translation_table)

    # Roman numeral pattern - matches valid Roman numerals
    roman_pattern = r'^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'

    return re.match(roman_pattern, s_translated) is not None

def is_number(s: str) -> bool:
    """Check if string represents a number."""
    try:
        float(s)
        return True
    except ValueError:
        return False

def tags_to_msd(tags: List[str], lemma: str) -> str:
    """Convert Ukrainian morphological tags to MULTEXT-East MSD format."""
    msd = ['-'] * 15  # Initialize 15-position MSD template
    
    # Convert tags to set for O(1) lookup
    tag_set = set(tags)
    
    # Handle special cases and determine base POS
    is_pronoun = 'pron' in tag_set
    is_predic = 'predic' in tag_set
    is_numr = 'numr' in tag_set
    is_adj = 'adj' in tag_set
    is_adjp = 'adjp' in tag_set
    is_abbr = 'abbr' in tag_set
    base_pos = tags[0] if tags else ''

    # Special case: numeral + adjective = Numeral
    if is_numr and is_adj:
        msd[0] = 'M'
    # POS mapping (Position 0)
    elif is_pronoun:
        msd[0] = 'P'  # Pronoun
    #elif is_abbr:
    #    msd[0] = 'Y'
    else:
        pos_map = {
            'noun': 'N',
            'verb': 'V',
            'adj': 'A',
            'adv': 'R',
            'advp': 'V',  # Gerund
            'adjp': 'A',  # Adjectival participle → Adjective
            'prep': 'S',
            'conj': 'C',
            'part': 'Q',
            'intj': 'I',
            'numr': 'M',
            'noninfl': 'X',
            'onomat': 'I',
            'insert': 'X',
        }
        msd[0] = pos_map.get(base_pos, 'X')

    # Process by POS category
    if msd[0] == 'N':  # Noun
        _process_noun(msd, tag_set)
    elif msd[0] == 'V':  # Verb
        _process_verb(msd, tag_set, base_pos)
    elif msd[0] == 'A':  # Adjective
        _process_adjective(msd, tag_set, base_pos, is_adjp)
    elif msd[0] == 'P':  # Pronoun
        _process_pronoun(msd, tag_set)
    elif msd[0] == 'R':  # Adverb
        _process_adverb(msd, tag_set)
    elif msd[0] == 'C':  # Conjunction
        _process_conjunction(msd, tag_set, lemma)
    elif msd[0] == 'M':  # Numeral
        _process_numeral(msd, tag_set, lemma, is_adj)
    elif msd[0] == 'S':  # Preposition
        _process_preposition(msd, tag_set, lemma)

    # Convert to string and remove trailing hyphens
    msd_str = ''.join(msd).rstrip('-')
    return msd_str if msd_str else '-'

def _process_noun(msd: List[str], tag_set: Set[str]) -> None:
    """Process noun-specific MSD positions."""
    # Type (Position 1)
    if any(t in tag_set for t in ['prop', 'geo', 'fname', 'lname', 'pname']):
        msd[1] = 'p'  # proper
    else:
        msd[1] = 'c'  # common
    
    # Gender (Position 2)
    if 'p' in tag_set and not any(g in tag_set for g in ['m', 'f', 'n']):
        msd[2] = '-'  # No gender for pluralia tantum
    else:
        if 'm' in tag_set: 
            msd[2] = 'm'
        elif 'f' in tag_set: 
            msd[2] = 'f'
        elif 'n' in tag_set: 
            msd[2] = 'n'
        elif 'c' in tag_set: 
            msd[2] = 'c'  # common gender
    
    # Number (Position 3)
    if 'p' in tag_set or 'ns' in tag_set: 
        msd[3] = 'p'  # plural
    else:
        msd[3] = 's'  # singular
    
    # Case (Position 4)
    if 'nv' in tag_set:
        msd[4] = '-'  # non-declining
    else:
        case_map = {
            'v_naz': 'n', 'v_rod': 'g', 'v_dav': 'd', 
            'v_zna': 'a', 'v_oru': 'i', 'v_mis': 'l', 'v_kly': 'v'
        }
        for tag, case in case_map.items():
            if tag in tag_set:
                msd[4] = case
                break
    
    # Animacy (Position 5)
    if 'anim' in tag_set or 'unanim' in tag_set: 
        msd[5] = 'y'  # animate
    elif 'inanim' in tag_set: 
        msd[5] = 'n'  # inanimate

def _process_verb(msd: List[str], tag_set: Set[str], base_pos: str) -> None:
    """Process verb-specific MSD positions."""
    # Type (Position 1)
    msd[1] = 'm'  # main verb (assuming all are main verbs)
    
    # Aspect (Position 2)
    if 'imperf' in tag_set: 
        msd[2] = 'p'  # imperfective = progressive
    elif 'perf' in tag_set: 
        msd[2] = 'e'  # perfective 
    else: 
        msd[2] = 'b'  # biaspectual

    # VForm (Position 3)
    if 'impers' in tag_set: 
        msd[3] = 'o'  # impersonal
    elif 'inf' in tag_set: 
        msd[3] = 'n'  # infinitive
    elif 'impr' in tag_set: 
        msd[3] = 'm'  # imperative
    elif 'advp' in tag_set or base_pos == 'advp': 
        msd[3] = 'g'  # gerund
    else: 
        msd[3] = 'i'  # indicative
    
    # Tense (Position 4)
    tense_map = {'pres': 'p', 'futr': 'f', 'past': 's'}
    for tag, tense in tense_map.items():
        if tag in tag_set:
            msd[4] = tense
            break
    
    # Person (Position 5)
    person_map = {'1': '1', '2': '2', '3': '3'}
    for tag, person in person_map.items():
        if tag in tag_set:
            msd[5] = person
            break
    
    # Number (Position 6)
    if 'p' in tag_set: 
        msd[6] = 'p'  # plural
    elif 's' in tag_set: 
        msd[6] = 's'  # singular

    # Gender (Position 7) - for past tense
    gender_map = {'m': 'm', 'f': 'f', 'n': 'n'}
    for tag, gender in gender_map.items():
        if tag in tag_set:
            msd[7] = gender
            break

def _process_adjective(msd: List[str], tag_set: Set[str], base_pos: str, is_adjp: bool) -> None:
    """Process adjective-specific MSD positions."""
    # Type (Position 1)
    if is_adjp:
        msd[1] = 'p'  # participle
    elif 'ord' in tag_set: 
        msd[1] = 'o'  # ordinal
    else: 
        msd[1] = 'f'  # general adjective
    
    # Degree (Position 2)
    if 'compc' in tag_set: 
        msd[2] = 'c'  # comparative
    elif 'comps' in tag_set: 
        msd[2] = 's'  # superlative
    elif msd[1] == 'f':
        msd[2] = 'p'  # positive
    
    # Gender (Position 3)
    gender_map = {'m': 'm', 'f': 'f', 'n': 'n', 'c': 'c'}
    for tag, gender in gender_map.items():
        if tag in tag_set:
            msd[3] = gender
            break
    
    # Number (Position 4)
    msd[4] = 'p' if 'p' in tag_set else 's'
    
    # Case (Position 5)
    if 'nv' in tag_set:
        msd[5] = '-'  # non-declining
    else:
        case_map = {
            'v_naz': 'n', 'v_rod': 'g', 'v_dav': 'd', 
            'v_zna': 'a', 'v_oru': 'i', 'v_mis': 'l', 'v_kly': 'v'
        }
        for tag, case in case_map.items():
            if tag in tag_set:
                msd[5] = case
                break
    
    # Definiteness (Position 6)
    if 'long' in tag_set: 
        msd[6] = 'f'  # full (long form)
    elif 'short' in tag_set: 
        msd[6] = 's'  # short form
    
    # Animacy (Position 7) - only for accusative
    if 'v_zna' in tag_set:
        if 'ranim' in tag_set: 
            msd[7] = 'y'  # animate
        elif 'rinanim' in tag_set: 
            msd[7] = 'n'  # inanimate
    
    # Aspect (Position 8) - for participles
    if is_adjp:
        if 'imperf' in tag_set: 
            msd[8] = 'p'
        elif 'perf' in tag_set: 
            msd[8] = 'e'
    
    # Voice (Position 9) - for participles
    if is_adjp:
        if 'actv' in tag_set: 
            msd[9] = 'a'
        elif 'pasv' in tag_set: 
            msd[9] = 'p'
    
    # Tense (Position 10) - for participles
    if is_adjp:
        if 'pres' in tag_set: 
            msd[10] = 'p'
        elif 'past' in tag_set: 
            msd[10] = 's'

def _process_pronoun(msd: List[str], tag_set: Set[str]) -> None:
    """Process pronoun-specific MSD positions."""
    # Type (Position 1)
    pronoun_type_map = {
        'pers': 'p', 'refl': 'x', 'pos': 's', 'dem': 'd',
        'int': 'q', 'rel': 'r', 'neg': 'z', 'ind': 'i',
        'gen': 'g', 'emph': 'h'
    }
    for tag, pron_type in pronoun_type_map.items():
        if tag in tag_set:
            msd[1] = pron_type
            break

    # Referent Type (Position 2) - only for possessive pronouns
    if 'pos' in tag_set: 
        msd[2] = 's'
    
    # Person (Position 3)
    person_map = {'1': '1', '2': '2', '3': '3'}
    for tag, person in person_map.items():
        if tag in tag_set:
            msd[3] = person
            break
    
    # Gender (Position 4)
    gender_map = {'m': 'm', 'f': 'f', 'n': 'n'}
    for tag, gender in gender_map.items():
        if tag in tag_set:
            msd[4] = gender
            break
    else:
        msd[4] = 'c'  # common gender as default
    
    # Animacy (Position 5)
    if 'anim' in tag_set or 'unanim' in tag_set: 
        msd[5] = 'y'
    elif 'inanim' in tag_set: 
        msd[5] = 'n'
    
    # Number (Position 6)
    msd[6] = 'p' if 'p' in tag_set else 's'

    # Case (Position 7)
    if 'nv' in tag_set:
        msd[7] = '-'
    else:
        case_map = {
            'v_naz': 'n', 'v_rod': 'g', 'v_dav': 'd', 
            'v_zna': 'a', 'v_oru': 'i', 'v_mis': 'l', 'v_kly': 'v'
        }
        for tag, case in case_map.items():
            if tag in tag_set:
                msd[7] = case
                break
            
    # Syntactic Type (Position 8)
    syntactic_map = {'noun': 'n', 'adj': 'a', 'adv': 'r'}
    for tag, syn_type in syntactic_map.items():
        if tag in tag_set:
            msd[8] = syn_type
            break

def _process_adverb(msd: List[str], tag_set: Set[str]) -> None:
    """Process adverb-specific MSD positions."""
    # Degree (Position 1)
    if 'compc' in tag_set: 
        msd[1] = 'c'  # comparative
    elif 'comps' in tag_set: 
        msd[1] = 's'  # superlative
    else: 
        msd[1] = 'p'  # positive

def _process_conjunction(msd: List[str], tag_set: Set[str], lemma: str) -> None:
    """Process conjunction-specific MSD positions."""
    # Type (Position 1)
    if 'coord' in tag_set: 
        msd[1] = 'c'  # coordinative
    else: 
        msd[1] = 's'  # subordinative

    # Formation (Position 2)
    if '-' in lemma or ' ' in lemma: 
        msd[2] = 'c'  # compound
    else: 
        msd[2] = 's'  # simple

def _process_numeral(msd: List[str], tag_set: Set[str], lemma: str, is_adj: bool) -> None:
    """Process numeral-specific MSD positions."""
    # Form (Position 1)
    if is_number(lemma): 
        msd[1] = 'd'  # digit
    elif is_roman(lemma):
        msd[1] = 'r'  # roman
    else: 
        msd[1] = 'l'  # letter
    
    # Type (Position 2)
    if is_adj: 
        msd[2] = 'o'  # ordinal
    else: 
        msd[2] = 'c'  # cardinal
    
    # Gender (Position 3)
    gender_map = {'m': 'm', 'f': 'f', 'n': 'n'}
    for tag, gender in gender_map.items():
        if tag in tag_set:
            msd[3] = gender
            break
            
    # Number (Position 4)
    msd[4] = 's' if 's' in tag_set else 'p'
    
    # Case (Position 5)
    if 'nv' in tag_set:
        msd[5] = '-'
    else:
        case_map = {
            'v_naz': 'n', 'v_rod': 'g', 'v_dav': 'd', 
            'v_zna': 'a', 'v_oru': 'i', 'v_mis': 'l', 'v_kly': 'v'
        }
        for tag, case in case_map.items():
            if tag in tag_set:
                msd[5] = case
                break
    
    # Animacy (Position 6)
    if 'anim' in tag_set: 
        msd[6] = 'y'
    elif 'inanim' in tag_set: 
        msd[6] = 'n'

def _process_preposition(msd: List[str], tag_set: Set[str], lemma: str) -> None:
    """Process preposition-specific MSD positions."""
    # Type (Position 1)
    msd[1] = 'p'  # preposition
    
    # Formation (Position 2)
    if '-' in lemma: 
        msd[2] = 'c'  # compound
    else: 
        msd[2] = 's'  # simple

    # Case (Position 3) - Define preposition case government
    # This is a comprehensive mapping based on Ukrainian grammar
    case_government = {
        # Dative only
        'завдяки': 'd', 'всупереч': 'd', 'усупереч': 'd', 'наперекір': 'd', 
        'услід': 'd', 'назустріч': 'd', 'напротивагу': 'd',
        
        # Accusative only
        'во': 'a', 'ві': 'a', 'про': 'a', 'через': 'a', 'крізь': 'a', 
        'скрізь': 'a', 'об': 'a', 'поза': 'a', 'між': 'a', 
        'незважаючи': 'a',
        
        # Instrumental only
        'згідно з': 'i', 'надо': 'i', 'наді': 'i', 'передо': 'i', 
        'переді': 'i',
        
        # Locative only
        'при': 'l', 'вві': 'l', 'уві': 'l',
        
        # Multiple cases
        'в': 'agl', 'у': 'agl', 'ув': 'agl', 'попри': 'agl',
        'за': 'ai', 'над': 'ai', 'перед': 'ai', 'понад': 'ai', 
        'попід': 'ai', 'під': 'ai',
        'на': 'al', 'о': 'al', 'по': 'al',
        'з': 'gi', 'із': 'gi',
        'меж': 'agi', 'межи': 'agi', 'поміж': 'agi', 'поперед': 'agi',
        'повз': 'ag',
    }
    
    msd[3] = case_government.get(lemma, 'g')  # Default to genitive

def main():
    """Main function to process files."""
    parser = argparse.ArgumentParser(
        description='Convert Ukrainian tags to MULTEXT-East MSD format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_file", help='Input file path')
    parser.add_argument("output_file", help='Output file path')
    parser.add_argument("--encoding", default="utf-8", help='File encoding')
    
    args = parser.parse_args()
    
    try:
        with open(args.input_file, 'r', encoding=args.encoding) as infile, \
             open(args.output_file, 'w', encoding=args.encoding) as outfile:
            
            line_count = 0
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    outfile.write('\n')
                    continue
                
                parts = line.split(maxsplit=2)
                if len(parts) < 3:
                    print(f"Warning: Line {line_num} has insufficient parts: {line}", 
                          file=sys.stderr)
                    outfile.write(line + '\n')
                    continue
                
                lemma, word, tag_str = parts
                tags = tag_str.split(':')
                
                try:
                    msd_tag = tags_to_msd(tags, lemma)
                    outfile.write(f"{lemma}\t{word}\t{tag_str}\t{msd_tag}\n")
                    line_count += 1
                except Exception as e:
                    print(f"Error processing line {line_num}: {line}\nError: {e}", 
                          file=sys.stderr)
                    outfile.write(line + '\n')
                    
        print(f"Successfully processed {line_count} lines.", file=sys.stderr)
        
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except PermissionError:
        print(f"Error: Permission denied accessing files.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

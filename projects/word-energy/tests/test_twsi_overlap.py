from scripts.audit_twsi_wikitext_overlap import find_matches, tokens

def test_complete_subsequence_and_row_boundary():
    a=tokens('The CAT sat on the old mat')
    b=tokens('the cat sat on the red mat')
    index={a[:5]:{a,b}}
    assert find_matches(index,'Before: the cat SAT on the old mat! After.')=={a}
    assert find_matches(index,'the cat sat on the old')==set()
    assert find_matches(index,'mat')==set()
    assert find_matches(index,'the cat sat on the red mat')=={b}

from scripts.phase3p2_sar_diagnosis import options, support

def test_mcq_option_parser_returns_all_complete_candidates():
    parsed=options('Which? a) one, b) two, c) three, d) four')
    assert parsed == {'a':'one,','b':'two,','c':'three,','d':'four'}

def test_supervision_schema_is_conservative_and_closed():
    label,_=support({'task_type':'caption','caption':'captured during winter in country X'})
    assert label in {'SAR_SUPPORTED','SAR_PARTIALLY_SUPPORTED','SAR_NOT_ESTABLISHED'}

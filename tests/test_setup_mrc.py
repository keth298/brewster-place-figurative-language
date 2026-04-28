from setup_mrc import parse_mrc


def test_parse_mrc_extracts_concreteness(tmp_path):
    # MRC line format: space-separated numeric fields then |WORD|PRONUNCIATION
    # Field index 9 (0-based) is CNC (concreteness). 0 means no rating → skip.
    lines = (
        " 4  4 1  15  2  1  35   6 524 413 488 500 560 532 N N N N F N N N|ABLE|1EYB.AH\n"
        " 4  4 1 3022  7 30 2327 688 593 617 624 570 578 568 N N N N F N N N|BACK|BAK\n"
        " 3  3 1  32  3  2  18   4 600   0 582 480 540 420 N N N N F N N N|THE|DHAH\n"
    )
    mrc_file = tmp_path / "mrc2.dct"
    mrc_file.write_text(lines, encoding="latin-1")
    result = parse_mrc(str(mrc_file))
    assert result["able"] == 413
    assert result["back"] == 617
    assert "the" not in result  # CNC=0 means no rating, skip


def test_parse_mrc_lowercases_words(tmp_path):
    line = " 4  4 1  15  2  1  35   6 524 413 488 500 560 532 N N N N F N N N|ABLE|1EYB.AH\n"
    mrc_file = tmp_path / "mrc2.dct"
    mrc_file.write_text(line, encoding="latin-1")
    result = parse_mrc(str(mrc_file))
    assert "able" in result
    assert "ABLE" not in result


def test_parse_mrc_skips_malformed_lines(tmp_path):
    lines = "this is not a valid line\n"
    mrc_file = tmp_path / "mrc2.dct"
    mrc_file.write_text(lines, encoding="latin-1")
    result = parse_mrc(str(mrc_file))
    assert result == {}

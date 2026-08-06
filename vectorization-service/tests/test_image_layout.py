from app.ingest.image_layout import analyze_layout


def _detection(left: int, top: int, text: str, *, width: int = 80, height: int = 20):
    return ([[left, top], [left + width, top], [left + width, top + height], [left, top + height]], text, 0.99)


def test_repeated_aligned_two_column_rows_are_structured():
    rows = [
        [_detection(50, row * 40, f"left-{row}"), _detection(700, row * 40, f"right-{row}")]
        for row in range(6)
    ]

    analysis = analyze_layout(rows, median_height=20)

    assert analysis.repeated_pair_row_ratio == 1.0
    assert analysis.structured_row_ratio == 1.0
    assert analysis.reason == "repeated_two_column_layout"


def test_isolated_right_aligned_metadata_does_not_trigger_pair_layout():
    rows = [
        [_detection(50, 0, "Title"), _detection(700, 0, "Status")],
        [_detection(50, 40, "ordinary prose", width=500)],
        [_detection(50, 80, "another prose line", width=500)],
        [_detection(50, 120, "Footer"), _detection(700, 120, "Page 1")],
    ]

    analysis = analyze_layout(rows, median_height=20)

    assert analysis.repeated_pair_row_ratio == 0.0
    assert analysis.structured_row_ratio == 0.0
    assert analysis.reason is None


def test_unaligned_pairs_do_not_look_like_a_repeated_form():
    rows = [
        [_detection(50, 0, "a"), _detection(300, 0, "b")],
        [_detection(50, 40, "c"), _detection(500, 40, "d")],
        [_detection(50, 80, "e"), _detection(700, 80, "f")],
        [_detection(50, 120, "g"), _detection(900, 120, "h")],
    ]

    analysis = analyze_layout(rows, median_height=20)

    assert analysis.repeated_pair_row_ratio == 0.0


def test_three_column_table_remains_structured():
    rows = [
        [
            _detection(50, row * 40, f"a-{row}"),
            _detection(400, row * 40, f"b-{row}"),
            _detection(750, row * 40, f"c-{row}"),
        ]
        for row in range(4)
    ]

    analysis = analyze_layout(rows, median_height=20)

    assert analysis.multi_column_row_ratio == 1.0
    assert analysis.structured_row_ratio == 1.0
    assert analysis.reason == "multi_column_layout"


def test_two_column_signal_is_resolution_independent():
    ratios = []
    for scale in (1, 8):
        rows = [
            [
                _detection(50 * scale, row * 40 * scale, f"left-{row}", width=80 * scale, height=20 * scale),
                _detection(700 * scale, row * 40 * scale, f"right-{row}", width=80 * scale, height=20 * scale),
            ]
            for row in range(5)
        ]
        ratios.append(analyze_layout(rows, median_height=20 * scale).structured_row_ratio)

    assert ratios == [1.0, 1.0]

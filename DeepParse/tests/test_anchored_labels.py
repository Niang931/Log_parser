import re

from deepparse.tools.build_training_set import _align_template_with_context, _classify_value


def _labels(line, template):
    slots = _align_template_with_context(line, template)
    out = []
    for idx, (value, before, after) in enumerate(slots):
        regex = _classify_value(value, before, after, line=line, at_line_start=idx == 0)
        out.append(regex)
    return out


def test_shapeless_values_get_anchored_regexes_that_hit_the_gold_span():
    line = "proxy.cse.cuhk.edu.hk:5070 open through proxy gw.example.org:443 HTTPS"
    labels = _labels(line, "<*>:<*> open through proxy <*>:<*> HTTPS")
    assert all(labels)
    assert re.search(labels[0], line).group(0) == "proxy.cse.cuhk.edu.hk"
    assert re.search(labels[2], line).group(0) == "gw.example.org"


def test_anchor_does_not_run_on_from_an_earlier_delimiter():
    line = "x close, 451 bytes sent, 18846 bytes (18.4 KB) received, lifetime 00:01"
    regex = _classify_value(
        "18846 bytes (18.4 KB) received", ", ", ", lifetime ", line=line
    )
    assert re.search(regex, line).group(0) == "18846 bytes (18.4 KB) received"


def test_whitespace_only_left_context_gets_no_standalone_label():
    assert _classify_value("sent", " ", ",", line="451 bytes sent,") is None

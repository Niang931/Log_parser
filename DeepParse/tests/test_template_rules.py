from deepparse.metrics.parsing_accuracy import parsing_accuracy
from deepparse.metrics.template_rules import normalise_template


def test_rules():
    assert normalise_template("data_thread() got answer  ") == "data_thread() got answer"  # DS
    assert normalise_template("open /var/log/messages failed") == "open <*> failed"  # PS
    assert normalise_template("session closed for user root") == "session closed for user <*>"  # US
    assert normalise_template("retry 3 enabled=true") == "retry <*> enabled=<*>"  # DG, BL
    assert normalise_template("block blk_<*> served") == "block <*> served"  # WV
    assert normalise_template("from <*>.<*>.<*>") == "from <*>"  # DV
    assert normalise_template("took <*><*> ms") == "took <*> ms"  # CV


def test_pa_normalises_both_sides_unless_disabled():
    gold, pred = ["user <*> logged in "], ["user root logged in"]
    assert parsing_accuracy(gold, pred) == 1.0
    assert parsing_accuracy(gold, pred, normalise=False) == 0.0

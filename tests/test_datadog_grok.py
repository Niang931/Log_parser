from logpipe.datadog_grok import date_pattern, pipeline, to_dd_rule
from logpipe.showcase import PG_RULE


def test_literals_are_escaped_and_whitespace_is_any_run():
    rule = "[%{integer:pid}]  GET %{path:url} took %{number:ms}ms from %{notspace:ip} %{quoted:q}"
    assert to_dd_rule(rule)[0] == (
        r"\[%{integer:pid}\]\s+GET\s+%{notSpace:url}\s+took\s+%{number:ms}ms\s+from\s+"
        r"%{notSpace:ip}\s+%{quotedString:q}"
    )
    assert to_dd_rule('%{regex("[a-z]+"):w} x.y')[0] == r'%{regex("[a-z]+"):w}\s+x\.y'


def test_date_formats_come_from_the_sample():
    assert date_pattern("2026-11-13 10:00:00.123") == '"yyyy-MM-dd HH:mm:ss.SSS"'
    assert date_pattern("2026-11-13T10:00:00Z") == "\"yyyy-MM-dd'T'HH:mm:ss'Z'\", \"UTC\""
    assert date_pattern("2026-11-13T10:00:00,5+01:00") == "\"yyyy-MM-dd'T'HH:mm:ss,SZZ\""
    assert date_pattern("2026-11-13T10:00:00-0500") == "\"yyyy-MM-dd'T'HH:mm:ssZ\""
    assert date_pattern("13/Nov/2026") is None

    text, dates = to_dd_rule(PG_RULE, {"ts": "2026-11-13 10:00:00.123"})
    assert text.startswith('%{date("yyyy-MM-dd HH:mm:ss.SSS"):ts}\\s+UTC\\s+\\[')
    assert dates == ["ts"]
    text, dates = to_dd_rule(PG_RULE)  # no sample: still matches, as text
    assert text.startswith('%{regex("') and dates == []


def test_filters_need_the_name_and_a_sample_that_fits():
    sample = {
        "user_agent": '"Mozilla/5.0 (X11; Linux x86_64)"',
        "url": "https://shop.example/cart?id=1",
        "params": "a=1&b=2",
        "body": '{"order": 7}',
        "attrs": "user=7 cart=3",
        "agent": "none",  # a name that fits, a value that doesn't
    }
    rule = " ".join(f"{k}=%{{data:{k}}}" for k in sample)
    text, _ = to_dd_rule(rule, sample)
    for part in (
        "%{data:user_agent:useragent}",
        "%{data:url:url}",
        "%{data:params:querystring}",
        "%{data:body:json}",
        "%{data:attrs:keyvalue}",
        "%{data:agent}",
    ):
        assert part in text
    assert to_dd_rule("ua %{data:ua}")[0] == r"ua\s+%{data:ua}"  # no sample, no filter


def test_pipeline_is_ordered_scoped_and_remaps_dates():
    body = pipeline(
        [
            {"template_id": 1, "rule": "%{data:msg} done", "service": "shop-api"},
            {
                "template_id": 2,
                "rule": PG_RULE,
                "sample": {"ts": "2026-11-13 10:00:00.123"},
                "service": "postgres",
            },
            {"template_id": 3, "rule": "", "service": "ignored"},
            {"template_id": 4, "rule": "x %{word:w}", "service": "my svc"},
        ]
    )
    rules = body["processors"][0]["grok"]["match_rules"].splitlines()
    assert [r.split()[0] for r in rules] == ["t2", "t1", "t4"]  # most fixed text first
    assert body["filter"]["query"] == 'service:("my svc" OR postgres OR shop-api)'
    assert body["processors"][1] == {
        "type": "date-remapper",
        "name": "logpipe timestamp",
        "is_enabled": True,
        "sources": ["ts"],
    }
    assert pipeline([{"template_id": 1, "rule": "a %{word:w}"}])["filter"]["query"] == "*"
    assert len(pipeline([{"template_id": 1, "rule": "a %{word:w}"}])["processors"]) == 1

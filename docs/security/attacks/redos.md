# Regular Expression Denial of Service (ReDoS)

ReDoS occurs when an attacker supplies input that causes a non-deterministic regular expression engine to perform an excessive amount of work. Backtracking engines can take exponentially longer on carefully constructed near-matches.

The purpose is to consume CPU resources, tie up application workers, delay requests, or make a service unavailable.

## Nested quantifiers

Nested repetition can cause the regex engine to explore many alternative ways to match the same input.

```regex
(a+)+$
(.*)*
([a-zA-Z]+)+$
```

Example near-match input:

```text
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa!
```

## Overlapping alternation

Alternatives that share prefixes can cause extensive backtracking when repeated.

```regex
(a|aa)+$
(a|a?)+$
(ab|a)+$
```

Example near-match input:

```text
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaac
```

## Ambiguous wildcard patterns

Repeated wildcards and optional groups can create many possible matching paths.

```regex
(.*a){10}$
(.+)*END$
(a.*a)+$
```

## Oversized patterns and inputs

Even without catastrophic backtracking, extremely large patterns and log lines can consume excessive memory and CPU.

```text
Pattern length: 100000 characters
Log line length: 10000000 characters
```

## ReDoS defenses

- Prefer a restricted regex grammar or a linear-time regex engine.
- Use a regex engine with execution timeouts where possible.
- Do not rely exclusively on blacklisting suspicious regex fragments.
- Enforce maximum pattern and input lengths.
- Limit the number of patterns evaluated per request.
- Run untrusted regex matching in a process that can be terminated.

## Source
https://www.imperva.com/learn/ddos/regular-expression-denial-of-service-redos/

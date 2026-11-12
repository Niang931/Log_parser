# Unsafe Key and Field-Name Injection

Key injection occurs when untrusted input or LLM output supplies field names that are malformed, unexpected, duplicated, or meaningful to downstream systems.

The purpose may be to confuse parsing, overwrite values, bypass schema assumptions, trigger exceptions, or exploit unsafe consumers.

A syntactically valid key is not necessarily an authorized key.

## __proto__ and special property names

These names are especially important when data crosses into JavaScript or object-oriented consumers.

```json
{
  "keys": [
    "__proto__",
    "constructor",
    "prototype"
  ]
}
```

Note: Python dictionaries do not have JavaScript prototype-pollution behavior by default. The risk depends on downstream conversion and usage.

## Duplicate keys

Duplicate names can cause silent overwrites or inconsistent interpretation between parsers.

```json
{
  "keys": [
    "password",
    "password",
    "password"
  ]
}
```

## Invalid or malformed names

Names containing spaces, punctuation, control characters, or unexpected types can break assumptions in downstream code.

```json
{
  "keys": [
    "x y z",
    "",
    "user-name",
    "user.name",
    "name\nrole"
  ]
}
```

## Type-confusion payloads

The validator should handle values that are not strings without crashing.

```json
{
  "keys": [
    null,
    123,
    true,
    [],
    {},
    ["nested"]
  ]
}
```

## Unexpected but syntactically valid fields

A key may pass a regular expression but still be outside the application's permitted schema.

```json
{
  "keys": [
    "admin",
    "is_authenticated",
    "role",
    "tenant_id"
  ]
}
```

## Key normalization collisions

Different spellings may normalize to the same field name.

```text
userName
user_name
user-name
user.name
```

For example, converting punctuation and camel case to underscores may cause different inputs to become the same normalized key.

## Key-validation defenses

- Require a list or tuple of strings.
- Use full-string validation rather than prefix matching.
- Enforce uniqueness after normalization.
- Enforce one key per extracted slot.
- Apply a maximum key length.
- Use an explicit schema allowlist for sensitive fields.
- Define a duplicate-key policy.
- Safely serialize data when passing it between Python, JavaScript, databases, and templates.

## Source
https://cwe.mitre.org/data/definitions/1321.html
https://owasp.org/www-project-top-ten/
https://json-schema.org/learn/getting-started-step-by-step

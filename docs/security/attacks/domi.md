# DOM Injection: HTML and JavaScript Injection

HTML and JavaScript injection occurs when untrusted content is interpreted as markup or executable code.

DOM-based XSS is a client-side form of this problem: JavaScript takes untrusted data and passes it to a dangerous DOM or URL-handling sink.

## HTML element injection

Tests if a string is interpreted as HTML instead of displayed literally.

```html
<h1>LOGPIPE_TEST_MARKER</h1>
<script>alert(1)</script>
```

Expected behavior: tags appear as text or are safely sanitized; no script executes.

## HTML event-handler injection

Tests whether an attacker can introduce an event handler through an HTML element.

```html
<img src=x onerror=alert(1)>
<svg onload=alert(1)>
```

Expected behavior: event handlers do not execute.

## Attribute breakout

Tests whether untrusted data inserted into an HTML attribute can escape its quoted value and add another attribute.

```text
" autofocus onfocus="alert(1)
"><img src=x onerror=alert(1)>
```

Expected behavior: the value remains inside its intended text or attribute context.

## JavaScript URL injection

Tests whether untrusted links can use an executable URL scheme.

```html
<a href="javascript:alert(1)">LOGPIPE_TEST_LINK</a>
```

Expected behavior: reject unsafe URL schemes. Allow only explicitly supported schemes, typically HTTPS and, where required, HTTP.

## Iframe and embedded-content injection

Tests whether untrusted content can embed another document.

```html
<iframe src="https://example.invalid"></iframe>
<iframe src="data:text/html,LOGPIPE_TEST_MARKER"></iframe>
```

Expected behavior: untrusted content cannot create active embedded documents.

## CSS and style-context breakout

Tests whether content inserted into a style block or style attribute can escape its intended context.

```html
</style><script>alert(1)</script>
```

Expected behavior: do not concatenate untrusted values into CSS or style blocks.

## DOM sink injection

The same string can be safe or unsafe depending on the browser API that receives it.

```javascript
// Safe for plain text:
element.textContent = untrustedValue;

// Dangerous with untrusted HTML:
element.innerHTML = untrustedValue;
```

Review uses of:

- `innerHTML`
- `outerHTML`
- `insertAdjacentHTML`
- `document.write`
- Dynamic script creation
- Dynamic URL construction

Prefer `textContent`, safe DOM creation APIs, and context-aware sanitization when rich HTML is genuinely required.

## Markdown and link-renderer injection

Markdown renderers may allow raw HTML or dangerous link schemes unless configured safely.

```markdown
[LOGPIPE_TEST_LINK](javascript:alert(1))

<img src=x onerror=alert(1)>
```

Expected behavior: disable raw HTML if it is not needed, sanitize rendered HTML, and validate link protocols.

## Encoded and double-decoded content

Tests inconsistent decoding or sanitization order.

```text
&lt;script&gt;alert(1)&lt;/script&gt;
%3Cscript%3Ealert(1)%3C/script%3E
```

Expected behavior: content remains inert through decoding, storage, API serialization, and display.

## Template-expression injection

If content is passed into a template engine as template source rather than as a value, expressions may be evaluated.

```text
{{7*7}}

${7*7}

<%= 7*7 %>
```

Expected behavior: treat user-controlled content as data, never compile it as a template.

## DOM/XSS Defenses

- Render logs and explanations with `textContent` or the framework's normal escaped text interpolation.
- Avoid unsafe HTML sinks for untrusted content.
- If rich HTML is required, use a maintained sanitizer with a narrow allowlist.
- Validate URL schemes before creating links.
- Configure Markdown rendering to disable raw HTML unless explicitly needed.
- Apply a restrictive Content Security Policy as defense in depth.
- Test the actual frontend rendering path, not just the JSON API response.

## Source
https://github.com/swisskyrepo/PayloadsAllTheThings/tree/master/XSS%20Injection
https://owasp.org/www-community/attacks/xss/
https://owasp.org/www-community/attacks/DOM_Based_XSS

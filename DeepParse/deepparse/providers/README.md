# Adding a provider

The pipeline never imports a vendor. It asks `registry.get_provider(name)` for something
satisfying `base.ChatProvider` and uses four things from it, which is the whole of the
coupling:

| What | Why the pipeline needs it |
|---|---|
| `build(...)` | A LangChain chat model with the credential, model and effort applied. |
| `structured(model, schema)` | A runnable that returns a validated object plus the raw message. **How** — native structured output, a forced tool call, a JSON-mode parser — is yours. |
| `price(model)` | So a job can be costed and the monthly budget enforced without the caller knowing the vendor's units. Returning `None` is legal and means "this build does not know", which under-counts rather than refusing to run. |
| `default_model` | What a credential gets when the author has not chosen one. |

## Three steps

**1. Write the adapter.** Copy `anthropic.py`. The only parts that are genuinely
vendor-specific are the constructor arguments in `build` and the `structured` method.

```python
class MyProvider:
    name = "myvendor"

    @property
    def default_model(self) -> str:
        return "their-best-model"

    def build(self, *, api_key, model, max_tokens, effort, timeout, max_retries, seed=None):
        # seed: pass it on if the vendor takes one, otherwise ignore it
        return ChatMyVendor(model=model, api_key=api_key, max_tokens=max_tokens, ...)

    def structured(self, model, schema):
        # Must return LangChain's include_raw shape:
        #   {"raw": AIMessage, "parsed": schema | None, "parsing_error": Exception | None}
        # The raw message is not optional — it carries usage_metadata, and a pipeline that
        # cannot cost its own calls cannot be budgeted.
        return model.with_structured_output(schema, include_raw=True)

    def price(self, model):
        return _PRICES.get(model)
```

If the vendor has no effort concept, ignore the argument. If it has no structured output at
all, bind a `PydanticOutputParser` and a repair retry — the pipeline only cares that an
instance of `schema` comes back.

**2. Register it.** One line in `registry.py`:

```python
providers: list[ChatProvider] = [AnthropicProvider(), MyProvider()]
```

**3. Teach DeepParse about it.** Two one-line edits:

- add the environment variable(s) its key is read from to `API_KEY_ENV` in
  `deepparse/synth/chat_provider.py`;
- add its name to `LLM_PROVIDERS` in `deepparse/cli.py`, so `--provider` accepts it.

The vendor's LangChain package goes in the `llm` extra in `pyproject.toml`.

`web.py` is the exception to all three steps: it is not registered, needs no key, and is
built by name in `chat_provider.py` only when `--provider web` asks for it.

## What not to do

**Do not reach for a vendor inside `deepparse/synth/`.** If something cannot be expressed
through `ChatProvider`, widen the protocol — that is a decision worth making once, in one
place, rather than a special case that quietly makes one vendor the real one.

**Do not use an OpenAI-compatible shim to cover several vendors at once.** It works until
it does not: structured output, thinking, and usage reporting are exactly the three things
such shims approximate, and all three are load-bearing here.

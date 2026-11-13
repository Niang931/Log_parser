"""Every setting logpipe reads from the environment (see ``.env.example``), and the numbers an
operator may want to tune, in one place.

Each group reads its own environment prefix: ``CLICKHOUSE_HOST`` fills ``ClickHouseSettings.host``,
``LOGPIPE_RCA_BEFORE_MINUTES`` fills ``RcaSettings.before_minutes``. Nothing else in logpipe reads
``os.environ`` for configuration; it asks :func:`get_settings`.

Fixed tables (level regexes, prompts, column maps, allowed stages) stay next to the code that uses
them: they are part of the code's contract, not something to tune.

``RuntimeSettings`` is different: the settings a person changes in the UI, stored in the control
store and validated here.
"""

from __future__ import annotations

from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _env(prefix: str = "") -> SettingsConfigDict:
    return SettingsConfigDict(env_prefix=prefix, extra="ignore", frozen=True)


class AppSettings(BaseSettings):
    """Where the CLI pipeline reads and writes, and which store and queue it uses."""

    model_config = _env()
    logs_path: Path = Path("logs")
    registry_path: Path = Path("artifacts/registry.json")
    model_path: Path = Path("artifacts/logbert")
    store: Literal["memory", "clickhouse", "mongodb"] = "memory"
    queue: Literal["memory", "kafka"] = "memory"


class LLMSettings(BaseSettings):
    model_config = _env()
    llm_backend: str = "offline"  # offline | llm:<provider> | llm:web:<site> | hf
    web_browser: str = "camoufox"  # for llm:web:<site>
    hf_base: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    hf_adapter: str = "MThien213/deepparse-iot-r1-8b"

    @field_validator("web_browser")
    @classmethod
    def _browser(cls, v: str) -> str:
        return v.replace("chromium", "chrome")


class ClickHouseSettings(BaseSettings):
    model_config = _env("CLICKHOUSE_")
    host: str = "127.0.0.1"
    port: int = 8123
    user: str = "default"
    password: str = ""
    db: str = "default"
    # Free-form and Ask queries run as this user when set (docker/clickhouse-init.sh creates it).
    readonly_user: str | None = None
    readonly_password: str = ""


class MongoSettings(BaseSettings):
    model_config = _env("MONGO_")
    uri: str = "mongodb://127.0.0.1:27017"
    db: str = "logpipe"
    readonly_uri: str | None = None  # free-form pipelines run with this connection when set


class KafkaSettings(BaseSettings):
    model_config = _env("KAFKA_")
    bootstrap: str = "127.0.0.1:9092"
    topic: str = "logpipe.unresolved"
    group: str = "logpipe-workers"
    dlq_topic: str | None = None
    key_by: Literal["shape", "device"] = "shape"


class ServerSettings(BaseSettings):
    model_config = _env("LOGPIPE_")
    log_format: Literal["text", "json"] = "text"
    bootstrap_key: str | None = None
    admin_password: str | None = None
    session_days: float = 7
    min_password_length: int = 8
    login_max_failures: int = 5  # per client address, within the window below
    login_failure_window_minutes: float = 15
    max_body_mb: int = 64  # a decompressed intake batch
    online_seconds: int = 45  # agents that don't hold /intake/v1/wait open (older builds)
    gone_seconds: int = 1  # an agent whose wait connection closed and didn't come back is offline
    min_train_windows: int = 8
    min_train_steps: int = 1500
    min_draft_lines: int = 50  # below this a sample can't say much about a source's templates
    incident_gap_minutes: float = 5  # alerts on one service closer than this are one incident
    max_query_rows: int = 500  # rows the Query page returns
    max_builder_rows: int = 500  # LIMIT the visual builder may ask for
    review_samples: int = 5  # example lines shown per draft template
    review_check_lines: int = 20_000  # stored lines a draft rule is checked against
    drift_keep: int = 2000  # live windows kept per stream
    drift_min_live: int = 200  # fewer live windows than this: no PSI yet
    metrics_latency_cap: int = 20_000  # latencies kept per sample (a uniform reservoir beyond that)

    @property
    def session_seconds(self) -> int:
        return int(self.session_days * 86_400)

    @property
    def incident_gap(self) -> timedelta:
        return timedelta(minutes=self.incident_gap_minutes)


class AskSettings(BaseSettings):
    """The Ask box: limits on a query spec, plans, and the template index."""

    model_config = _env("LOGPIPE_ASK_")
    model: str = "sentence-transformers/all-MiniLM-L6-v2"  # text -> template embeddings
    plan_ttl_minutes: float = 15  # a plan waiting for approval expires after this
    max_rounds: int = 7  # discovery, evidence, one widening, up to four follow-ups
    default_minutes: int = 60
    max_span_days: int = 90
    max_keywords: int = 5
    max_rows: int = 500  # lines a keyword search shows
    row_cap: int = 2000  # lines one read may return
    agg_cap: int = 20_000  # rows of an aggregate (template x bucket)
    budget_tokens: int = 6000  # evidence bundle sent to the LLM
    chars_per_token: int = 4
    shutdown_min_score: float = 0.45  # how close a template must be to a shutdown phrase
    # one LLM call may take this long (a web chat retries a slow reply twice at 60 s each)
    llm_timeout_seconds: float = 240

    @property
    def plan_ttl(self) -> timedelta:
        return timedelta(minutes=self.plan_ttl_minutes)

    @property
    def max_span(self) -> timedelta:
        return timedelta(days=self.max_span_days)


class RcaSettings(BaseSettings):
    """Root-cause analysis: the window around the anchor and the ranking (see ``logpipe.ask.rca``)."""

    model_config = _env("LOGPIPE_RCA_")
    before_minutes: float = 30
    after_minutes: float = 2
    wide_before_minutes: float = 120
    wide_after_minutes: float = 5
    baseline_hours: float = 24
    silence_min_rate: float = 2.0  # lines/minute a service must have kept up before going quiet
    silence_minutes: int = 3
    error_weight: float = 0.8
    warn_weight: float = 0.5
    other_weight: float = 0.1
    shutdown_weight: float = 0.9
    # a LogBERT neighbour of the anchor, or a template matching the question
    related_weight: float = 0.5
    proximity_seconds: float = 600  # proximity = exp(-|first seen - anchor| / this)
    precursor_seconds: float = 120
    strong: float = 0.15  # a candidate that stands out: a new warning near the anchor scores 0.5
    min_strong: int = 3  # fewer strong candidates than this widens the window once
    max_candidates: int = 15
    max_lines: int = 40
    line_chars: int = 300
    follow_ups: int = 4  # rounds of reads the LLM may ask for, each approved like the rest
    moves_per_round: int = 3
    move_max_hours: float = 24  # how far before the anchor a follow-up read may look
    follow_up_lines: int = 20  # lines one follow-up round may add to the bundle
    # value hints: a host or extracted value carried by error lines far more often than by the others
    hint_min_ratio: float = 3.0  # share among errors / share among the other lines
    hint_min_share: float = 0.2  # of the error lines
    hint_min_lines: int = 3
    hint_min_others: int = 10  # fewer other lines than this: nothing to contrast with
    max_hints: int = 5

    @property
    def before(self) -> timedelta:
        return timedelta(minutes=self.before_minutes)

    @property
    def after(self) -> timedelta:
        return timedelta(minutes=self.after_minutes)

    @property
    def wide_before(self) -> timedelta:
        return timedelta(minutes=self.wide_before_minutes)

    @property
    def wide_after(self) -> timedelta:
        return timedelta(minutes=self.wide_after_minutes)

    @property
    def baseline(self) -> timedelta:
        return timedelta(hours=self.baseline_hours)

    def level_weight(self, level: str | None) -> float:
        return {"error": self.error_weight, "warn": self.warn_weight}.get(
            level or "", self.other_weight
        )


class DatadogSettings(BaseSettings):
    """Datadog: self-monitoring through the Agent (``--datadog``) and the REST loop with a customer's
    org (``--datadog-pull``, events, pipelines, webhook). Same ``DD_*`` names the Agent uses."""

    model_config = _env("DD_")
    service: str = "logpipe"
    site: str = "datadoghq.com"
    agent_host: str = "127.0.0.1"
    dogstatsd_port: int = 8125
    log_file: Path = Path(".logpipe/dd-logs.jsonl")
    llmobs_ml_app: str = "logpipe"
    api_key: str | None = None
    app_key: str | None = None
    pull_query: str = ""  # Logs search query, e.g. "service:shop*"; empty means no pulling
    pull_every_s: float = 15
    public_url: str = ""  # where Datadog event links point, e.g. the tunnel URL of `logpipe server`

    @property
    def api_url(self) -> str:
        return f"https://api.{self.site}"

    @property
    def app_url(self) -> str:
        # datadoghq.eu -> app.datadoghq.eu, us5.datadoghq.com -> us5.datadoghq.com
        return f"https://{'app.' if self.site.count('.') == 1 else ''}{self.site}"


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)
    app: AppSettings = Field(default_factory=AppSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    clickhouse: ClickHouseSettings = Field(default_factory=ClickHouseSettings)
    mongo: MongoSettings = Field(default_factory=MongoSettings)
    kafka: KafkaSettings = Field(default_factory=KafkaSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    ask: AskSettings = Field(default_factory=AskSettings)
    rca: RcaSettings = Field(default_factory=RcaSettings)
    datadog: DatadogSettings = Field(default_factory=DatadogSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The settings, read from the environment once. Tests that change the environment call
    ``get_settings.cache_clear()`` (``tests/conftest.py`` does it before every test)."""
    return Settings()


class RuntimeSettings(BaseModel):
    """What a person changes in the UI (Settings page), kept in the control store."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)
    sample_cap: int = Field(5000, ge=100, le=200_000)
    retrain_days: float = Field(7, ge=0.01, le=3650)
    draft_lines: int = Field(500, ge=50, le=200_000)
    draft_after_minutes: float = Field(60, ge=1, le=100_000)
    ask_require_approval: bool = True  # every Ask query waits for a person (logpipe.ask.service)
    calibrated: bool = False
    calibrated_at: str | None = None
    setup_done: bool = False

    # What PUT /api/settings may change; the rest is set by the server as setup progresses.
    EDITABLE: ClassVar[tuple[str, ...]] = (
        "sample_cap",
        "retrain_days",
        "draft_lines",
        "draft_after_minutes",
        "ask_require_approval",
    )

    def changed(self, body: dict) -> RuntimeSettings:
        """A copy with the editable fields in ``body`` applied; ValueError naming the bad one."""
        from pydantic import ValidationError

        fields = {k: body[k] for k in self.EDITABLE if k in body}
        if "ask_require_approval" in fields and not isinstance(
            fields["ask_require_approval"], bool
        ):
            raise ValueError("ask_require_approval is true or false")
        try:
            return self.model_validate(self.model_dump() | fields)
        except ValidationError as error:
            first = error.errors()[0]
            raise ValueError(f"{first['loc'][0]}: {first['msg']}") from None

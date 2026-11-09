"""A small web shop's logs, for the showcase: nginx access log, a Java API (with multiline stack
traces), Postgres, and later a new inventory service.

Normal traffic repeats one order flow (cart -> checkout -> payment -> order), so each service's
template sequence is regular and LogBERT learns it. (Rare variants inside normal traffic would be
flagged too: with a 20-line window a single unexpected line moves the miss ratio a lot.)
:meth:`ShopGenerator.incident` switches to a payment-gateway outage: retries piling up, timeouts,
failed orders with Java stack traces and an exhausted database pool, which breaks that sequence.
"""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# service -> (file under the output dir, agent source, multi_line start pattern or None)
FILES = {
    "nginx": ("nginx/access.log", "nginx", None),
    "shop-api": ("api/app.log", "java", r"^\d{4}-\d{2}-\d{2} "),
    "postgres": ("db/postgres.log", "postgresql", r"^\d{4}-\d{2}-\d{2} "),
    "inventory": ("inventory/inventory.log", "go", None),
}
AGENTS = ("Mozilla/5.0 (X11; Linux x86_64)", "Mozilla/5.0 (Macintosh)", "curl/8.5.0", "okhttp/4.12")


# mode -> root-cause lines (service, level, text) at the fault's start, the symptom lines each order
# logs after it, services that stop logging, and `truth`: text only the root-cause line contains.
FAULTS: dict[str, dict] = {
    "sigterm": {"root": [("pg", "", "LOG:  received fast shutdown request"),
                         ("pg", "", "LOG:  aborting any active transactions")], "silent": ("pg",),
                "symptoms": [("api", "ERROR", "could not get JDBC connection: Connection to postgres:5432 refused after {ms} ms")],
                "truth": "received fast shutdown request"},
    "oom": {"root": [("api", "ERROR", "java.lang.OutOfMemoryError: Java heap space")], "logger": "c.s.cart.CartController",
            "silent": ("api",), "gateway": True, "truth": "OutOfMemoryError"},
    "segfault": {"root": [("pg", "", "LOG:  server process (PID {pid}) was terminated by signal 11: Segmentation fault"),
                          ("pg", "", "LOG:  terminating any other active server processes")],
                 "symptoms": [("api", "ERROR", "could not get JDBC connection: the database system is in recovery mode")],
                 "truth": "Segmentation fault"},
    "disk_full": {"root": [("pg", "", "PANIC:  could not write to file \"pg_wal/xlogtemp.{pid}\": No space left on device")],
                  "symptoms": [("api", "ERROR", "could not get JDBC connection: the database system is shutting down")],
                  "silent": ("pg",), "truth": "No space left on device"},
    "db_port_blocked": {"root": [("api", "ERROR", "Connection to db.internal:5432 refused. Check that the hostname and port are correct")],
                        "logger": "c.s.db.ConnectionPool", "repeat": 7,
                        "symptoms": [("api", "WARN", "retrying database call, attempt {n}")],
                        "truth": "Connection to db.internal:5432 refused"},
    "pool_exhausted": {"root": [("api", "WARN", "HikariPool-1 - Connection is not available, request timed out after 30000ms")],
                       "logger": "com.zaxxer.hikari.pool.HikariPool", "repeat": 3,
                       "symptoms": [("api", "ERROR", "transaction rolled back after {ms} ms")],
                       "truth": "Connection is not available"},
    "gateway_timeout": {"root": [("api", "ERROR", "payment pay_{pid} declined: gateway timeout after 30000 ms")],
                        "logger": "c.s.payment.PaymentClient", "repeat": 2,
                        "symptoms": [("api", "WARN", "payment slow response {ms} ms, retrying")],
                        "truth": "gateway timeout after 30000 ms"},
    "dns_failure": {"root": [("api", "ERROR", "java.net.UnknownHostException: payments.internal: Name or service not known")],
                    "logger": "c.s.payment.PaymentClient", "repeat": 5, "truth": "UnknownHostException"},
    "tls_expired": {"root": [("api", "ERROR", ("javax.net.ssl.SSLHandshakeException: PKIX path validation failed: "
                                              "validity check failed: certificate expired on 2026-11-01"))],
                    "logger": "c.s.payment.PaymentClient", "repeat": 5, "truth": "certificate expired"},
    "config_reload": {"root": [("nginx", "", "[emerg] {pid}#0: unknown directive \"proxy_passs\" in /etc/nginx/conf.d/shop.conf:14")],
                      "silent": ("nginx", "api"), "truth": "unknown directive"},
    "dependency_503": {"root": [("api", "ERROR", "payment provider returned HTTP 503 Service Unavailable")],
                       "logger": "c.s.payment.PaymentClient", "repeat": 3, "truth": "returned HTTP 503"},
    "gc_thrash": {"root": [("api", "WARN", "GC overhead: 92% of time spent in GC over the last 60 s (pause {n}.4 s)")],
                  "logger": "jvm.gc", "repeat": 4, "symptoms": [("api", "WARN", "request took {ms} ms, over the 1000 ms budget")],
                  "truth": "GC overhead"},
    "lock_contention": {"root": [("pg", "", "LOG:  process {pid} still waiting for ShareLock on transaction 9{n} after 1000.{n} ms")],
                        "repeat": 3, "symptoms": [("api", "ERROR", "statement timeout after {ms} ms on table orders")],
                        "truth": "still waiting for ShareLock"},
    "clock_skew": {"root": [("api", "ERROR", "JWT validation failed: token issued in the future, clock skew {n}12 s")],
                   "logger": "c.s.auth.TokenFilter", "repeat": 4, "truth": "clock skew"},
    "api_hang": {"root": [], "silent": ("api",), "gateway": True, "truth": "silence:shop-api"},
    "control_quiet": {"root": [], "normal": True, "truth": None},
    "control_busy": {"root": [("api", "INFO", "cache warmed: {n} entries")], "logger": "c.s.cache.Cache", "repeat": 9,
                     "normal": True, "truth": None},
}


@dataclass
class Event:
    service: str
    message: str  # may span lines (stack traces)


class ShopGenerator:
    def __init__(self, out: Path, rate: float = 3.0, seed: int = 7):
        self.out, self.rate = out, rate
        self.rng = random.Random(seed)
        self.incident_until = 0.0
        self.inventory = False
        # shop-api v2.1: checkouts carry the customer's currency (a constant becomes a variable) and
        # "payment … authorized" reads "payment … approved" (same event, new wording)
        self.firmware = False
        self.orders = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- what happens -----------------------------------------------------------------
    # Like real services: the API logs the host's local wall clock without a zone, nginx its
    # local time with the offset, Postgres and inventory say UTC.
    def _api(self, now: datetime, level: str, thread: int, logger: str, text: str) -> str:
        now = now.astimezone()
        return f"{now:%Y-%m-%d %H:%M:%S}.{now.microsecond // 1000:03d} {level:<5} [http-nio-8080-exec-{thread}] {logger} - {text}"

    def _pg(self, now: datetime, text: str) -> str:
        now = now.astimezone(UTC)
        return f"{now:%Y-%m-%d %H:%M:%S}.{now.microsecond // 1000:03d} UTC [{self.rng.randint(2000, 2999)}] {text}"

    def _nginx(self, now: datetime, method: str, path: str, status: int) -> str:
        r, now = self.rng, now.astimezone()
        return (f'10.0.{r.randint(0, 3)}.{r.randint(2, 254)} - - [{now:%d/%b/%Y:%H:%M:%S %z}] '
                f'"{method} {path} HTTP/1.1" {status} {r.randint(120, 9000)} "-" "{r.choice(AGENTS)}"')

    def _stack(self, error: str, frames: list[str]) -> str:
        return "\n".join([error, *(f"\tat {f}" for f in frames)])

    def order(self, now: datetime | None = None, incident: bool = False) -> list[Event]:
        """One customer order, as the four services log it."""
        now = now or datetime.now(UTC)
        r = self.rng
        self.orders += 1
        user, cart, sku = r.randint(1, 500), r.randint(1000, 9999), r.randint(1, 80)
        pay, oid, th = f"pay_{r.randint(10**6, 10**7 - 1)}", r.randint(10**5, 10**6 - 1), r.randint(1, 12)
        total = f"{r.randint(5, 400)}.{r.randint(0, 99):02d}"
        ev = [
            Event("nginx", self._nginx(now, "GET", f"/api/products/{sku}", 200)),
            Event("postgres", self._pg(now, f"LOG:  duration: {r.uniform(0.2, 4):.3f} ms  statement: SELECT id, name, price FROM products WHERE id = {sku}")),
            Event("nginx", self._nginx(now, "GET", "/api/cart", 200)),
            Event("shop-api", self._api(now, "INFO", th, "c.s.cart.CartController", f"cart {cart} viewed by user {user} ({r.randint(1, 6)} items)")),
            Event("nginx", self._nginx(now, "POST", "/api/checkout", 200 if not incident else 502)),
            Event("shop-api", self._api(now, "INFO", th, "c.s.checkout.CheckoutService", f"checkout started for cart {cart} total={total} {r.choice(('EUR', 'USD', 'GBP')) if self.firmware else 'EUR'}")),
        ]
        api = lambda lvl, logger, text: ev.append(Event("shop-api", self._api(now, lvl, th, logger, text)))
        if incident:  # gateway outage: retries, timeouts, failed orders, exhausted DB pool
            for _ in range(r.randint(2, 3)):
                api("WARN", "c.s.payment.PaymentClient", f"payment {pay} slow response {r.randint(3000, 9000)} ms, retrying")
            api("ERROR", "c.s.payment.PaymentClient", f"payment {pay} declined: gateway timeout after 30000 ms")
            api("ERROR", "c.s.order.OrderService", self._stack(
                f"order {oid} failed\njava.util.concurrent.TimeoutException: payment gateway did not answer in 30000 ms",
                ["com.shop.payment.PaymentClient.authorize(PaymentClient.java:118)",
                 "com.shop.checkout.CheckoutService.complete(CheckoutService.java:88)",
                 "com.shop.order.OrderService.place(OrderService.java:57)"]))
            ev.append(Event("postgres", self._pg(now, "FATAL:  remaining connection slots are reserved for non-replication superuser connections")))
            return ev
        api("INFO", "c.s.payment.PaymentClient", f"payment {pay} {'approved' if self.firmware else 'authorized'} in {r.randint(80, 900)} ms")
        ev.append(Event("postgres", self._pg(now, f"LOG:  duration: {r.uniform(1, 12):.3f} ms  statement: INSERT INTO orders (id, user_id, total) VALUES ({oid}, {user}, {total})")))
        api("INFO", "c.s.order.OrderService", f"order {oid} confirmed for user {user}")
        if self.inventory:
            ev.append(Event("inventory", f'time={now.astimezone(UTC):%Y-%m-%dT%H:%M:%SZ} level=info msg="stock reserved" sku={sku} qty={r.randint(1, 4)} warehouse=ber-{r.randint(1, 3)}'))
            if r.random() < 0.15:
                ev.append(Event("inventory", f'time={now.astimezone(UTC):%Y-%m-%dT%H:%M:%SZ} level=warn msg="stock level low" sku={sku} left={r.randint(0, 9)}'))
        return ev

    # --- injected faults (logpipe.eval.rca_eval) ----------------------------------------
    def fault(self, mode: str, now: datetime, step: int) -> list[Event]:
        """What the services log ``step`` orders after fault ``mode`` started: step 0 carries the
        root cause; the steps after it the symptoms a person would see first (502s, failed orders)."""
        f = FAULTS[mode]
        r = self.rng
        api = lambda lvl, logger, text: Event("shop-api", self._api(now, lvl, r.randint(1, 12), logger, text))
        pg = lambda text: Event("postgres", self._pg(now, text))
        root: list[Event] = []
        ev: list[Event] = []
        if step == 0 or (f.get("repeat") and step % f["repeat"] == 0):
            root += [{"api": lambda lvl, text: api(lvl, f.get("logger", "c.s.Main"), text), "pg": lambda _lvl, text: pg(text),
                    "nginx": lambda _lvl, text: Event("nginx", f"{now.astimezone():%Y/%m/%d %H:%M:%S} {text}")}[svc](lvl, text.format(
                        pid=r.randint(200, 9999), ms=r.randint(1000, 9000), n=r.randint(1, 99)))
                   for svc, lvl, text in f["root"]]
        if f.get("normal"):  # the fault doesn't break orders (a control, or a slow burn)
            ev += self.order(now)
        else:
            nginx_status = 502 if f.get("gateway") else 504
            ev += [Event("nginx", self._nginx(now, "GET", f"/api/products/{r.randint(1, 80)}", 200)),
                   Event("nginx", self._nginx(now, "POST", "/api/checkout", nginx_status))]
            if "api" not in f.get("silent", ()) and (step > 0 or not f["root"]):  # the cause comes first
                oid = r.randint(10**5, 10**6 - 1)
                for svc, lvl, text in f.get("symptoms", ()):
                    ev.append(api(lvl, "c.s.order.OrderService", text.format(oid=oid, ms=r.randint(1000, 30000), n=r.randint(1, 50))))
                ev.append(api("ERROR", "c.s.order.OrderService", f"order {oid} failed: checkout could not be completed"))
        silent = {"api": "shop-api", "pg": "postgres", "nginx": "nginx"}
        return root + [e for e in ev if e.service not in {silent[x] for x in f.get("silent", ())}]  # its last words, then quiet

    # --- running ----------------------------------------------------------------------
    def incident(self, seconds: float = 60) -> None:
        self.incident_until = time.monotonic() + seconds

    def path(self, service: str) -> Path:
        return self.out / FILES[service][0]

    def write(self, events: list[Event]) -> None:
        by_file: dict[Path, list[str]] = {}
        for e in events:
            by_file.setdefault(self.path(e.service), []).append(e.message)
        for path, messages in by_file.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write("\n".join(messages) + "\n")

    def start(self) -> None:
        for service in ("nginx", "shop-api", "postgres"):
            self.path(service).parent.mkdir(parents=True, exist_ok=True)
            self.path(service).touch()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="shop-generator")
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(1 / self.rate):
            self.write(self.order(incident=time.monotonic() < self.incident_until))

    def stop(self) -> None:
        self._stop.set()

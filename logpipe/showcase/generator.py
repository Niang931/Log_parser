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
            Event("shop-api", self._api(now, "INFO", th, "c.s.checkout.CheckoutService", f"checkout started for cart {cart} total={total} EUR")),
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
        api("INFO", "c.s.payment.PaymentClient", f"payment {pay} authorized in {r.randint(80, 900)} ms")
        ev.append(Event("postgres", self._pg(now, f"LOG:  duration: {r.uniform(1, 12):.3f} ms  statement: INSERT INTO orders (id, user_id, total) VALUES ({oid}, {user}, {total})")))
        api("INFO", "c.s.order.OrderService", f"order {oid} confirmed for user {user}")
        if self.inventory:
            ev.append(Event("inventory", f'time={now.astimezone(UTC):%Y-%m-%dT%H:%M:%SZ} level=info msg="stock reserved" sku={sku} qty={r.randint(1, 4)} warehouse=ber-{r.randint(1, 3)}'))
            if r.random() < 0.15:
                ev.append(Event("inventory", f'time={now.astimezone(UTC):%Y-%m-%dT%H:%M:%SZ} level=warn msg="stock level low" sku={sku} left={r.randint(0, 9)}'))
        return ev

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

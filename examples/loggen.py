"""Writes a live, realistic app log (with Java stack traces) and an nginx access log, for the
compose demo: the agent tails them like it would on a real host."""
import datetime
import random
import sys
import time
from pathlib import Path

out = Path(sys.argv[1] if len(sys.argv) > 1 else "/logs")
out.mkdir(parents=True, exist_ok=True)
r = random.Random()
while True:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    with (out / "shop.log").open("a") as f:
        for _ in range(r.randint(2, 6)):
            k = r.random()
            if k < 0.4:
                f.write(f"{now} INFO  [http-{r.randint(1, 8)}] GET /api/orders/{r.randint(1, 9999)} 200 in {r.randint(2, 400)}ms\n")
            elif k < 0.65:
                f.write(f"{now} INFO  [worker-{r.randint(1, 4)}] processed batch {r.randint(1, 99999)} size={r.randint(1, 500)}\n")
            elif k < 0.85:
                f.write(f"{now} DEBUG [cache] hit ratio={r.random():.2f} entries={r.randint(100, 9000)}\n")
            elif k < 0.97:
                f.write(f"{now} WARN  [db] slow query took {r.randint(500, 3000)}ms on table orders\n")
            else:
                f.write(f"{now} ERROR [http-{r.randint(1, 8)}] request failed\n"
                        f"java.lang.IllegalStateException: order {r.randint(1, 9999)} not found\n"
                        "\tat com.shop.OrderService.load(OrderService.java:42)\n"
                        "\tat com.shop.Api.get(Api.java:17)\n")
    with (out / "access.log").open("a") as f:
        stamp = datetime.datetime.now().strftime("%d/%b/%Y:%H:%M:%S +0000")
        for _ in range(r.randint(1, 4)):
            f.write(f'10.0.{r.randint(0, 9)}.{r.randint(1, 250)} - - [{stamp}] "GET /static/img{r.randint(1, 50)}.png HTTP/1.1" '
                    f'{r.choice([200, 200, 200, 304, 404])} {r.randint(100, 9000)}\n')
    time.sleep(0.5)

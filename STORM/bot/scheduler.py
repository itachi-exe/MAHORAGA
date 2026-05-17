"""
Continuous London weather bet monitor.

Loop:
  Every 5 minutes → scan Polymarket for tomorrow's London highest-temp markets
  → if new markets found → run model immediately → place bets → log → mark done
  → sleep → repeat

Usage:
  python scheduler.py            # live mode (real orders)
  python scheduler.py --dry-run  # simulate, no real orders
"""
import sys, json, time, logging, argparse, fcntl
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import LOG_FILE, DAILY_BUDGET_USDC
from predictor import run_forecast
from polymarket_client import fetch_london_markets
from betting_strategy import evaluate_markets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("monitor")

POLL_INTERVAL  = 300   # 5 minutes between scans
SEEN_FILE      = LOG_FILE.parent / "already_bet.json"


def _log(entry: dict) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as fp:
        fcntl.flock(fp, fcntl.LOCK_EX)
        try:
            fp.write(json.dumps(entry) + "\n")
        finally:
            fcntl.flock(fp, fcntl.LOCK_UN)


def _load_seen() -> set:
    if SEEN_FILE.exists():
        try:
            return set(json.loads(SEEN_FILE.read_text()))
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Could not load seen file (%s) — starting fresh", e)
    return set()


def _save_seen(seen: set) -> None:
    SEEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    SEEN_FILE.write_text(json.dumps(list(seen)))


def run_monitor(dry_run: bool) -> None:
    log.info("══════════════════════════════════════════════════")
    log.info("London Weather Bet Bot  |  dry_run=%s", dry_run)
    log.info("Daily budget: $%.2f USDC  |  Poll every %ds", DAILY_BUDGET_USDC, POLL_INTERVAL)
    log.info("══════════════════════════════════════════════════")

    # Persist already-bet condition_ids across restarts
    already_bet: set[str] = _load_seen()
    log.info("Loaded %d previously seen condition_ids", len(already_bet))
    forecast_cache: dict  = {}

    while True:
        try:
            now       = datetime.now(timezone.utc)
            tomorrow  = now + timedelta(days=1)

            log.info("Scanning for London temp markets — target date: %s", tomorrow.strftime("%B %-d"))

            markets = fetch_london_markets(target_date=tomorrow)
            new_markets = [m for m in markets if m["condition_id"] not in already_bet]

            if not new_markets:
                if markets:
                    log.info("All %d markets already bet on. Next scan in %ds.", len(markets), POLL_INTERVAL)
                else:
                    log.info("No London markets live yet for %s. Next scan in %ds.",
                             tomorrow.strftime("%B %-d"), POLL_INTERVAL)
            else:
                log.info("%d new London market(s) found — running model now...", len(new_markets))

                today_str = now.strftime("%Y-%m-%d")
                if forecast_cache.get("date") != today_str:
                    fc = run_forecast()
                    if fc is None:
                        log.warning("Forecast returned None — skipping bet cycle")
                        log.info("Next scan in %ds\n", POLL_INTERVAL)
                        time.sleep(POLL_INTERVAL)
                        continue
                    forecast_cache = {"date": today_str, **fc}

                log.info(
                    "Forecast → %.2f°C  (range %.2f–%.2f°C)",
                    forecast_cache["temperature_2m"],
                    forecast_cache["temp_low"],
                    forecast_cache["temp_high"],
                )

                records = evaluate_markets(new_markets, forecast_cache, dry_run=dry_run)

                # Only mark a market as seen if we placed a successful order.
                # Failed / skipped markets stay eligible for retry on the next scan.
                record_map = {r["condition_id"]: r for r in records if "condition_id" in r}
                for m in new_markets:
                    r = record_map.get(m["condition_id"])
                    if r is None:
                        # Was SKIP — not attempted, leave unblocked for retry
                        continue
                    order = r.get("order", {})
                    if isinstance(order, dict) and order.get("success"):
                        already_bet.add(m["condition_id"])
                    elif dry_run and isinstance(order, dict) and order.get("dry_run"):
                        already_bet.add(m["condition_id"])
                    # else: order failed — do NOT add, will retry next scan
                _save_seen(already_bet)

                placed = [r for r in records if r["action"] != "SKIP"]
                log.info(
                    "Bets placed: %d  |  Skipped (low edge): %d",
                    len(placed), len(records) - len(placed),
                )

                _log({
                    "timestamp":  now.isoformat(),
                    "dry_run":    dry_run,
                    "target_date": tomorrow.strftime("%Y-%m-%d"),
                    "forecast":   forecast_cache,
                    "bets":       records,
                })

        except Exception as e:
            log.error("Monitor loop error (will retry in %ds): %s", POLL_INTERVAL, e)

        log.info("Next scan in %ds\n", POLL_INTERVAL)
        time.sleep(POLL_INTERVAL)


def main():
    parser = argparse.ArgumentParser(description="London weather Polymarket bot")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate without placing real orders")
    args = parser.parse_args()
    run_monitor(dry_run=args.dry_run)


if __name__ == "__main__":
    main()

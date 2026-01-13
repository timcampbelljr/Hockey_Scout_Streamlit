#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AHL shot/goal scraper (Play-by-Play feed)

Features:
- Scrape a single game, a list of games, or a continuous range (--start_id/--end_id)
- Saves per-game CSVs and raw JSON responses (for debugging)
- Graceful retries, simple logging, and minimal dependencies

Output:
- ./ahl_shots_<GAMEID>.csv
- ./raw_<GAMEID>/prob_gameCenterPlayByPlay.json

Example:
  python ahl_shot_scrape.py --game_ids 1026478 1027801 --out_dir .
  python ahl_shot_scrape.py --start_id 1026400 --end_id 1026450 --out_dir .
"""

import argparse
import csv
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

PLAYBYPLAY_URL = (
    "https://lscluster.hockeytech.com/feed/index.php"
    "?feed=statviewfeed"
    "&view=gameCenterPlayByPlay"
    "&game_id={gid}"
    "&key={key}"
    "&client_code=ahl"
    "&lang=en"
    "&league_id="
    "&callback=angular.callbacks._5"
)

# Default key seen in your logs; expose as CLI param in case it changes.
DEFAULT_KEY = "ccb91f29d6744675"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (AHL scraper; +https://example.com)",
    "Accept": "*/*",
    "Connection": "keep-alive",
}

JSONP_RE = re.compile(r'^[^(]+\((.*)\)\s*;?\s*$', re.DOTALL)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scrape AHL shot/goal coordinates from HockeyTech play-by-play.")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--game_ids", nargs="+", type=int, help="One or more game IDs (space-separated).")
    grp.add_argument("--start_id", type=int, help="Start of game ID range (inclusive). Use with --end_id.")
    p.add_argument("--end_id", type=int, help="End of game ID range (inclusive). Required with --start_id.")
    p.add_argument("--out_dir", default=".", help="Directory where CSVs (and raw_*/ folders) are written. Default: .")
    p.add_argument("--key", default=DEFAULT_KEY, help=f"Feed key. Default: {DEFAULT_KEY}")
    p.add_argument("--delay", type=float, default=0.7, help="Delay between games in seconds (to be polite). Default: 0.7")
    p.add_argument("--retries", type=int, default=3, help="HTTP retry attempts per game. Default: 3")
    p.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout seconds. Default: 20.0")
    p.add_argument("--overwrite", action="store_true", help="Overwrite CSV if it already exists.")
    return p.parse_args()


def expand_game_ids(args: argparse.Namespace) -> List[int]:
    if args.game_ids:
        return list(dict.fromkeys([int(g) for g in args.game_ids]))  # unique, preserve order
    if args.start_id is not None and args.end_id is not None:
        if args.end_id < args.start_id:
            raise ValueError("--end_id must be >= --start_id")
        return list(range(args.start_id, args.end_id + 1))
    raise ValueError("You must provide --game_ids OR --start_id and --end_id.")


def get_jsonp(url: str, retries: int, timeout: float) -> Optional[Any]:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code != 200:
                last_err = RuntimeError(f"HTTP {r.status_code}")
            else:
                text = r.text.strip()
                # Strip JSONP wrapper
                m = JSONP_RE.match(text)
                payload = m.group(1) if m else text
                return json.loads(payload)
        except Exception as e:
            last_err = e
        time.sleep(0.5 * attempt)  # backoff
    print(f"  !! Failed to fetch after {retries} attempts: {last_err}")
    return None


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def save_raw_json(gid: int, out_dir: str, data: Any) -> None:
    raw_dir = os.path.join(out_dir, f"raw_{gid}")
    ensure_dir(raw_dir)
    path = os.path.join(raw_dir, "prob_gameCenterPlayByPlay.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_row(gid: int, ev: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Normalize an event dict to a flat CSV row. We keep both 'shot' and 'goal' events
    since both can carry x/y coordinates.

    Returns None if the event cannot be normalized to a row with coordinates.
    """
    event_type = ev.get("event")
    details = ev.get("details") or {}

    # We only care about shot-like events with coordinates.
    # Many pbp entries are penalties, goalie changes, etc.
    # For 'goal', the coords live on 'details' as well.
    x = details.get("xLocation")
    y = details.get("yLocation")
    if x is None or y is None:
        return None

    # period/time
    period = (details.get("period") or {}).get("shortName") or ""
    clock = details.get("time") or ""

    # Shooter info (if present)
    shooter = details.get("shooter") or {}
    goalie = details.get("goalie") or {}

    # Flags and other fields that often appear on shots
    is_goal = details.get("isGoal")
    shot_quality = details.get("shotQuality")
    shot_type = details.get("shotType")
    shooter_team_id = details.get("shooterTeamId")

    # Some 'goal' objects have different structure for scorer/assists; we still map to shooter-like fields when possible.
    if event_type == "goal" and not shooter:
        # fall back to 'scoredBy' block if present
        scored = details.get("scoredBy") or {}
        shooter = scored

    row = {
        "game_id": gid,
        "event": event_type,
        "period": period,
        "time": clock,
        "shooter_id": shooter.get("id"),
        "shooter_first": shooter.get("firstName"),
        "shooter_last": shooter.get("lastName"),
        "shooter_team_id": shooter_team_id,
        "goalie_id": goalie.get("id"),
        "goalie_first": goalie.get("firstName"),
        "goalie_last": goalie.get("lastName"),
        "is_goal": is_goal,
        "shot_quality": shot_quality,
        "shot_type": shot_type,
        "x": x,
        "y": y,
    }

    return row


def write_csv(path: str, rows: List[Dict[str, Any]], overwrite: bool) -> None:
    if rows and (overwrite or not os.path.exists(path)):
        fieldnames = [
            "game_id", "event", "period", "time",
            "shooter_id", "shooter_first", "shooter_last", "shooter_team_id",
            "goalie_id", "goalie_first", "goalie_last",
            "is_goal", "shot_quality", "shot_type",
            "x", "y",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)


def scrape_game(gid: int, key: str, out_dir: str, retries: int, timeout: float, overwrite: bool) -> Tuple[int, int]:
    """
    Returns (rows_written, total_events_seen_for_debug)
    """
    csv_path = os.path.join(out_dir, f"ahl_shots_{gid}.csv")
    if os.path.exists(csv_path) and not overwrite:
        print(f"[{gid}] CSV exists → {csv_path} (skip; use --overwrite to regenerate)")
        return (0, 0)

    url = PLAYBYPLAY_URL.format(gid=gid, key=key)
    print(f"[{gid}] Fetching PBP…")
    data = get_jsonp(url, retries=retries, timeout=timeout)
    if data is None:
        print(f"[{gid}] No data returned.")
        return (0, 0)

    # Save raw for debugging
    save_raw_json(gid, out_dir, data)

    rows: List[Dict[str, Any]] = []
    total = 0
    if isinstance(data, list):
        for ev in data:
            total += 1
            row = extract_row(gid, ev)
            if row:
                rows.append(row)
    else:
        print(f"[{gid}] Unexpected payload type: {type(data)}")

    write_csv(csv_path, rows, overwrite=overwrite)

    if rows:
        print(f"[{gid}] Found {len(rows)} rows → {csv_path}")
    else:
        print(f"[{gid}] Found 0 rows → {csv_path}")
        print(f"[{gid}] Tip: inspect ./raw_{gid}/prob_gameCenterPlayByPlay.json for available fields.")

    return (len(rows), total)


def main():
    args = parse_args()
    game_ids = expand_game_ids(args)
    ensure_dir(args.out_dir)

    print(f"[ALL] Total games: {len(game_ids)} | out_dir={args.out_dir}")
    grand_rows = 0
    for idx, gid in enumerate(game_ids, 1):
        print(f"== {idx}/{len(game_ids)}: {gid} ==")
        try:
            wrote, _ = scrape_game(
                gid=gid,
                key=args.key,
                out_dir=args.out_dir,
                retries=args.retries,
                timeout=args.timeout,
                overwrite=args.overwrite,
            )
            grand_rows += wrote
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            break
        except Exception as e:
            print(f"[{gid}] Error: {e}")
        time.sleep(args.delay)

    print(f"[ALL] Done. Total shot/goal rows written: {grand_rows}")


if __name__ == "__main__":
    main()
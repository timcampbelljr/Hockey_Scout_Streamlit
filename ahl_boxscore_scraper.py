import argparse
import csv
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import requests

BASE_URL = "https://lscluster.hockeytech.com/feed/index.php"
DEFAULT_KEY = "ccb91f29d6744675"
DEFAULT_CLIENT = "ahl"
DEFAULT_LANG = "en"
DEFAULT_SITE = 3

CSV_COLS = [
    "a", "g", "ga", "game_id", "is_goalie", "mins", "number", "pim",
    "plus_minus", "pos", "shots", "skater", "svs", "team_id", "team_name", "team_side"
]

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)

# ---------------- utils ----------------

def log(s: str, flush=True):
    print(s, flush=flush)

def strip_jsonp(text: str) -> str:
    t = text.strip()
    m = re.match(r"^\s*angular\.callbacks\._\d+\s*\(\s*(.+?)\s*\)\s*;?\s*$", t, re.DOTALL)
    return m.group(1) if m else t

def as_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        try:
            return json.loads(strip_jsonp(text))
        except Exception:
            return None

def params_plain(game_id: int, key: str, client: str, lang: str,
                 site: int, with_league_param: bool) -> Dict[str, str]:
    p = {
        "feed": "statviewfeed",
        "view": "gameSummary",
        "game_id": str(game_id),
        "key": key,
        "client_code": client,
        "lang": lang,
        "site_id": str(site),
    }
    if with_league_param:
        p["league_id"] = ""
    return p

def params_jsonp(game_id: int, key: str, client: str, lang: str, site: int,
                 with_league_param: bool, cb: int) -> Dict[str, str]:
    p = params_plain(game_id, key, client, lang, site, with_league_param)
    p["callback"] = f"angular.callbacks._{cb}"
    return p

def attempt_matrix(game_id: int, key: str, client: str, lang: str, site: int) -> List[Dict[str, str]]:
    attempts: List[Dict[str, str]] = []
    sites = [site] if site != 1 else [1, 3]
    if 1 not in sites:
        sites.append(1)
    for s in sites:
        for with_league in (True, False):
            attempts.append(params_plain(game_id, key, client, lang, s, with_league))
            attempts.append(params_jsonp(game_id, key, client, lang, s, with_league, 6))
            attempts.append(params_jsonp(game_id, key, client, lang, s, with_league, 5))
    return attempts

def fetch_summary(game_id: int, key: str, client: str, lang: str, site: int,
                  retries: int, timeout: int, delay: float, debug: bool
                  ) -> Optional[Dict[str, Any]]:
    for params in attempt_matrix(game_id, key, client, lang, site):
        q = "&".join([f"{k}={v}" for k, v in params.items()])
        for attempt in range(retries + 1):
            try:
                r = requests.get(
                    BASE_URL, params=params, timeout=timeout,
                    headers={
                        "User-Agent": UA,
                        "Accept": "application/json,text/javascript,*/*;q=0.9",
                    },
                )
                text = r.text or ""
                if debug:
                    log(f"       GET {BASE_URL}?{q}")
                    log(f"       → HTTP {r.status_code}, {len(text)} bytes")
                    if text:
                        log(f"       preview: {text[:180].replace(chr(10),' ')}…")
                if r.status_code != 200:
                    if delay:
                        time.sleep(delay)
                    continue
                data = as_json(text)
                if isinstance(data, dict):
                    if "error" in data:
                        if debug:
                            log(f"       API error: {data.get('error')}")
                        break
                    return data
            except Exception as e:
                if debug:
                    log(f"       request error: {e}")
            if delay:
                time.sleep(delay)
    return None

# -------------- parse → rows --------------

def pname(info: Dict[str, Any]) -> str:
    return (" ".join(x for x in [(info or {}).get("firstName", "").strip(),
                                 (info or {}).get("lastName", "").strip()]
                      if x)).strip()

def normalize_toi(v: Any) -> str:
    return v or ""

def rows_from_team(team_blob: Dict[str, Any], side: str, game_id: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    info = (team_blob or {}).get("info") or {}
    team_id = info.get("id", "")
    team_name = info.get("name", "")

    for p in (team_blob or {}).get("skaters") or []:
        pi = p.get("info") or {}
        ps = p.get("stats") or {}
        out.append({
            "a": ps.get("assists", 0) or 0,
            "g": ps.get("goals", 0) or 0,
            "ga": "",
            "game_id": game_id,
            "is_goalie": 0,
            "mins": normalize_toi(ps.get("toi") or ps.get("timeOnIce")),
            "number": pi.get("jerseyNumber", ""),
            "pim": ps.get("penaltyMinutes", 0) or 0,
            "plus_minus": ps.get("plusMinus", 0) or 0,
            "pos": (pi.get("position") or "") or "",
            "shots": ps.get("shots", 0) or 0,
            "skater": pname(pi),
            "svs": "",
            "team_id": team_id,
            "team_name": team_name,
            "team_side": side,
        })

    for g in (team_blob or {}).get("goalies") or []:
        pi = g.get("info") or {}
        ps = g.get("stats") or {}
        out.append({
            "a": ps.get("assists", 0) if ps.get("assists") is not None else 0,
            "g": ps.get("goals", 0) if ps.get("goals") is not None else 0,
            "ga": ps.get("goalsAgainst", ""),
            "game_id": game_id,
            "is_goalie": 1,
            "mins": normalize_toi(ps.get("timeOnIce") or ps.get("toi")),
            "number": pi.get("jerseyNumber", ""),
            "pim": ps.get("penaltyMinutes", 0) or 0,
            "plus_minus": ps.get("plusMinus", 0) or 0,
            "pos": "G",
            "shots": "",
            "skater": pname(pi),
            "svs": ps.get("saves", ""),
            "team_id": team_id,
            "team_name": team_name,
            "team_side": side,
        })
    return out

def parse_boxscore(payload: Dict[str, Any], game_id: int, debug: bool = False) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        if debug:
            log("       parse_boxscore: non-dict payload")
        return []
    if "homeTeam" not in payload or "visitingTeam" not in payload:
        if debug:
            log("       parse_boxscore: missing homeTeam/visitingTeam")
        return []
    rows: List[Dict[str, Any]] = []
    rows += rows_from_team(payload.get("homeTeam") or {}, "home", game_id)
    rows += rows_from_team(payload.get("visitingTeam") or {}, "away", game_id)
    return rows

# -------------- IO --------------

def write_csv(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_COLS})

def save_raw(path: str, payload: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

# -------------- main --------------

def main():
    ap = argparse.ArgumentParser(description="AHL box-score scraper (gameSummary only)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--game_ids", nargs="+", type=int)
    g.add_argument("--start_id", type=int)
    ap.add_argument("--end_id", type=int)

    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--key", default=DEFAULT_KEY)
    ap.add_argument("--client_code", default=DEFAULT_CLIENT)
    ap.add_argument("--lang", default=DEFAULT_LANG)
    ap.add_argument("--site_id", type=int, default=DEFAULT_SITE)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=15)
    ap.add_argument("--delay", type=float, default=0.6)
    ap.add_argument("--per_game_dirs", action="store_true", help="(ignored, kept for compatibility)")
    ap.add_argument("--save_raw", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.game_ids:
        ids = [int(x) for x in args.game_ids]
    else:
        if args.end_id is None:
            ids = [int(args.start_id)]
        else:
            step = 1 if args.end_id >= args.start_id else -1
            ids = list(range(int(args.start_id), int(args.end_id) + step, step))

    out_root = os.path.abspath(args.out_dir)
    os.makedirs(out_root, exist_ok=True)
    log(f"[ALL] Total games: {len(ids)} | out_dir={out_root}")

    for i, gid in enumerate(ids, 1):
        log(f"== {i}/{len(ids)}: {gid} ==")

        payload = fetch_summary(
            gid, args.key, args.client_code, args.lang, args.site_id,
            retries=args.retries, timeout=args.timeout, delay=args.delay, debug=args.debug
        )

        csv_path = os.path.join(out_root, f"ahl_boxscore_{gid}.csv")

        if payload is None:
            log(f"[{gid}] ERROR: could not fetch or parse JSON.")
            write_csv(csv_path, [])
            continue

        if args.save_raw:
            save_raw(os.path.join(out_root, f"box_gameSummary_{gid}.json"), payload)

        rows = parse_boxscore(payload, gid, debug=args.debug)
        write_csv(csv_path, rows)
        log(f"[{gid}] Box score: {len(rows)} rows → {csv_path}")

        if i != len(ids) and args.delay:
            time.sleep(args.delay)

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()

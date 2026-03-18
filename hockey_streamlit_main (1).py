"""
Syracuse Crunch Scouting Dashboard - Single Page with Player Cards
Complete player profiles with box scores, shot charts, shootout, and faceoffs

Run with: streamlit run hockey_streamlit_main.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import logging
import json

# Page config
st.set_page_config(
    page_title="Syracuse Crunch Scouting",
    page_icon="🏒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.5rem;
        color: #3b82f6;
        text-align: center;
        margin-bottom: 2rem;
    }
    .player-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
    }
    .player-name {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .player-position {
        font-size: 1.25rem;
        opacity: 0.9;
        margin-bottom: 1rem;
    }
    .stat-box {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .stat-label {
        font-size: 0.875rem;
        opacity: 0.85;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e3a8a;
        margin: 2rem 0 1rem 0;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 0.5rem;
    }
    .goalie-card {
        background: linear-gradient(135deg, #7c3aed 0%, #a78bfa 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    .roster-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e3a8a;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3b82f6;
    }
    div[data-testid="stVerticalBlock"] > div:has(button) {
        gap: 0.5rem;
    }
    button[kind="secondary"] {
        border: 1px solid #e5e7eb !important;
    }
    button[kind="primary"] {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%) !important;
        border: none !important;
    }
    .excluded-player {
        opacity: 0.5;
        text-decoration: line-through;
    }
    .manage-roster-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        border: 2px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

UPLOAD_DIR = Path("uploaded_data")
ASSETS_DIR = Path("assets")
CRUNCH_DATA_DIR = Path("Crunch_Box_and_Shot")
EXCLUDED_PLAYERS_FILE = CRUNCH_DATA_DIR / "excluded_players.json"
ROSTER_FILE = CRUNCH_DATA_DIR / "Crunch_Roster.txt"
SCREEN_RECORDINGS_DIR = Path("Screen Recordings")

UPLOAD_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True)
CRUNCH_DATA_DIR.mkdir(exist_ok=True)
SCREEN_RECORDINGS_DIR.mkdir(exist_ok=True)

TARGET_TEAM = "Syracuse Crunch"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ============================================================================
# PLAYER EXCLUSION MANAGEMENT
# ============================================================================

def load_current_roster():
    if not ROSTER_FILE.exists():
        return set()
    try:
        with open(ROSTER_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        roster = set()
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '–' in line or '-' in line:
                parts = line.replace('–', '-').split('-', 1)
                if len(parts) == 2:
                    name = parts[1].strip()
                    if name:
                        roster.add(name)
        logging.info(f"Loaded {len(roster)} players from current roster")
        return roster
    except Exception as e:
        logging.error(f"Error loading roster file: {e}")
        return set()


def load_excluded_players():
    if EXCLUDED_PLAYERS_FILE.exists():
        try:
            with open(EXCLUDED_PLAYERS_FILE, 'r') as f:
                data = json.load(f)
                return set(data.get('excluded_players', []))
        except Exception as e:
            logging.error(f"Error loading excluded players: {e}")
            return set()
    return set()


def save_excluded_players(excluded_set):
    try:
        with open(EXCLUDED_PLAYERS_FILE, 'w') as f:
            json.dump({'excluded_players': list(excluded_set)}, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving excluded players: {e}")


def filter_excluded_players(df, excluded_players):
    if df.empty or not excluded_players:
        return df
    return df[~df['skater'].isin(excluded_players)].copy()


# ============================================================================
# XG MODEL FUNCTIONS
# ============================================================================

def calculate_shot_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    goal_y, goal_left_x, goal_right_x = 200, 50, 798
    d1 = np.sqrt((df["x"] - goal_left_x) ** 2 + (df["y"] - goal_y) ** 2)
    d2 = np.sqrt((df["x"] - goal_right_x) ** 2 + (df["y"] - goal_y) ** 2)
    df["distance"] = np.minimum(d1, d2)
    dx = np.minimum(np.abs(df["x"] - goal_left_x), np.abs(df["x"] - goal_right_x)).replace(0, 0.1)
    dy = np.abs(df["y"] - goal_y)
    df["angle"]   = np.arctan(dy / dx) * (180 / np.pi)
    df["abs_y"]   = dy
    df["is_slot"] = ((df["distance"] < 100) & (dy < 40)).astype(int)
    return df


def estimate_xg_simple(distance: float, angle: float) -> float:
    dist_factor  = 1.0 if pd.isna(distance) or distance < 0 else 0.8 * np.exp(-0.05 * distance)
    angle_factor = 1.0 if pd.isna(angle) else max(0.1, 1.0 - abs(angle) / 90.0)
    return float(np.clip(dist_factor * angle_factor, 0.01, 0.99))


def predict_xg_batch(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    return df.apply(
        lambda row: estimate_xg_simple(row.get("distance", 50), row.get("angle", 0)), axis=1)


# ============================================================================
# DATA LOADING
# ============================================================================

def determine_season_from_game_id(game_id: int) -> str:
    game_id_str = str(game_id)
    if game_id_str.startswith("10") and len(game_id_str) >= 5:
        try:
            season_code = int(game_id_str[2:4])
            if season_code == 28:
                season_code = 27
            start_year    = season_code + 1997
            end_year_short = (start_year + 1) % 100
            return f"{start_year}-{end_year_short:02d}"
        except (ValueError, IndexError):
            pass
    return "2023-24" if game_id > 1024000 else "2024-25"


@st.cache_data
def load_all_data():
    def get_unique_files(pattern):
        all_files = {}
        for d in [UPLOAD_DIR, ASSETS_DIR, CRUNCH_DATA_DIR]:
            if d.exists():
                for f in d.glob(pattern):
                    all_files[f.name] = f
        return sorted(all_files.values(), key=lambda x: x.name)

    boxscore_files = get_unique_files("ahl_boxscore_*.csv")
    shot_files     = get_unique_files("ahl_shots_*.csv")
    logging.info(f"Found {len(boxscore_files)} boxscore files, {len(shot_files)} shot files")

    all_shots = []
    for f in shot_files:
        try:
            all_shots.append(pd.read_csv(f))
        except Exception as e:
            logging.exception(f"Error reading {f}: {e}")

    shot_df = pd.DataFrame()
    if all_shots:
        shot_df = pd.concat(all_shots).drop_duplicates()
        shot_df["shooter"] = (
            shot_df["shooter_first"].fillna("") + " " + shot_df["shooter_last"].fillna("")
        ).str.strip()
        shot_df["goalie"] = (
            shot_df["goalie_first"].fillna("") + " " + shot_df["goalie_last"].fillna("")
        ).str.strip()
        shot_df["game_id"]         = shot_df["game_id"].astype(int)
        shot_df["shooter_team_id"] = pd.to_numeric(shot_df["shooter_team_id"], errors="coerce")
        shot_df["x"]               = pd.to_numeric(shot_df["x"], errors="coerce") * (848 / 600)
        shot_df["y"]               = pd.to_numeric(shot_df["y"], errors="coerce") * (400 / 300)
        shot_df["is_goal"]         = shot_df["is_goal"].astype(bool)
        try:
            shot_df             = calculate_shot_features(shot_df)
            shot_df["xg"]       = predict_xg_batch(shot_df).round(3)
            shot_df["distance"] = shot_df["distance"].round(1)
        except Exception as e:
            logging.exception(f"xG error: {e}")
            shot_df["xg"] = 0.0

    all_players, all_goalies, all_games, game_team_ids = [], [], [], []

    for f in boxscore_files:
        try:
            df = pd.read_csv(f)
            if df.empty:
                continue
            game_id      = int(df["game_id"].iloc[0])
            season       = determine_season_from_game_id(game_id)
            home_team    = df[df["team_side"] == "home"]["team_name"].iloc[0]
            away_team    = df[df["team_side"] == "away"]["team_name"].iloc[0]
            home_team_id = int(df[df["team_side"] == "home"]["team_id"].iloc[0])
            away_team_id = int(df[df["team_side"] == "away"]["team_id"].iloc[0])
            all_games.append({"game_id": game_id, "home_team": home_team, "away_team": away_team,
                               "home_team_id": home_team_id, "away_team_id": away_team_id, "season": season})
            crunch_row = df[df["team_name"] == TARGET_TEAM]
            if not crunch_row.empty:
                cid = int(crunch_row["team_id"].iloc[0])
                oid = away_team_id if cid == home_team_id else home_team_id
                game_team_ids.append({"game_id": game_id, "crunch_team_id": cid, "opponent_team_id": oid})
            df["is_goalie"] = df["pos"] == "G"
            df["season"]    = season
            all_players.append(df[~df["is_goalie"]].copy().rename(
                columns={"g": "goals", "a": "assists", "pim": "penalty_minutes", "sog": "shots"}))
            all_goalies.append(df[df["is_goalie"]].copy().rename(
                columns={"svs": "saves", "ga": "goals_against", "mins": "minutes_played"}))
        except Exception as e:
            logging.exception(f"Error reading {f}: {e}")

    players_df = pd.concat(all_players) if all_players else pd.DataFrame()
    goalies_df = pd.concat(all_goalies) if all_goalies else pd.DataFrame()

    if not players_df.empty:
        agg = {c: "sum" for c in ["goals", "assists", "penalty_minutes", "plus_minus", "shots"]}
        agg.update({"pos": "first", "team_name": "first", "team_id": "first",
                    "team_side": "first", "season": "first", "number": "first"})
        players_df = players_df.groupby(["game_id", "skater"], as_index=False).agg(agg)

    if not goalies_df.empty:
        agg = {c: "sum" for c in ["saves", "goals_against"]}
        agg.update({"pos": "first", "team_name": "first", "team_id": "first",
                    "team_side": "first", "season": "first", "number": "first",
                    "minutes_played": "first"})
        goalies_df = goalies_df.groupby(["game_id", "skater"], as_index=False).agg(agg)

    games_df     = pd.DataFrame(all_games).drop_duplicates(subset=["game_id"]) if all_games else pd.DataFrame()
    game_team_df = pd.DataFrame(game_team_ids).drop_duplicates(subset=["game_id"]) if game_team_ids else pd.DataFrame()

    if not players_df.empty:
        players_df = players_df[players_df["team_name"] == TARGET_TEAM]
    if not goalies_df.empty:
        goalies_df = goalies_df[goalies_df["team_name"] == TARGET_TEAM]

    shot_df_players = pd.DataFrame()
    shot_df_goalies = pd.DataFrame()

    if not shot_df.empty and not games_df.empty:
        if "season" not in shot_df.columns:
            shot_df = pd.merge(shot_df, games_df[["game_id", "season"]], on="game_id", how="left")
        if not game_team_df.empty:
            shot_df = pd.merge(shot_df, game_team_df, on="game_id", how="left")
            shot_df_players = shot_df[shot_df["shooter_team_id"] == shot_df["crunch_team_id"]].copy()
            if not players_df.empty:
                shot_df_players = shot_df_players[
                    shot_df_players["shooter"].isin(players_df["skater"].unique())]
            shot_df_goalies = shot_df[shot_df["shooter_team_id"] == shot_df["opponent_team_id"]].copy()
            if not goalies_df.empty:
                shot_df_goalies = shot_df_goalies[
                    shot_df_goalies["goalie"].isin(goalies_df["skater"].unique())]
        else:
            if not goalies_df.empty:
                shot_df_goalies = shot_df[
                    shot_df["goalie"].isin(goalies_df["skater"].unique())].copy()

    return players_df, goalies_df, games_df, shot_df_players, shot_df_goalies


@st.cache_data
def load_faceoff_data():
    try:
        files = list(ASSETS_DIR.glob("Faceoffs*.csv")) + list(CRUNCH_DATA_DIR.glob("Faceoffs*.csv"))
        if not files:
            return pd.DataFrame()
        df = pd.read_csv(files[0])
        df.columns = df.columns.str.lower()
        df = df.rename(columns={"name": "player"}).dropna(subset=["player"])
        for col in ["overall", "offensive", "defensive", "neutral"]:
            if col in df.columns:
                df[col] = (df[col] * 100).round(1)
        if "total_faceoffs" in df.columns:
            df["total_faceoffs"] = df["total_faceoffs"].fillna(0).astype(int)
        return df
    except Exception as e:
        logging.exception(f"Error loading faceoff data: {e}")
        return pd.DataFrame()


@st.cache_data
def load_shootout_data():
    try:
        for directory in [CRUNCH_DATA_DIR, ASSETS_DIR, UPLOAD_DIR, Path(".")]:
            if not directory.exists():
                continue
            f = directory / "Shootout_Scouting(Crunch SO).csv"
            if not f.exists():
                continue
            df = None
            for enc in ['utf-8', 'cp1252', 'latin-1']:
                try:
                    df = pd.read_csv(f, encoding=enc, on_bad_lines='skip')
                    break
                except Exception:
                    continue
            if df is None or df.empty:
                continue
            df.columns = df.columns.str.strip().str.lower()
            df = df.rename(columns={
                'where player shot from on ice':     'shot_location_ice',
                'where the shot went on goal':       'shot_location_goal',
                'what move they made':               'move_type',
                "goalie (don't worry about this)":   'goalie',
                'goalie (don\'t worry about this)':  'goalie',
            })
            df["player"] = df["player"].fillna("").astype(str).str.strip()
            df["team"]   = df["team"].fillna("").astype(str).str.strip() if 'team' in df.columns else ""
            df = df[df["player"].notna() & (df["player"] != "") &
                    (df["player"].str.lower() != "idle")].copy()
            df["goal"] = df["goal"].fillna("No").apply(
                lambda x: "Yes" if str(x).strip().lower() in ["yes","y","goal","1","true"] else "No")
            logging.info(f"Loaded {len(df)} shootout records")
            return df
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error loading shootout data: {e}")
        return pd.DataFrame()


# ============================================================================
# VIDEO LOOKUP FUNCTIONS
# ============================================================================

def normalize_date_to_slug(date_str: str):
    """Convert any date string to MM-DD-YYYY."""
    try:
        parsed = pd.to_datetime(date_str, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.strftime("%m-%d-%Y")
    except Exception:
        return None


def find_shootout_video(team: str, shooter_last: str, goalie_full: str, date_str: str):
    """
    Locate an MP4 in 'Screen Recordings/' using the primary simplified format:
        Team_ShooterLastName_GoalieLastName_M-DD-YYYY.mp4

    Also handles legacy dot/On formats and fuzzy matching for:
      - Leading zero variants  (03-14-2026 vs 3-14-2026)
      - Special characters     (Kähkönen → Kahkonen)
      - Separator variations   (dots, _On_, double underscores)
    """
    import unicodedata

    if not SCREEN_RECORDINGS_DIR.exists():
        return None

    date_slug = normalize_date_to_slug(date_str)
    if not date_slug:
        return None

    parts       = date_slug.split("-")
    date_nozero = "-".join(str(int(p)) for p in parts)

    team_slug    = team.strip().replace(" ", "")
    shooter_slug = shooter_last.strip().split()[-1]
    goalie_slug  = goalie_full.strip().split()[-1]

    stems_to_try = set()
    for d in [date_slug, date_nozero]:
        # PRIMARY — simple underscores, no "On":
        stems_to_try.add(f"{team_slug}_{shooter_slug}_{goalie_slug}_{d}")
        # Legacy A — dots with On:
        stems_to_try.add(f"{team_slug}.{shooter_slug}_On_{goalie_slug}.{d}")
        # Legacy B — double underscore:
        stems_to_try.add(f"{team_slug}_{shooter_slug}_On__{goalie_slug}_{d}")
        # Legacy D — single underscore with On:
        stems_to_try.add(f"{team_slug}_{shooter_slug}_On_{goalie_slug}_{d}")

    # Exact match first
    for stem in stems_to_try:
        candidate = SCREEN_RECORDINGS_DIR / f"{stem}.mp4"
        if candidate.exists():
            return candidate

    # Fuzzy: ASCII + lowercase + collapse all separator variants
    def normalize_stem(s):
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()
        s = s.replace("__", "_").replace(".", "_").replace("_on_", "_")
        return s

    normalized_targets = {normalize_stem(s) for s in stems_to_try}

    try:
        for f in SCREEN_RECORDINGS_DIR.iterdir():
            if f.suffix.lower() != ".mp4":
                continue
            if f.stem.lower() in {s.lower() for s in stems_to_try}:
                return f
            if normalize_stem(f.stem) in normalized_targets:
                return f
    except Exception:
        pass

    return None


def get_videos_for_shooter(team: str, shooter_last: str, scouting_df: pd.DataFrame) -> list:
    """Return all video-matched attempts for a shooter, sorted newest-first."""
    results = []
    if scouting_df.empty:
        return results

    player_rows = scouting_df[
        scouting_df["player"].str.strip().str.lower() == shooter_last.strip().lower()
    ]

    for _, row in player_rows.iterrows():
        goalie_raw = str(row.get("goalie", "")).strip()
        date_raw   = str(row.get("date",   "")).strip()
        goal_val   = str(row.get("goal",   "No")).strip()
        team_val   = str(row.get("team",   team)).strip()

        if goalie_raw in ("", "nan") or date_raw in ("", "nan"):
            continue

        video_path = find_shootout_video(team_val, shooter_last, goalie_raw, date_raw)
        if video_path is None:
            continue

        outcome     = "✅ GOAL" if goal_val.lower() == "yes" else "❌ No Goal"
        goalie_last = goalie_raw.strip().split()[-1]
        try:
            label_date = pd.to_datetime(date_raw).strftime("%b %d, %Y")
        except Exception:
            label_date = date_raw

        results.append({
            "label":    f"{label_date} — vs {goalie_last} — {outcome}",
            "path":     video_path,
            "date_raw": date_raw,
            "outcome":  outcome,
            "goalie":   goalie_raw,
        })

    results.sort(key=lambda x: pd.to_datetime(x["date_raw"], errors="coerce"), reverse=True)
    return results


def get_videos_for_goalie(goalie_last: str, scouting_df: pd.DataFrame) -> list:
    """Return all video-matched attempts faced by this goalie, sorted newest-first."""
    results = []
    if scouting_df.empty or "goalie" not in scouting_df.columns:
        return results

    goalie_rows = scouting_df[
        scouting_df["goalie"].astype(str).str.strip().str.lower()
        .str.contains(goalie_last.strip().lower(), na=False)
    ]

    for _, row in goalie_rows.iterrows():
        shooter_raw = str(row.get("player", "")).strip()
        goalie_raw  = str(row.get("goalie", "")).strip()
        date_raw    = str(row.get("date",   "")).strip()
        goal_val    = str(row.get("goal",   "No")).strip()
        team_val    = str(row.get("team",   "")).strip()

        if shooter_raw in ("", "nan") or goalie_raw in ("", "nan") or date_raw in ("", "nan"):
            continue

        shooter_last_name = shooter_raw.strip().split()[-1]
        video_path = find_shootout_video(team_val, shooter_last_name, goalie_raw, date_raw)
        if video_path is None:
            continue

        outcome = "✅ GOAL against" if goal_val.lower() == "yes" else "❌ Save"
        try:
            label_date = pd.to_datetime(date_raw).strftime("%b %d, %Y")
        except Exception:
            label_date = date_raw

        results.append({
            "label":    f"{label_date} — {team_val} {shooter_last_name} — {outcome}",
            "path":     video_path,
            "date_raw": date_raw,
            "outcome":  outcome,
            "shooter":  shooter_raw,
            "team":     team_val,
        })

    results.sort(key=lambda x: pd.to_datetime(x["date_raw"], errors="coerce"), reverse=True)
    return results


def render_video_section(team: str, player_last: str, scouting_df: pd.DataFrame, key_suffix: str = ""):
    """Render video clips for a shooter with dropdown by date."""
    st.markdown("---")
    st.markdown("### 📹 Video Clips")
    videos = get_videos_for_shooter(team, player_last, scouting_df)

    if not videos:
        st.info(
            f"No MP4 clips found for **{player_last}**. "
            f"Add files to `Screen Recordings/` using the format:  \n"
            f"`{team}_{player_last}_GoalieName_M-DD-YYYY.mp4`"
        )
        return

    if len(videos) == 1:
        st.caption(videos[0]["label"])
        st.video(str(videos[0]["path"]))
        return

    # Use a fully unique key combining player last name AND the full key_suffix
    widget_key = f"vid_sel_{player_last}_{key_suffix}".replace(" ", "_")
    options = [v["label"] for v in videos]
    selected_label = st.selectbox(
        f"Select attempt ({len(videos)} clips available):",
        options=options,
        index=0,
        key=widget_key
    )
    chosen = next(v for v in videos if v["label"] == selected_label)
    st.video(str(chosen["path"]))


def render_goalie_video_section(goalie_name: str, scouting_df: pd.DataFrame, key_suffix: str = ""):
    """Render all shootout clips where this goalie was in net."""
    st.markdown("---")
    st.markdown("### 📹 Shootout Film")
    goalie_last = goalie_name.strip().split()[-1]
    videos      = get_videos_for_goalie(goalie_last, scouting_df)

    if not videos:
        st.info(
            f"No MP4 clips found for **{goalie_name}**. "
            f"Add files to `Screen Recordings/` using the format:  \n"
            f"`OpponentTeam_ShooterLastName_{goalie_last}_M-DD-YYYY.mp4`"
        )
        return

    if len(videos) == 1:
        st.caption(videos[0]["label"])
        st.video(str(videos[0]["path"]))
        return

    widget_key = f"goalie_film_sel_{goalie_last}_{key_suffix}".replace(" ", "_")
    options = [v["label"] for v in videos]
    selected_label = st.selectbox(
        f"Select clip ({len(videos)} available):",
        options=options,
        index=0,
        key=widget_key
    )
    chosen = next(v for v in videos if v["label"] == selected_label)
    st.video(str(chosen["path"]))


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_shot_chart(df, player_name, view_type="player"):
    if df.empty:
        return None
    if 'x' not in df.columns or 'y' not in df.columns:
        return None
    df = df.dropna(subset=['x', 'y'])
    df = df[(df['x'] >= 0) & (df['x'] <= 848) & (df['y'] >= 0) & (df['y'] <= 400)]
    if df.empty:
        return None

    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=848, y1=400,
                  line=dict(color="#000000", width=3), fillcolor="rgba(0,0,0,0)")
    for x, c, w in [(50,"red",3),(798,"red",3),(274,"blue",2),(574,"blue",2),(424,"red",3)]:
        fig.add_shape(type="line", x0=x, y0=0, x1=x, y1=400, line=dict(color=c, width=w))
    for x0, x1 in [(35,50),(798,813)]:
        fig.add_shape(type="rect", x0=x0, y0=170, x1=x1, y1=230,
                      line=dict(color="red", width=2), fillcolor="rgba(255,0,0,0.1)")

    if view_type == "player":
        saves = df[~df["is_goal"]]
        if not saves.empty:
            fig.add_trace(go.Scatter(
                x=saves["x"], y=saves["y"], mode='markers',
                marker=dict(size=8, color='#3b82f6', symbol='circle',
                            line=dict(width=1, color='#1e3a8a')),
                name='Shot',
                hovertemplate='<b>Shot</b><br>xG: %{customdata:.2%}<extra></extra>',
                customdata=saves.get('xg', 0)))
        goals = df[df["is_goal"]]
        if not goals.empty:
            fig.add_trace(go.Scatter(
                x=goals["x"], y=goals["y"], mode='markers',
                marker=dict(size=14, color='#10b981', symbol='star',
                            line=dict(width=2, color='#065f46')),
                name='Goal',
                hovertemplate='<b>GOAL</b><br>xG: %{customdata:.2%}<extra></extra>',
                customdata=goals.get('xg', 0)))
    else:
        if not df.empty:
            fig.add_trace(go.Scatter(
                x=df["x"], y=df["y"], mode='markers',
                marker=dict(size=10, color='#ef4444', symbol='x',
                            line=dict(width=2, color='#991b1b')),
                name='Goal Against',
                hovertemplate='<b>Goal Against</b><br>xG: %{customdata:.2%}<extra></extra>',
                customdata=df.get('xg', 0)))

    fig.update_layout(
        title=f"{player_name} - Shot Chart",
        xaxis=dict(range=[0,848], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[0,400], showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x", scaleratio=1),
        plot_bgcolor='#f8f9fa', height=400, hovermode='closest', showlegend=True,
        margin=dict(l=0, r=0, t=40, b=0))
    return fig


def create_nhl_rink_shootout():
    fig = go.Figure()
    fig.add_shape(type="rect", x0=-100, y0=-42.5, x1=100, y1=42.5,
                  line=dict(color="black", width=3), fillcolor="white", layer="below")
    fig.add_shape(type="line", x0=0, y0=-42.5, x1=0, y1=42.5,
                  line=dict(color="red", width=3), layer="below")
    for x in [-25, 25]:
        fig.add_shape(type="line", x0=x, y0=-42.5, x1=x, y1=42.5,
                      line=dict(color="blue", width=2), layer="below")
    for x in [-89, 89]:
        fig.add_shape(type="line", x0=x, y0=-42.5, x1=x, y1=42.5,
                      line=dict(color="red", width=2), layer="below")
    for zx in [-69, 69]:
        for yc in [-22, 22]:
            fig.add_shape(type="circle", x0=zx-15, y0=yc-15, x1=zx+15, y1=yc+15,
                          line=dict(color="red", width=2), fillcolor="rgba(255,0,0,0.1)", layer="below")
            fig.add_shape(type="circle", x0=zx-1, y0=yc-1, x1=zx+1, y1=yc+1,
                          fillcolor="red", line=dict(color="red", width=1), layer="below")
    fig.add_shape(type="circle", x0=-15, y0=-15, x1=15, y1=15,
                  line=dict(color="blue", width=2), fillcolor="rgba(0,0,255,0.05)", layer="below")
    fig.add_shape(type="circle", x0=-1, y0=-1, x1=1, y1=1,
                  fillcolor="blue", line=dict(color="blue", width=1), layer="below")
    for path in [
        "M -89 -4 L -89 4 L -85 4.5 Q -83 4.5 -83 3 L -83 -3 Q -83 -4.5 -85 -4.5 L -89 -4 Z",
        "M 89 -4 L 89 4 L 85 4.5 Q 83 4.5 83 3 L 83 -3 Q 83 -4.5 85 -4.5 L 89 -4 Z",
    ]:
        fig.add_shape(type="path", path=path,
                      line=dict(color="red", width=2), fillcolor="rgba(173,216,230,0.4)", layer="below")
    for x0, x1 in [(-92,-89),(89,92)]:
        fig.add_shape(type="rect", x0=x0, y0=-3, x1=x1, y1=3,
                      line=dict(color="red", width=2), fillcolor="rgba(255,255,255,0.3)", layer="below")
    fig.update_layout(
        showlegend=True,
        xaxis=dict(range=[-105,105], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-45,45],  showgrid=False, zeroline=False, visible=False,
                   scaleanchor="x", scaleratio=1),
        plot_bgcolor='#f0f8ff', height=400, margin=dict(l=10,r=10,t=30,b=10))
    return fig


def create_nhl_goal_net():
    fig = go.Figure()
    fig.add_shape(type="rect", x0=-36, y0=-24, x1=36, y1=24,
                  line=dict(color="red", width=3), fillcolor="rgba(255,255,255,0.3)")
    for x, y in [(-36,-24),(-36,24),(36,-24),(36,24)]:
        fig.add_shape(type="circle", x0=x-2.4, y0=y-2.4, x1=x+2.4, y1=y+2.4,
                      fillcolor="red", line=dict(color="darkred", width=1))
    for x in [-12, 12]:
        fig.add_shape(type="line", x0=x, y0=-24, x1=x, y1=24,
                      line=dict(color="gray", width=1, dash="dash"))
    for y in [-8, 8]:
        fig.add_shape(type="line", x0=-36, y0=y, x1=36, y1=y,
                      line=dict(color="gray", width=1, dash="dash"))
    for x, y, label in [
        (-24,-16,"Bottom\nLeft"),(0,-16,"Bottom\nCenter"),(24,-16,"Bottom\nRight"),
        (-24,  0,"Middle\nLeft"),(0,  0,"Five\nHole"),    (24,  0,"Middle\nRight"),
        (-24, 16,"Top\nLeft"),   (0, 16,"Top\nCenter"),   (24, 16,"Top\nRight"),
    ]:
        fig.add_annotation(x=x, y=y, text=label, showarrow=False,
                           font=dict(size=9, color="gray"), opacity=0.6)
    fig.update_layout(
        showlegend=True,
        xaxis=dict(range=[-45,45], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-30,30], showgrid=False, zeroline=False, visible=False,
                   scaleanchor="x", scaleratio=1),
        plot_bgcolor='white', height=350, margin=dict(l=10,r=10,t=30,b=10))
    return fig


def get_net_zone(x, y):
    if pd.isna(x) or pd.isna(y):
        return "Unknown"
    h = "Left" if x < -12 else ("Right" if x > 12 else "Center")
    v = "Bottom" if y < -8 else ("Top" if y > 8 else "Middle")
    return "Five Hole" if h == "Center" and v == "Middle" else f"{v} {h}"


# ============================================================================
# AGGREGATION FUNCTIONS
# ============================================================================

def aggregate_player_stats(players_df, shots_df, season="2024-25"):
    if players_df.empty:
        return pd.DataFrame()
    season_df = players_df[players_df["season"] == season].copy()
    if season_df.empty:
        return pd.DataFrame()
    for col in ["goals","assists","penalty_minutes","plus_minus","shots"]:
        season_df[col] = pd.to_numeric(season_df[col], errors="coerce").fillna(0)
    season_df["points"] = season_df["goals"] + season_df["assists"]
    agg_df = (
        season_df.groupby("skater")
        .agg(pos=("pos","first"), games_played=("game_id","nunique"),
             goals=("goals","sum"), assists=("assists","sum"), points=("points","sum"),
             plus_minus=("plus_minus","sum"), penalty_minutes=("penalty_minutes","sum"),
             shots=("shots","sum"))
        .reset_index()
    )
    if not shots_df.empty and "xg" in shots_df.columns:
        xg = shots_df.groupby("shooter")["xg"].mean().reset_index()
        xg.rename(columns={"shooter":"skater","xg":"avg_xg"}, inplace=True)
        agg_df = pd.merge(agg_df, xg, on="skater", how="left")
        agg_df["avg_xg"] = agg_df["avg_xg"].fillna(0).round(3)
    else:
        agg_df["avg_xg"] = 0.0
    for col in ["games_played","goals","assists","points","plus_minus","penalty_minutes","shots"]:
        agg_df[col] = agg_df[col].astype(int)
    return agg_df


def aggregate_goalie_stats(goalies_df, season="2024-25"):
    if goalies_df.empty:
        return pd.DataFrame()
    season_df = goalies_df[goalies_df["season"] == season].copy()
    if season_df.empty:
        return pd.DataFrame()
    for col in ["saves","goals_against","minutes_played"]:
        season_df[col] = pd.to_numeric(season_df[col], errors="coerce").fillna(0)
    season_df["played_game"] = season_df["saves"] > 0
    gp = (season_df[season_df["played_game"]].groupby("skater")["game_id"]
          .nunique().reset_index(name="games_played"))
    agg_df = (season_df.groupby("skater")
              .agg(saves=("saves","sum"), goals_against=("goals_against","sum"))
              .reset_index())
    if not gp.empty:
        agg_df = pd.merge(agg_df, gp, on="skater", how="left")
        agg_df["games_played"] = agg_df["games_played"].fillna(0).astype(int)
    else:
        agg_df["games_played"] = 0
    agg_df["save_percentage"] = agg_df.apply(
        lambda r: r["saves"]/(r["saves"]+r["goals_against"])
        if r["saves"]+r["goals_against"] > 0 else 0, axis=1).round(3)
    agg_df["goals_against_average"] = agg_df.apply(
        lambda r: r["goals_against"]/r["games_played"]
        if r["games_played"] > 0 else 0, axis=1).round(2)
    for col in ["saves","goals_against"]:
        agg_df[col] = agg_df[col].astype(int)
    return agg_df


# ============================================================================
# PLAYER CARD
# ============================================================================

def render_player_card(player_name, player_stats, player_shots, faceoff_data, shootout_data, games_df):
    st.markdown(f"""
        <div style='text-align:center;padding:20px;
                    background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                    border-radius:10px;margin-bottom:20px;'>
            <h1 style='color:white;margin:0;'>{player_name}</h1>
            <p style='color:rgba(255,255,255,0.9);margin:5px 0 0 0;font-size:1.1em;'>
                {player_stats['pos']} • Syracuse Crunch</p>
        </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4,c5,c6,c7,c8 = st.columns(8)
    with c1: st.metric("GP",     player_stats['games_played'])
    with c2: st.metric("G",      player_stats['goals'])
    with c3: st.metric("A",      player_stats['assists'])
    with c4: st.metric("PTS",    player_stats['points'])
    with c5: st.metric("+/-",    player_stats['plus_minus'])
    with c6: st.metric("PIM",    player_stats['penalty_minutes'])
    with c7: st.metric("SOG",    player_stats['shots'])
    with c8: st.metric("Avg xG", f"{player_stats['avg_xg']:.3f}")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Box Score","🎯 Shot Chart","🥅 Shootout","⚔️ Faceoffs"])

    # ── TAB 1: BOX SCORE ──────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="stat-card"><h3>Game-by-Game Stats</h3></div>', unsafe_allow_html=True)
        pg = st.session_state.players_df[
            st.session_state.players_df["skater"] == player_name].copy()
        if not pg.empty and not games_df.empty:
            pg = pg.merge(games_df, on="game_id", suffixes=("","_game"))
            pg["opponent"] = pg.apply(
                lambda r: r["away_team"] if r["team_name"] == r["home_team"] else r["home_team"], axis=1)
            pg["points"] = pg["goals"] + pg["assists"]
            pg = pg.sort_values("game_id")
            pg["game_number"] = range(1, len(pg)+1)
            st.dataframe(
                pg[["game_number","opponent","goals","assists","points",
                    "plus_minus","penalty_minutes","shots"]].iloc[::-1],
                hide_index=True, use_container_width=True,
                column_config={"game_number":"Game #","opponent":"Opponent","goals":"G",
                               "assists":"A","points":"PTS","plus_minus":"+/-",
                               "penalty_minutes":"PIM","shots":"SOG"})
        else:
            st.info("No game data available")

    # ── TAB 2: SHOT CHART ─────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="stat-card"><h3>Shot Chart</h3></div>', unsafe_allow_html=True)
        if not player_shots.empty:
            avail_games = sorted(player_shots["game_id"].unique())
            game_lookup = {}
            pg2 = st.session_state.players_df[
                st.session_state.players_df["skater"] == player_name].copy()
            if not pg2.empty and not games_df.empty:
                pg2 = pg2.merge(games_df, on="game_id", suffixes=("","_game"))
                pg2["opponent"] = pg2.apply(
                    lambda r: r["away_team"] if r["team_name"] == r["home_team"] else r["home_team"], axis=1)
                pg2 = pg2.sort_values("game_id")
                pg2["game_number"] = range(1, len(pg2)+1)
                for _, r in pg2.iterrows():
                    game_lookup[r["game_id"]] = f"Game {r['game_number']}: {r['opponent']}"

            col1, col2 = st.columns([3,1])
            with col1:
                gf = st.radio("Show shots from:", ["All Games","Single Game"],
                              horizontal=True, key=f"player_game_filter_{player_name}")
            filtered = player_shots.copy()
            if gf == "Single Game":
                with col2:
                    sg = st.selectbox("Select Game:", avail_games,
                                      index=len(avail_games)-1,
                                      format_func=lambda x: game_lookup.get(x, f"Game {x}"),
                                      key=f"player_single_game_{player_name}")
                filtered = player_shots[player_shots["game_id"] == sg]

            st.markdown("---")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total Shots", len(filtered))
            gc = (filtered["is_goal"]==True).sum()
            c2.metric("Goals", gc)
            c3.metric("Shooting %",
                      f"{(gc/len(filtered))*100:.1f}%" if len(filtered) > 0 else "0.0%")
            c4.metric("Avg xG",
                      f"{filtered['xg'].mean():.3f}" if "xg" in filtered.columns and len(filtered) > 0 else "0.000")
            fig = create_shot_chart(filtered, player_name, view_type="player")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No shot data available for this player")

    # ── TAB 3: SHOOTOUT ───────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">🥅 Shootout Performance</div>', unsafe_allow_html=True)

        so_ice = pd.DataFrame()
        so_net = pd.DataFrame()
        try:
            f = CRUNCH_DATA_DIR / "Crunch25-26SO.csv"
            if f.exists():
                so_ice = pd.read_csv(f)
                so_ice.columns = so_ice.columns.str.strip()
        except Exception as e:
            logging.warning(f"Could not load ice data: {e}")
        try:
            f = CRUNCH_DATA_DIR / "SO_Goalzone.csv"
            if f.exists():
                so_net = pd.read_csv(f)
                so_net.columns = so_net.columns.str.strip()
        except Exception as e:
            logging.warning(f"Could not load net data: {e}")

        psd = pd.DataFrame()
        if not shootout_data.empty:
            psd = shootout_data[shootout_data["player"] == player_name]
            if psd.empty and " " in player_name:
                ln = player_name.split()[-1]
                psd = shootout_data[shootout_data["player"].str.lower() == ln.lower()]

        has_ice = not so_ice.empty
        has_sc  = not psd.empty

        if not has_ice and not has_sc:
            st.info("No shootout data available for this player")
        else:
            ln = player_name.split()[-1] if " " in player_name else player_name
            fn = player_name.split()[0]  if " " in player_name else player_name

            pid = pd.DataFrame()
            pnd = pd.DataFrame()

            if has_ice:
                ci = so_ice[so_ice["Team"] == "Home"].copy()
                pid = ci[ci["Player"].str.lower() == ln.lower()]
                if pid.empty:
                    pid = ci[ci["Player"].str.lower() == player_name.lower()]
                if pid.empty:
                    pid = ci[ci["Player"].str.lower().str.contains(ln.lower(), na=False)]
                if pid.empty:
                    pid = ci[ci["Player"].str.lower().str.contains(fn.lower(), na=False)]

            if not so_net.empty:
                cn = so_net[so_net["Team"] == "Home"].copy()
                pnd = cn[cn["Player"].str.lower() == ln.lower()]
                if pnd.empty:
                    pnd = cn[cn["Player"].str.lower().str.contains(ln.lower(), na=False)]

            found = not pid.empty or not pnd.empty or not psd.empty

            if found:
                if not psd.empty:
                    att  = len(psd)
                    gls  = (psd["goal"] == "Yes").sum()
                elif not pid.empty:
                    att  = len(pid)
                    gls  = (pid["Type"] == "Goal").sum()
                else:
                    att = gls = 0

                c1,c2,c3 = st.columns(3)
                c1.metric("Shootout Attempts", att)
                c2.metric("Goals", gls)
                c3.metric("Success Rate", f"{(gls/att*100):.1f}%" if att > 0 else "0.0%")
                st.markdown("---")

                if not pid.empty or not pnd.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("🏒 Shot Locations on Ice")
                        if not pid.empty:
                            fig_rink = create_nhl_rink_shootout()
                            s_df = pid[pid["Type"] == "Shot"]
                            g_df = pid[pid["Type"] == "Goal"]
                            if not s_df.empty:
                                fig_rink.add_trace(go.Scatter(
                                    x=s_df["X"], y=s_df["Y"], mode='markers',
                                    marker=dict(size=12, color='lightblue', symbol='circle',
                                                line=dict(width=2, color='blue')),
                                    name='Miss', hovertemplate='<b>Miss</b><extra></extra>'))
                            if not g_df.empty:
                                fig_rink.add_trace(go.Scatter(
                                    x=g_df["X"], y=g_df["Y"], mode='markers',
                                    marker=dict(size=16, color='red', symbol='star',
                                                line=dict(width=2, color='darkred')),
                                    name='Goal ⭐', hovertemplate='<b>GOAL!</b><extra></extra>'))
                            fig_rink.update_layout(title=f"{player_name} - Shootout Shot Locations")
                            st.plotly_chart(fig_rink, use_container_width=True)
                        else:
                            st.info("Ice location data not available")

                    with col2:
                        st.subheader("🥅 Shot Locations on Net")
                        if not pnd.empty:
                            fig_net = create_nhl_goal_net()
                            ng = pnd[pnd["Type"].str.lower().isin(['goal','goals'])]
                            ns = pnd[pnd["Type"].str.lower().isin(['save','saves','saved'])]
                            if not ns.empty:
                                fig_net.add_trace(go.Scatter(
                                    x=ns["X"], y=ns["Y"], mode='markers',
                                    marker=dict(size=12, color='lightblue', symbol='x',
                                                line=dict(width=2, color='blue')),
                                    name='Save', hovertemplate='<b>SAVE</b><extra></extra>'))
                            if not ng.empty:
                                fig_net.add_trace(go.Scatter(
                                    x=ng["X"], y=ng["Y"], mode='markers',
                                    marker=dict(size=16, color='red', symbol='star',
                                                line=dict(width=2, color='darkred')),
                                    name='Goal', hovertemplate='<b>GOAL!</b><extra></extra>'))
                            fig_net.update_layout(title=f"{player_name} - Shots on Net")
                            st.plotly_chart(fig_net, use_container_width=True)
                            if not ng.empty:
                                st.markdown("**Goal Locations by Zone:**")
                                ngc = ng.copy()
                                ngc["Zone"] = ngc.apply(lambda r: get_net_zone(r["X"], r["Y"]), axis=1)
                                zc = ngc["Zone"].value_counts().reset_index()
                                zc.columns = ["Zone","Goals"]
                                st.dataframe(zc, hide_index=True, use_container_width=True)
                        else:
                            st.info("Net location data not available")

                if not psd.empty:
                    st.markdown("---")
                    st.subheader("📋 Shootout Details")
                    st.dataframe(psd.head(10), hide_index=True, use_container_width=True)

                # ── VIDEO CLIPS ──────────────────────────────────────

                render_video_section(
                        team=team,
                        player_last=sel.split()[-1],
                        scouting_df=full_sd,
                        key_suffix=f"{goalie_name}_{sel}".replace(" ", "_"),
                    )
            else:
                st.info(f"No shootout data available for {player_name}")
                st.caption("Player must be on the Syracuse Crunch to appear in shootout data")

    # ── TAB 4: FACEOFFS ───────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">Faceoff Statistics</div>', unsafe_allow_html=True)
        if not faceoff_data.empty:
            pf = faceoff_data[faceoff_data["player"] == player_name]
            if not pf.empty:
                row = pf.iloc[0]
                c1,c2,c3,c4,c5 = st.columns(5)
                c1.metric("Total",     row.get("total_faceoffs", 0))
                c2.metric("Overall",   f"{row.get('overall',   0):.1f}%")
                c3.metric("Offensive", f"{row.get('offensive', 0):.1f}%")
                c4.metric("Defensive", f"{row.get('defensive', 0):.1f}%")
                c5.metric("Neutral",   f"{row.get('neutral',   0):.1f}%")
            else:
                st.info("No faceoff data available for this player")
        else:
            st.info("No faceoff data loaded")


# ============================================================================
# GOALIE CARD
# ============================================================================

def render_goalie_card(goalie_name, goalie_stats, goalie_shots, shootout_data, games_df):
    st.markdown(f"""
    <div class="goalie-card">
        <div class="player-name">{goalie_name}</div>
        <div class="player-position">Goalie • Syracuse Crunch</div>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.metric("GP",  goalie_stats['games_played'])
    with c2: st.metric("SVS", goalie_stats['saves'])
    with c3: st.metric("GA",  goalie_stats['goals_against'])
    with c4: st.metric("SV%", f"{goalie_stats['save_percentage']:.3f}")
    with c5: st.metric("GAA", f"{goalie_stats['goals_against_average']:.2f}")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📊 Box Score","🎯 Goals Against Map","🥅 Shootout"])

    # ── TAB 1: BOX SCORE ──────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">Game-by-Game Stats</div>', unsafe_allow_html=True)
        gg = st.session_state.goalies_df[
            st.session_state.goalies_df["skater"] == goalie_name].copy()
        if not gg.empty and not games_df.empty:
            gg = gg.merge(games_df, on="game_id", suffixes=('','_game'))
            gg["opponent"] = gg.apply(
                lambda r: r["away_team"] if r["team_name"] == r["home_team"] else r["home_team"], axis=1)
            gg = gg.sort_values("game_id")
            gg["game_number"] = range(1, len(gg)+1)
            summary = pd.DataFrame([{
                "game_number":"TOTAL", "opponent":f"{len(gg)} Games",
                "saves":gg["saves"].sum(), "goals_against":gg["goals_against"].sum(),
                "sv_pct":f"{goalie_stats['save_percentage']:.3f}",
                "gaa":f"{goalie_stats['goals_against_average']:.2f}",
            }])
            ind = gg[["game_number","opponent","saves","goals_against"]].copy()
            ind["sv_pct"] = ind.apply(
                lambda r: f"{r['saves']/(r['saves']+r['goals_against']):.3f}"
                if r['saves']+r['goals_against'] > 0 else "0.000", axis=1)
            ind["gaa"] = "N/A"
            st.dataframe(
                pd.concat([summary, ind.iloc[::-1]], ignore_index=True),
                hide_index=True, use_container_width=True,
                column_config={"game_number":"Game #","opponent":"Opponent",
                               "saves":"SVS","goals_against":"GA","sv_pct":"SV%","gaa":"GAA"})
        else:
            st.info("No game data available")

    # ── TAB 2: GOALS AGAINST MAP ──────────────────────────────────────
    with tab2:
        st.markdown('<div class="stat-card"><h3>Goals Against Map</h3></div>', unsafe_allow_html=True)
        if not goalie_shots.empty:
            avail = sorted(goalie_shots["game_id"].unique())
            gl    = {}
            gg2   = st.session_state.goalies_df[
                st.session_state.goalies_df["skater"] == goalie_name].copy()
            if not gg2.empty and not games_df.empty:
                gg2 = gg2.merge(games_df, on="game_id", suffixes=("","_game"))
                gg2["opponent"] = gg2.apply(
                    lambda r: r["away_team"] if r["team_name"] == r["home_team"] else r["home_team"], axis=1)
                gg2 = gg2.sort_values("game_id")
                gg2["game_number"] = range(1, len(gg2)+1)
                for _, r in gg2.iterrows():
                    gl[r["game_id"]] = f"Game {r['game_number']}: {r['opponent']}"

            col1, col2 = st.columns([3,1])
            with col1:
                gf = st.radio("Show shots from:", ["All Games","Single Game"],
                              horizontal=True, key=f"goalie_game_filter_{goalie_name}")
            filtered = goalie_shots.copy()
            if gf == "Single Game":
                with col2:
                    sg = st.selectbox("Select Game:", avail, index=len(avail)-1,
                                      format_func=lambda x: gl.get(x, f"Game {x}"),
                                      key=f"goalie_single_game_{goalie_name}")
                filtered = goalie_shots[goalie_shots["game_id"] == sg]

            st.markdown("---")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Shots Faced", len(filtered))
            ga = (filtered["is_goal"]==True).sum()
            c2.metric("Goals Against", ga)
            c3.metric("Save %",
                      f"{((len(filtered)-ga)/len(filtered))*100:.1f}%" if len(filtered) > 0 else "0.0%")
            c4.metric("Avg xG Against",
                      f"{filtered['xg'].mean():.3f}" if "xg" in filtered.columns and len(filtered) > 0 else "0.000")
            fig = create_shot_chart(filtered[filtered["is_goal"]], goalie_name, view_type="goalie")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No goals scored against this goalie in selected game(s)")
        else:
            st.info("No shot data available for this goalie")

    # ── TAB 3: GOALIE SHOOTOUT ────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">🥅 Shootout Performance</div>', unsafe_allow_html=True)

        so_ice = pd.DataFrame()
        so_net = pd.DataFrame()
        try:
            f = CRUNCH_DATA_DIR / "Crunch25-26SO.csv"
            if f.exists():
                so_ice = pd.read_csv(f)
                so_ice.columns = so_ice.columns.str.strip()
        except Exception as e:
            st.warning(f"Could not load shootout ice data: {e}")
        try:
            f = CRUNCH_DATA_DIR / "SO_Goalzone.csv"
            if f.exists():
                so_net = pd.read_csv(f)
                so_net.columns = so_net.columns.str.strip()
        except Exception as e:
            st.warning(f"Could not load shootout net data: {e}")

        # Preserve full scouting data with goalie column intact for video lookups
        full_sd = shootout_data.copy() if not shootout_data.empty else pd.DataFrame()

        gsd = pd.DataFrame()
        if not full_sd.empty and "goalie" in full_sd.columns:
            sd = full_sd.copy()
            sd["goalie"] = sd["goalie"].astype(str)
            gsd = sd[sd["goalie"].str.contains(goalie_name, case=False, na=False)]
            if gsd.empty and " " in goalie_name:
                gsd = sd[sd["goalie"].str.contains(goalie_name.split()[-1], case=False, na=False)]

        # Display copy — goalie column dropped only for the table
        gsd_display = gsd.drop(columns=["goalie"], errors="ignore").copy() if not gsd.empty else pd.DataFrame()

        last_name = goalie_name.split()[-1] if " " in goalie_name else goalie_name
        gid = pd.DataFrame()
        gnd = pd.DataFrame()

        if not so_ice.empty and "Player" in so_ice.columns:
            ai = so_ice[so_ice["Team"] == "Away"].copy()
            ai["Player"] = ai["Player"].astype(str)
            gid = ai[ai["Player"].str.contains(last_name, case=False, na=False)]

        if not so_net.empty and "Player" in so_net.columns:
            an = so_net[so_net["Team"] == "Away"].copy()
            an["Player"] = an["Player"].astype(str)
            gnd = an[an["Player"].str.contains(last_name, case=False, na=False)]

        found = not gsd.empty or not gid.empty or not gnd.empty

        if not found:
            st.info(f"No shootout data available for goalie {goalie_name}")
        else:
            if not gsd.empty and "goal" in gsd.columns:
                att  = len(gsd)
                ga   = (gsd["goal"].astype(str).str.lower() == "yes").sum()
                svs  = att - ga
                spct = (svs / att * 100) if att > 0 else 0
            elif not gid.empty and "Type" in gid.columns:
                att  = len(gid)
                ga   = (gid["Type"].str.lower() == "goal").sum()
                svs  = att - ga
                spct = (svs / att * 100) if att > 0 else 0
            else:
                att = ga = svs = spct = 0

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Shootout Attempts", att)
            c2.metric("Goals Against", ga)
            c3.metric("Saves", svs)
            c4.metric("Save %", f"{spct:.1f}%")
            st.markdown("---")

            if not gid.empty or not gnd.empty:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🏒 Shooter Locations (Ice)")
                    if not gid.empty:
                        fig_rink = create_nhl_rink_shootout()
                        gd = gid[gid["Type"].str.lower() == "goal"]
                        sd = gid[gid["Type"].str.lower().isin(["shot","save","miss"])]
                        if not sd.empty:
                            fig_rink.add_trace(go.Scatter(x=sd["X"], y=sd["Y"], mode="markers",
                                name="Save", marker=dict(color="green", size=10)))
                        if not gd.empty:
                            fig_rink.add_trace(go.Scatter(x=gd["X"], y=gd["Y"], mode="markers",
                                name="Goal Against", marker=dict(color="blue", size=10)))
                        fig_rink.update_layout(title=f"{goalie_name} — Shootout Ice Map")
                        st.plotly_chart(fig_rink, use_container_width=True)
                    else:
                        st.info("No ice data available")

                with col2:
                    st.subheader("🥅 Shot Locations (Net)")
                    if not gnd.empty:
                        fig_net = create_nhl_goal_net()
                        gd = gnd[gnd["Type"].str.lower() == "goal"]
                        sd = gnd[gnd["Type"].str.lower() == "save"]
                        if not sd.empty:
                            fig_net.add_trace(go.Scatter(x=sd["X"], y=sd["Y"], mode="markers",
                                name="Save", marker=dict(color="green", size=10)))
                        if not gd.empty:
                            fig_net.add_trace(go.Scatter(x=gd["X"], y=gd["Y"], mode="markers",
                                name="Goal Against", marker=dict(color="blue", size=10)))
                        fig_net.update_layout(title=f"{goalie_name} — Shootout Net Map")
                        st.plotly_chart(fig_net, use_container_width=True)
                    else:
                        st.info("No net data available")

            if not gsd_display.empty:
                st.markdown("---")
                st.subheader("📋 Shootout Details — Shots Faced")
                sc = gsd_display.copy()
                for col in ["player","shot_location_ice","shot_location_goal","move_type","goal","date"]:
                    if col in sc.columns:
                        sc[col] = sc[col].astype(str)
                if "date" in sc.columns:
                    sc["date"] = pd.to_datetime(sc["date"], errors="coerce")
                st.dataframe(sc.head(15), hide_index=True, use_container_width=True)

            # ── OPPONENT SHOOTER CLIPS ────────────────────────────────
            st.markdown("---")
            st.markdown("### 📹 Opponent Shooter Clips")
            if not gsd.empty and "player" in gsd.columns:
                shooters = sorted(gsd["player"].dropna().unique().tolist())
                if shooters:
                    sel = st.selectbox("Select opponent shooter:", options=shooters,
                                       key=f"goalie_vid_shooter_{goalie_name}")
                    tr = full_sd[full_sd["player"].str.strip().str.lower() == sel.lower()
                                 ] if not full_sd.empty else pd.DataFrame()
                    team = tr["team"].iloc[0] if not tr.empty else "Unknown"
                    render_video_section(
                        team=team,
                        player_last=sel.split()[-1],
                        scouting_df=full_sd,
                        key_suffix=f"{goalie_name}_{sel}",
                    )
                else:
                    st.caption("No opponent shooters found in scouting data.")
            else:
                st.caption("No opponent scouting data available to match clips against.")

        # ── ALL GOALIE FILM — always renders outside the if/else ──────
        render_goalie_video_section(
    goalie_name=goalie_name,
    scouting_df=full_sd,
    key_suffix=goalie_name.replace(" ", "_"),
)


# ============================================================================
# SESSION STATE
# ============================================================================

def initialize_session_state():
    defaults = {
        'data_loaded': False, 'players_df': pd.DataFrame(), 'goalies_df': pd.DataFrame(),
        'games_df': pd.DataFrame(), 'shots_df': pd.DataFrame(), 'shots_df_goalies': pd.DataFrame(),
        'faceoff_df': pd.DataFrame(), 'shootout_df': pd.DataFrame(),
        'selected_forward': None, 'selected_defenseman': None, 'selected_goalie': None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if 'excluded_players' not in st.session_state:
        st.session_state.excluded_players = load_excluded_players()
    if 'current_roster' not in st.session_state:
        st.session_state.current_roster = load_current_roster()


# ============================================================================
# ROSTER MANAGEMENT
# ============================================================================

def render_roster_management(player_stats, goalie_stats, excluded_players, current_roster):
    with st.expander("⚙️ Manage Roster (Exclude Non-Roster Players)", expanded=False):
        st.markdown('<div class="manage-roster-section">', unsafe_allow_html=True)
        st.markdown("### Exclude players from roster views")
        st.caption("Hide players who have been traded or are no longer with the team.")

        all_players = sorted(set(
            (player_stats['skater'].tolist() if not player_stats.empty else []) +
            (goalie_stats['skater'].tolist()  if not goalie_stats.empty  else [])
        ))
        if not all_players:
            st.info("No players found in current dataset")
            st.markdown('</div>', unsafe_allow_html=True)
            return excluded_players

        not_on_roster = [p for p in all_players if p not in current_roster] if current_roster else []
        col1, col2 = st.columns([2,1])

        with col1:
            st.markdown("**Select players to exclude:**")
            newly_excluded = st.multiselect(
                "Players", options=all_players, default=list(excluded_players),
                label_visibility="collapsed")
            st.caption(f"Currently excluding: {len(newly_excluded)} player(s)")

        with col2:
            st.markdown("**Quick Actions:**")
            if st.button("🔄 Clear All Exclusions", use_container_width=True):
                newly_excluded = []
            if current_roster and not_on_roster:
                if st.button(f"📋 Exclude {len(not_on_roster)} Non-Roster Players",
                             use_container_width=True):
                    newly_excluded = list(set(newly_excluded + not_on_roster))
            if not player_stats.empty:
                zero_gp = player_stats[player_stats['games_played'] == 0]['skater'].tolist()
                if zero_gp:
                    if st.button(f"⚡ Exclude {len(zero_gp)} player(s) with 0 GP",
                                 use_container_width=True):
                        newly_excluded = list(set(newly_excluded + zero_gp))

        if current_roster:
            st.markdown("---")
            c1,c2,c3 = st.columns(3)
            c1.metric("📋 Current Roster", len(current_roster))
            c2.metric("📊 Players in Data", len(all_players))
            c3.metric("⚠️ Not on Roster",   len(not_on_roster))
            if not_on_roster:
                with st.expander(f"View {len(not_on_roster)} players not on current roster"):
                    st.dataframe(pd.DataFrame({"Player": sorted(not_on_roster)}),
                                 hide_index=True, use_container_width=True)
        else:
            st.info("💡 Place 'Crunch_Roster.txt' in 'Crunch_Box_and_Shot' to auto-detect traded players")

        st.markdown("---")
        if st.button("💾 Save Roster Changes", type="primary", use_container_width=True):
            save_excluded_players(set(newly_excluded))
            st.success(f"✅ Saved! Excluding {len(newly_excluded)} player(s).")
            st.rerun()

        if newly_excluded:
            st.markdown("---")
            st.markdown("**Currently Excluded Players:**")
            st.dataframe(pd.DataFrame({"Player": sorted(newly_excluded)}),
                         hide_index=True, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)
        return set(newly_excluded)


# ============================================================================
# MAIN
# ============================================================================

def main():
    initialize_session_state()

    st.markdown('<div class="main-title">🏒 Syracuse Crunch</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Player Scouting Dashboard</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([4,1])
    with col2:
        if st.button("🔄 Reload Data", use_container_width=True):
            st.cache_data.clear()
            st.session_state.clear()
            st.rerun()

    if not st.session_state.data_loaded:
        ph  = st.empty()
        bar = st.progress(0)
        with ph:
            st.info("🔄 Loading Syracuse Crunch data...")
        try:
            players_df, goalies_df, games_df, shots_df, shots_df_goalies = load_all_data()
            bar.progress(50)
            faceoff_df  = load_faceoff_data()
            bar.progress(75)
            shootout_df = load_shootout_data()
            bar.progress(100)

            st.session_state.players_df       = players_df
            st.session_state.goalies_df       = goalies_df
            st.session_state.games_df         = games_df
            st.session_state.shots_df         = shots_df
            st.session_state.shots_df_goalies = shots_df_goalies
            st.session_state.faceoff_df       = faceoff_df
            st.session_state.shootout_df      = shootout_df
            st.session_state.data_loaded      = True

            ph.success(
                f"✅ Loaded: {len(games_df)} games, "
                f"{players_df['skater'].nunique() if not players_df.empty else 0} players, "
                f"{goalies_df['skater'].nunique() if not goalies_df.empty else 0} goalies"
            )
            bar.empty()
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            logging.exception("Data loading error")
            ph.empty(); bar.empty()
            return
    else:
        with st.expander("📊 Data Summary", expanded=False):
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Games", len(st.session_state.games_df))
            c2.metric("Active Players",
                      (st.session_state.players_df['skater'].nunique()
                       if not st.session_state.players_df.empty else 0)
                      - len(st.session_state.excluded_players))
            c3.metric("Goalies",
                      st.session_state.goalies_df['skater'].nunique()
                      if not st.session_state.goalies_df.empty else 0)
            c4.metric("Total Shots",
                      len(st.session_state.shots_df)
                      if not st.session_state.shots_df.empty else 0)
            if st.session_state.excluded_players:
                st.caption(f"ℹ️ {len(st.session_state.excluded_players)} player(s) excluded")

    current_season = (st.session_state.games_df["season"].max()
                      if not st.session_state.games_df.empty else "2024-25")

    if st.session_state.players_df.empty and st.session_state.goalies_df.empty:
        st.warning("⚠️ No Syracuse Crunch data found. Ensure CSV files are in 'Crunch_Box_and_Shot'.")
        return

    player_stats_full = aggregate_player_stats(
        st.session_state.players_df, st.session_state.shots_df, current_season)
    goalie_stats_full = aggregate_goalie_stats(st.session_state.goalies_df, current_season)

    st.markdown("---")
    st.session_state.excluded_players = render_roster_management(
        player_stats_full, goalie_stats_full,
        st.session_state.excluded_players, st.session_state.current_roster)

    player_stats = filter_excluded_players(player_stats_full, st.session_state.excluded_players)
    goalie_stats = filter_excluded_players(goalie_stats_full, st.session_state.excluded_players)

    view_mode = st.radio("", ["👥 Players","🥅 Goalies"],
                         horizontal=True, label_visibility="collapsed")
    st.markdown("---")

    # ── PLAYERS ───────────────────────────────────────────────────────
    if view_mode == "👥 Players":
        if player_stats.empty:
            st.info("No active players available")
            return

        forwards   = player_stats[player_stats['pos'].isin(['C','LW','RW'])].sort_values("points", ascending=False)
        defensemen = player_stats[player_stats['pos'] == 'D'].sort_values("points", ascending=False)
        pt1, pt2   = st.tabs(["⚡ Forwards","🛡️ Defensemen"])

        for tab, group, key, label in [
            (pt1, forwards,   "forward",    "Forward"),
            (pt2, defensemen, "defenseman", "Defenseman"),
        ]:
            with tab:
                if group.empty:
                    st.info(f"No {label.lower()}s available")
                    continue
                opts  = [f"{r['skater']} ({r['pos']}) - {r['points']} PTS"
                         if key == "forward" else f"{r['skater']} - {r['points']} PTS"
                         for _, r in group.iterrows()]
                names = group['skater'].tolist()
                sk    = f"selected_{key}"
                if sk not in st.session_state or st.session_state[sk] not in names:
                    st.session_state[sk] = names[0]
                try:
                    idx = names.index(st.session_state[sk])
                except ValueError:
                    idx = 0
                    st.session_state[sk] = names[0]

                sel_opt  = st.selectbox(f"Select {label}:", options=opts,
                                        index=idx, key=f"{key}_select")
                sel_name = names[opts.index(sel_opt)]
                st.session_state[sk] = sel_name

                row   = group[group["skater"] == sel_name].iloc[0]
                shots = st.session_state.shots_df[
                    st.session_state.shots_df["shooter"] == sel_name
                ].copy() if not st.session_state.shots_df.empty else pd.DataFrame()

                render_player_card(sel_name, row, shots,
                                   st.session_state.faceoff_df,
                                   st.session_state.shootout_df,
                                   st.session_state.games_df)

    # ── GOALIES ───────────────────────────────────────────────────────
    else:
        if goalie_stats.empty:
            st.info("No active goalies available")
            return

        goalie_stats = goalie_stats.sort_values("save_percentage", ascending=False)
        goalie_list  = goalie_stats["skater"].tolist()
        goalie_opts  = [f"{r['skater']} - SV% {r['save_percentage']:.3f}"
                        for _, r in goalie_stats.iterrows()]

        if ('selected_goalie' not in st.session_state or
                st.session_state.selected_goalie not in goalie_list):
            st.session_state.selected_goalie = goalie_list[0] if goalie_list else None

        try:
            idx = goalie_list.index(st.session_state.selected_goalie)
        except (ValueError, AttributeError):
            idx = 0
            st.session_state.selected_goalie = goalie_list[0] if goalie_list else None

        if st.session_state.selected_goalie:
            sel_opt    = st.selectbox("Select Goalie:", options=goalie_opts,
                                      index=idx, key="goalie_select")
            sel_goalie = goalie_list[goalie_opts.index(sel_opt)]
            st.session_state.selected_goalie = sel_goalie

            goalie_row   = goalie_stats[goalie_stats["skater"] == sel_goalie].iloc[0]
            goalie_shots = st.session_state.shots_df_goalies[
                st.session_state.shots_df_goalies["goalie"] == sel_goalie
            ].copy() if not st.session_state.shots_df_goalies.empty else pd.DataFrame()

            render_goalie_card(sel_goalie, goalie_row, goalie_shots,
                               st.session_state.shootout_df,
                               st.session_state.games_df)


if __name__ == "__main__":
    main()

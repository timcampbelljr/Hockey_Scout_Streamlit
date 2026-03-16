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
import glob
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

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True)
CRUNCH_DATA_DIR.mkdir(exist_ok=True)
SCREEN_RECORDINGS_DIR.mkdir(exist_ok=True)

# Target team
TARGET_TEAM = "Syracuse Crunch"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ============================================================================
# PLAYER EXCLUSION MANAGEMENT
# ============================================================================

def load_current_roster():
    """Load current roster from Crunch_Roster.txt file."""
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
    """Load list of excluded players from JSON file."""
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
    """Save list of excluded players to JSON file."""
    try:
        with open(EXCLUDED_PLAYERS_FILE, 'w') as f:
            json.dump({'excluded_players': list(excluded_set)}, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving excluded players: {e}")


def filter_excluded_players(df, excluded_players):
    """Filter out excluded players from dataframe."""
    if df.empty or not excluded_players:
        return df
    return df[~df['skater'].isin(excluded_players)].copy()


# ============================================================================
# XG MODEL FUNCTIONS
# ============================================================================

def calculate_shot_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate distance and angle from coordinates."""
    if df.empty:
        return df
    df = df.copy()
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    goal_y = 200
    goal_left_x = 50
    goal_right_x = 798

    d1 = np.sqrt((df["x"] - goal_left_x) ** 2 + (df["y"] - goal_y) ** 2)
    d2 = np.sqrt((df["x"] - goal_right_x) ** 2 + (df["y"] - goal_y) ** 2)
    df["distance"] = np.minimum(d1, d2)

    dx = np.minimum(np.abs(df["x"] - goal_left_x), np.abs(df["x"] - goal_right_x))
    dx = dx.replace(0, 0.1)
    dy = np.abs(df["y"] - goal_y)
    df["angle"] = np.arctan(dy / dx) * (180 / np.pi)
    df["abs_y"] = dy
    df["is_slot"] = ((df["distance"] < 100) & (dy < 40)).astype(int)

    return df


def estimate_xg_simple(distance: float, angle: float) -> float:
    """Simple heuristic xG based on distance and angle."""
    if pd.isna(distance) or distance < 0:
        dist_factor = 1.0
    else:
        dist_factor = 0.8 * np.exp(-0.05 * distance)

    if pd.isna(angle):
        angle_factor = 1.0
    else:
        angle_factor = 1.0 - abs(angle) / 90.0
        angle_factor = max(0.1, angle_factor)

    prob = dist_factor * angle_factor
    return float(np.clip(prob, 0.01, 0.99))


def predict_xg_batch(df: pd.DataFrame) -> pd.Series:
    """Predict xG for a dataframe of shots."""
    if df.empty:
        return pd.Series(dtype=float)
    return df.apply(
        lambda row: estimate_xg_simple(row.get("distance", 50), row.get("angle", 0)),
        axis=1,
    )


# ============================================================================
# DATA LOADING
# ============================================================================

def determine_season_from_game_id(game_id: int) -> str:
    """Determine season from game ID."""
    game_id_str = str(game_id)
    if game_id_str.startswith("10") and len(game_id_str) >= 5:
        try:
            season_code = int(game_id_str[2:4])
            if season_code == 28:
                season_code = 27
            start_year = season_code + 1997
            end_year_short = (start_year + 1) % 100
            return f"{start_year}-{end_year_short:02d}"
        except (ValueError, IndexError):
            pass
    if game_id > 1024000:
        return "2023-24"
    return "2024-25"


@st.cache_data
def load_all_data():
    """Load all hockey data and filter for Syracuse Crunch only."""
    def get_unique_files(pattern):
        upload_files = list(UPLOAD_DIR.glob(pattern)) if UPLOAD_DIR.exists() else []
        assets_files = list(ASSETS_DIR.glob(pattern)) if ASSETS_DIR.exists() else []
        crunch_files = list(CRUNCH_DATA_DIR.glob(pattern)) if CRUNCH_DATA_DIR.exists() else []
        all_files = {}
        for f in upload_files + assets_files + crunch_files:
            all_files[f.name] = f
        return sorted(all_files.values(), key=lambda x: x.name)

    boxscore_files = get_unique_files("ahl_boxscore_*.csv")
    shot_files = get_unique_files("ahl_shots_*.csv")

    logging.info(f"Found {len(boxscore_files)} boxscore files")
    logging.info(f"Found {len(shot_files)} shot files")

    if boxscore_files:
        game_ids = [f.stem.replace("ahl_boxscore_", "") for f in boxscore_files]
        logging.info(f"Boxscore game IDs: {min(game_ids)} to {max(game_ids)}")
    if shot_files:
        game_ids = [f.stem.replace("ahl_shots_", "") for f in shot_files]
        logging.info(f"Shot file game IDs: {min(game_ids)} to {max(game_ids)}")

    all_shots = []
    for f in shot_files:
        try:
            all_shots.append(pd.read_csv(f))
        except Exception as e:
            logging.exception(f"Error processing shot file {f}: {e}")

    shot_df = pd.DataFrame()
    if all_shots:
        shot_df = pd.concat(all_shots).drop_duplicates()
        shot_df["shooter"] = (
            shot_df["shooter_first"].fillna("") + " " + shot_df["shooter_last"].fillna("")
        ).str.strip()
        shot_df["goalie"] = (
            shot_df["goalie_first"].fillna("") + " " + shot_df["goalie_last"].fillna("")
        ).str.strip()
        shot_df["game_id"] = shot_df["game_id"].astype(int)
        shot_df["shooter_team_id"] = pd.to_numeric(shot_df["shooter_team_id"], errors="coerce")
        shot_df["x"] = pd.to_numeric(shot_df["x"], errors="coerce")
        shot_df["y"] = pd.to_numeric(shot_df["y"], errors="coerce")
        shot_df["x"] = shot_df["x"] * (848 / 600)
        shot_df["y"] = shot_df["y"] * (400 / 300)
        shot_df["is_goal"] = shot_df["is_goal"].astype(bool)

        if not shot_df.empty:
            try:
                shot_df = calculate_shot_features(shot_df)
                shot_df["xg"] = predict_xg_batch(shot_df)
                shot_df["xg"] = shot_df["xg"].round(3)
                shot_df["distance"] = shot_df["distance"].round(1)
            except Exception as e:
                logging.exception(f"Error in xG calculation: {e}")
                shot_df["xg"] = 0.0

    all_players = []
    all_goalies = []
    all_games = []
    game_team_ids = []

    for f in boxscore_files:
        try:
            df = pd.read_csv(f)
            if df.empty:
                continue

            game_id = int(df["game_id"].iloc[0])
            season = determine_season_from_game_id(game_id)

            home_team = df[df["team_side"] == "home"]["team_name"].iloc[0]
            away_team = df[df["team_side"] == "away"]["team_name"].iloc[0]
            home_team_id = int(df[df["team_side"] == "home"]["team_id"].iloc[0])
            away_team_id = int(df[df["team_side"] == "away"]["team_id"].iloc[0])

            game_info = {
                "game_id": game_id,
                "home_team": home_team,
                "away_team": away_team,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "season": season,
            }
            all_games.append(game_info)

            crunch_row = df[df["team_name"] == TARGET_TEAM]
            if not crunch_row.empty:
                crunch_team_id = int(crunch_row["team_id"].iloc[0])
                opponent_team_id = away_team_id if crunch_team_id == home_team_id else home_team_id
                game_team_ids.append({
                    "game_id": game_id,
                    "crunch_team_id": crunch_team_id,
                    "opponent_team_id": opponent_team_id
                })

            df["is_goalie"] = df["pos"] == "G"
            df["season"] = season

            players_df = df[~df["is_goalie"]].copy()
            goalies_df = df[df["is_goalie"]].copy()

            players_df = players_df.rename(
                columns={"g": "goals", "a": "assists", "pim": "penalty_minutes", "sog": "shots"}
            )
            goalies_df = goalies_df.rename(
                columns={"svs": "saves", "ga": "goals_against", "mins": "minutes_played"}
            )

            all_players.append(players_df)
            all_goalies.append(goalies_df)

        except Exception as e:
            logging.exception(f"Error processing boxscore file {f}: {e}")

    players_df = pd.concat(all_players) if all_players else pd.DataFrame()
    goalies_df = pd.concat(all_goalies) if all_goalies else pd.DataFrame()

    if not players_df.empty:
        numeric_cols = ["goals", "assists", "penalty_minutes", "plus_minus", "shots"]
        agg_dict = {col: "sum" for col in numeric_cols}
        agg_dict.update({
            "pos": "first", "team_name": "first", "team_id": "first",
            "team_side": "first", "season": "first", "number": "first"
        })
        players_df = players_df.groupby(["game_id", "skater"], as_index=False).agg(agg_dict)

    if not goalies_df.empty:
        numeric_cols = ["saves", "goals_against"]
        agg_dict = {col: "sum" for col in numeric_cols}
        agg_dict.update({
            "pos": "first", "team_name": "first", "team_id": "first",
            "team_side": "first", "season": "first", "number": "first",
            "minutes_played": "first"
        })
        goalies_df = goalies_df.groupby(["game_id", "skater"], as_index=False).agg(agg_dict)

    games_df = pd.DataFrame(all_games).drop_duplicates(subset=["game_id"]) if all_games else pd.DataFrame()
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
                crunch_players = players_df["skater"].unique()
                shot_df_players = shot_df_players[shot_df_players["shooter"].isin(crunch_players)]

            shot_df_goalies = shot_df[shot_df["shooter_team_id"] == shot_df["opponent_team_id"]].copy()

            if not goalies_df.empty:
                crunch_goalies = goalies_df["skater"].unique()
                shot_df_goalies = shot_df_goalies[shot_df_goalies["goalie"].isin(crunch_goalies)]

            logging.info(f"Player shots (by Crunch): {len(shot_df_players)}")
            logging.info(f"Goalie shots (against Crunch): {len(shot_df_goalies)}")
        else:
            if not goalies_df.empty:
                crunch_goalies = goalies_df["skater"].unique()
                shot_df_goalies = shot_df[shot_df["goalie"].isin(crunch_goalies)].copy()
                logging.info(f"Using fallback goalie filtering: {len(shot_df_goalies)} shots")

    return players_df, goalies_df, games_df, shot_df_players, shot_df_goalies


@st.cache_data
def load_faceoff_data():
    """Load faceoff data."""
    try:
        faceoff_files = (
            list(ASSETS_DIR.glob("Faceoffs*.csv")) +
            list(CRUNCH_DATA_DIR.glob("Faceoffs*.csv"))
        ) if ASSETS_DIR.exists() or CRUNCH_DATA_DIR.exists() else []

        if not faceoff_files:
            return pd.DataFrame()

        df = pd.read_csv(faceoff_files[0])
        df.columns = df.columns.str.lower()
        df = df.rename(columns={"name": "player"})
        df = df.dropna(subset=["player"])

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
    """Load the specific Shootout_Scouting(Crunch SO).csv file."""
    try:
        for directory in [CRUNCH_DATA_DIR, ASSETS_DIR, UPLOAD_DIR, Path(".")]:
            if directory.exists():
                shootout_file = directory / "Shootout_Scouting(Crunch SO).csv"
                if shootout_file.exists():
                    logging.info(f"Loading shootout scouting file: {shootout_file}")

                    df = None
                    for encoding in ['utf-8', 'cp1252', 'latin-1']:
                        try:
                            df = pd.read_csv(
                                shootout_file,
                                encoding=encoding,
                                on_bad_lines='skip'
                            )
                            logging.info(f"Loaded with {encoding} encoding")
                            break
                        except Exception as e:
                            logging.info(f"Failed with {encoding}: {e}")
                            continue

                    if df is not None and not df.empty:
                        df.columns = df.columns.str.strip().str.lower()

                        column_mapping = {
                            'where player shot from on ice': 'shot_location_ice',
                            'where the shot went on goal': 'shot_location_goal',
                            'what move they made': 'move_type',
                            'goalie (don\'t worry about this)': 'goalie',
                            "goalie (don't worry about this)": 'goalie',
                        }
                        df = df.rename(columns=column_mapping)

                        df["player"] = df["player"].fillna("").astype(str).str.strip()
                        df["team"] = df["team"].fillna("").astype(str).str.strip() if 'team' in df.columns else ""

                        df = df[
                            df["player"].notna() &
                            (df["player"] != "") &
                            (df["player"].str.lower() != "idle")
                        ].copy()

                        df["goal"] = df["goal"].fillna("No")
                        df["goal"] = df["goal"].apply(
                            lambda x: "Yes" if str(x).strip().lower() in ["yes", "y", "goal", "1", "true"] else "No"
                        )

                        logging.info(f"Loaded {len(df)} shootout records")
                        return df

        logging.warning("Shootout_Scouting(Crunch SO).csv not found")
        return pd.DataFrame()

    except Exception as e:
        logging.error(f"Error loading shootout scouting data: {e}")
        return pd.DataFrame()


# ============================================================================
# VIDEO LOOKUP FUNCTIONS
# ============================================================================

def normalize_date_to_slug(date_str: str):
    """Convert any date string to MM-DD-YYYY for filename matching."""
    try:
        parsed = pd.to_datetime(date_str, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.strftime("%m-%d-%Y")
    except Exception:
        return None


def find_shootout_video(team: str, shooter_last: str, goalie_full: str, date_str: str):
    """
    Locate an MP4 in 'Screen Recordings/' matching:
    Team.ShooterLastName_On_GoalieLastName.MM-DD-YYYY.mp4

    Falls back to case-insensitive match if exact match fails.
    Returns Path if found, None otherwise.
    """
    if not SCREEN_RECORDINGS_DIR.exists():
        return None

    date_slug = normalize_date_to_slug(date_str)
    if not date_slug:
        return None

    team_slug = team.strip().replace(" ", "")
    shooter_slug = shooter_last.strip().split()[-1]
    goalie_slug = goalie_full.strip().split()[-1]  # last name only

    expected_stem = f"{team_slug}.{shooter_slug}_On_{goalie_slug}.{date_slug}"
    expected_file = SCREEN_RECORDINGS_DIR / f"{expected_stem}.mp4"

    # Exact match
    if expected_file.exists():
        return expected_file

    # Case-insensitive fallback
    try:
        for f in SCREEN_RECORDINGS_DIR.iterdir():
            if f.suffix.lower() == ".mp4" and f.stem.lower() == expected_stem.lower():
                return f
    except Exception:
        pass

    return None


def get_videos_for_shooter(team: str, shooter_last: str, scouting_df: pd.DataFrame) -> list:
    """
    Scan scouting CSV rows for a shooter and return all attempts that
    have a matching video file, as a list of dicts sorted newest-first:
      { label, path, date_raw, outcome, goalie }
    """
    results = []
    if scouting_df.empty:
        return results

    player_rows = scouting_df[
        scouting_df["player"].str.strip().str.lower() == shooter_last.strip().lower()
    ]

    for _, row in player_rows.iterrows():
        goalie_raw = str(row.get("goalie", "")).strip()
        date_raw = str(row.get("date", "")).strip()
        goal_val = str(row.get("goal", "No")).strip()
        team_val = str(row.get("team", team)).strip()

        # Skip rows with no goalie or no date — can't build a filename
        if goalie_raw in ("", "nan") or date_raw in ("", "nan"):
            continue

        video_path = find_shootout_video(team_val, shooter_last, goalie_raw, date_raw)
        if video_path is None:
            continue

        outcome = "✅ GOAL" if goal_val.lower() == "yes" else "❌ No Goal"
        goalie_last = goalie_raw.split()[-1]

        try:
            label_date = pd.to_datetime(date_raw).strftime("%b %d, %Y")
        except Exception:
            label_date = date_raw

        results.append({
            "label": f"{label_date} — vs {goalie_last} — {outcome}",
            "path": video_path,
            "date_raw": date_raw,
            "outcome": outcome,
            "goalie": goalie_raw,
        })

    # Newest attempt first in dropdown
    results.sort(
        key=lambda x: pd.to_datetime(x["date_raw"], errors="coerce"),
        reverse=True
    )
    return results


def render_video_section(team: str, player_last: str, scouting_df: pd.DataFrame, key_suffix: str = ""):
    """
    Renders the Video Clips block below shootout charts.
    Shows a date-based dropdown when multiple clips exist for the player.
    """
    st.markdown("---")
    st.markdown("### 📹 Video Clips")

    videos = get_videos_for_shooter(team, player_last, scouting_df)

    if not videos:
        st.info(
            f"No MP4 clips found for **{player_last}**. "
            f"Add files to `Screen Recordings/` using the format:  \n"
            f"`{team}.{player_last}_On_GoalieName.MM-DD-YYYY.mp4`"
        )
        return

    # Single clip — no dropdown needed
    if len(videos) == 1:
        v = videos[0]
        st.caption(v["label"])
        st.video(str(v["path"]))
        return

    # Multiple clips — dropdown sorted newest first
    options = [v["label"] for v in videos]
    selected_label = st.selectbox(
        f"Select attempt ({len(videos)} clips available):",
        options=options,
        index=0,  # default = most recent
        key=f"vid_sel_{player_last}_{key_suffix}"
    )
    chosen = next(v for v in videos if v["label"] == selected_label)
    st.video(str(chosen["path"]))


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_shot_chart(df, player_name, view_type="player"):
    """Create interactive shot chart with coordinate validation."""
    if df.empty:
        return None

    if 'x' not in df.columns or 'y' not in df.columns:
        logging.warning(f"Missing x/y coordinates for {player_name}")
        return None

    df = df.dropna(subset=['x', 'y'])
    df = df[(df['x'] >= 0) & (df['x'] <= 848) & (df['y'] >= 0) & (df['y'] <= 400)]

    if df.empty:
        logging.warning(f"No valid coordinates for {player_name}")
        return None

    fig = go.Figure()

    fig.add_shape(
        type="rect", x0=0, y0=0, x1=848, y1=400,
        line=dict(color="#000000", width=3),
        fillcolor="rgba(0,0,0,0)"
    )
    fig.add_shape(type="line", x0=50, y0=0, x1=50, y1=400, line=dict(color="red", width=3))
    fig.add_shape(type="line", x0=798, y0=0, x1=798, y1=400, line=dict(color="red", width=3))
    fig.add_shape(type="line", x0=274, y0=0, x1=274, y1=400, line=dict(color="blue", width=2))
    fig.add_shape(type="line", x0=574, y0=0, x1=574, y1=400, line=dict(color="blue", width=2))
    fig.add_shape(type="line", x0=424, y0=0, x1=424, y1=400, line=dict(color="red", width=3))

    goal_width = 60
    goal_height = 15

    fig.add_shape(
        type="rect",
        x0=50 - goal_height, y0=200 - goal_width / 2,
        x1=50, y1=200 + goal_width / 2,
        line=dict(color="red", width=2),
        fillcolor="rgba(255,0,0,0.1)"
    )
    fig.add_shape(
        type="rect",
        x0=798, y0=200 - goal_width / 2,
        x1=798 + goal_height, y1=200 + goal_width / 2,
        line=dict(color="red", width=2),
        fillcolor="rgba(255,0,0,0.1)"
    )

    if view_type == "player":
        saves = df[~df["is_goal"]]
        if not saves.empty:
            fig.add_trace(go.Scatter(
                x=saves["x"], y=saves["y"],
                mode='markers',
                marker=dict(size=8, color='#3b82f6', symbol='circle', line=dict(width=1, color='#1e3a8a')),
                name='Shot',
                hovertemplate='<b>Shot</b><br>xG: %{customdata:.2%}<extra></extra>',
                customdata=saves.get('xg', 0)
            ))

        goals = df[df["is_goal"]]
        if not goals.empty:
            fig.add_trace(go.Scatter(
                x=goals["x"], y=goals["y"],
                mode='markers',
                marker=dict(size=14, color='#10b981', symbol='star', line=dict(width=2, color='#065f46')),
                name='Goal',
                hovertemplate='<b>GOAL</b><br>xG: %{customdata:.2%}<extra></extra>',
                customdata=goals.get('xg', 0)
            ))
    else:
        if not df.empty:
            fig.add_trace(go.Scatter(
                x=df["x"], y=df["y"],
                mode='markers',
                marker=dict(size=10, color='#ef4444', symbol='x', line=dict(width=2, color='#991b1b')),
                name='Goal Against',
                text=df.get('shooter', 'Unknown'),
                hovertemplate='<b>Goal Against</b><br>%{text}<br>xG: %{customdata:.2%}<extra></extra>',
                customdata=df.get('xg', 0)
            ))

    fig.update_layout(
        title=f"{player_name} - Shot Chart",
        xaxis=dict(range=[0, 848], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[0, 400], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
        plot_bgcolor='#f8f9fa',
        height=400,
        hovermode='closest',
        showlegend=True,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig


def create_nhl_rink_shootout():
    """Create NHL rink for shootout visualization."""
    fig = go.Figure()

    fig.add_shape(
        type="rect", x0=-100, y0=-42.5, x1=100, y1=42.5,
        line=dict(color="black", width=3), fillcolor="white", layer="below"
    )
    fig.add_shape(
        type="line", x0=0, y0=-42.5, x1=0, y1=42.5,
        line=dict(color="red", width=3), layer="below"
    )
    for x in [-25, 25]:
        fig.add_shape(
            type="line", x0=x, y0=-42.5, x1=x, y1=42.5,
            line=dict(color="blue", width=2), layer="below"
        )
    for x in [-89, 89]:
        fig.add_shape(
            type="line", x0=x, y0=-42.5, x1=x, y1=42.5,
            line=dict(color="red", width=2), layer="below"
        )

    for y_center in [-22, 22]:
        fig.add_shape(
            type="circle", x0=-69 - 15, y0=y_center - 15, x1=-69 + 15, y1=y_center + 15,
            line=dict(color="red", width=2), fillcolor="rgba(255,0,0,0.1)", layer="below"
        )
        fig.add_shape(
            type="circle", x0=-69 - 1, y0=y_center - 1, x1=-69 + 1, y1=y_center + 1,
            fillcolor="red", line=dict(color="red", width=1), layer="below"
        )
    for y_center in [-22, 22]:
        fig.add_shape(
            type="circle", x0=69 - 15, y0=y_center - 15, x1=69 + 15, y1=y_center + 15,
            line=dict(color="red", width=2), fillcolor="rgba(255,0,0,0.1)", layer="below"
        )
        fig.add_shape(
            type="circle", x0=69 - 1, y0=y_center - 1, x1=69 + 1, y1=y_center + 1,
            fillcolor="red", line=dict(color="red", width=1), layer="below"
        )

    fig.add_shape(
        type="circle", x0=-15, y0=-15, x1=15, y1=15,
        line=dict(color="blue", width=2), fillcolor="rgba(0,0,255,0.05)", layer="below"
    )
    fig.add_shape(
        type="circle", x0=-1, y0=-1, x1=1, y1=1,
        fillcolor="blue", line=dict(color="blue", width=1), layer="below"
    )

    crease_path_left = "M -89 -4 L -89 4 L -85 4.5 Q -83 4.5 -83 3 L -83 -3 Q -83 -4.5 -85 -4.5 L -89 -4 Z"
    fig.add_shape(
        type="path", path=crease_path_left,
        line=dict(color="red", width=2), fillcolor="rgba(173,216,230,0.4)", layer="below"
    )
    crease_path_right = "M 89 -4 L 89 4 L 85 4.5 Q 83 4.5 83 3 L 83 -3 Q 83 -4.5 85 -4.5 L 89 -4 Z"
    fig.add_shape(
        type="path", path=crease_path_right,
        line=dict(color="red", width=2), fillcolor="rgba(173,216,230,0.4)", layer="below"
    )

    fig.add_shape(
        type="rect", x0=-92, y0=-3, x1=-89, y1=3,
        line=dict(color="red", width=2), fillcolor="rgba(255,255,255,0.3)", layer="below"
    )
    fig.add_shape(
        type="rect", x0=89, y0=-3, x1=92, y1=3,
        line=dict(color="red", width=2), fillcolor="rgba(255,255,255,0.3)", layer="below"
    )

    fig.update_layout(
        showlegend=True,
        xaxis=dict(range=[-105, 105], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-45, 45], showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
        plot_bgcolor='#f0f8ff',
        height=400,
        margin=dict(l=10, r=10, t=30, b=10)
    )

    return fig


def create_nhl_goal_net():
    """Create NHL goal net visualization."""
    fig = go.Figure()

    fig.add_shape(
        type="rect", x0=-36, y0=-24, x1=36, y1=24,
        line=dict(color="red", width=3), fillcolor="rgba(255,255,255,0.3)"
    )

    post_radius = 2.4
    for x, y in [(-36, -24), (-36, 24), (36, -24), (36, 24)]:
        fig.add_shape(
            type="circle",
            x0=x - post_radius, y0=y - post_radius,
            x1=x + post_radius, y1=y + post_radius,
            fillcolor="red", line=dict(color="darkred", width=1)
        )

    for x in [-12, 12]:
        fig.add_shape(type="line", x0=x, y0=-24, x1=x, y1=24, line=dict(color="gray", width=1, dash="dash"))
    for y in [-8, 8]:
        fig.add_shape(type="line", x0=-36, y0=y, x1=36, y1=y, line=dict(color="gray", width=1, dash="dash"))

    zones = [
        (-24, -16, "Bottom\nLeft"), (0, -16, "Bottom\nCenter"), (24, -16, "Bottom\nRight"),
        (-24, 0, "Middle\nLeft"), (0, 0, "Five\nHole"), (24, 0, "Middle\nRight"),
        (-24, 16, "Top\nLeft"), (0, 16, "Top\nCenter"), (24, 16, "Top\nRight"),
    ]
    for x, y, label in zones:
        fig.add_annotation(x=x, y=y, text=label, showarrow=False, font=dict(size=9, color="gray"), opacity=0.6)

    fig.update_layout(
        showlegend=True,
        xaxis=dict(range=[-45, 45], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-30, 30], showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
        plot_bgcolor='white',
        height=350,
        margin=dict(l=10, r=10, t=30, b=10)
    )

    return fig


def get_net_zone(x, y):
    """Determine which zone of the net the shot went to."""
    if pd.isna(x) or pd.isna(y):
        return "Unknown"

    if x < -12:
        horizontal = "Left"
    elif x > 12:
        horizontal = "Right"
    else:
        horizontal = "Center"

    if y < -8:
        vertical = "Bottom"
    elif y > 8:
        vertical = "Top"
    else:
        vertical = "Middle"

    if horizontal == "Center" and vertical == "Middle":
        return "Five Hole"

    return f"{vertical} {horizontal}"


# ============================================================================
# AGGREGATION FUNCTIONS
# ============================================================================

def aggregate_player_stats(players_df, shots_df, season="2024-25"):
    """Aggregate player statistics."""
    if players_df.empty:
        return pd.DataFrame()

    season_df = players_df[players_df["season"] == season].copy()
    if season_df.empty:
        return pd.DataFrame()

    numeric_cols = ["goals", "assists", "penalty_minutes", "plus_minus", "shots"]
    for col in numeric_cols:
        season_df[col] = pd.to_numeric(season_df[col], errors="coerce").fillna(0)

    season_df["points"] = season_df["goals"] + season_df["assists"]

    agg_df = (
        season_df.groupby("skater")
        .agg(
            pos=("pos", "first"),
            games_played=("game_id", "nunique"),
            goals=("goals", "sum"),
            assists=("assists", "sum"),
            points=("points", "sum"),
            plus_minus=("plus_minus", "sum"),
            penalty_minutes=("penalty_minutes", "sum"),
            shots=("shots", "sum"),
        )
        .reset_index()
    )

    if not shots_df.empty and "xg" in shots_df.columns and "shooter" in shots_df.columns:
        xg_stats = shots_df.groupby("shooter")["xg"].mean().reset_index()
        xg_stats.rename(columns={"shooter": "skater", "xg": "avg_xg"}, inplace=True)
        agg_df = pd.merge(agg_df, xg_stats, on="skater", how="left")
        agg_df["avg_xg"] = agg_df["avg_xg"].fillna(0).round(3)
    else:
        agg_df["avg_xg"] = 0.0

    for col in ["games_played", "goals", "assists", "points", "plus_minus", "penalty_minutes", "shots"]:
        agg_df[col] = agg_df[col].astype(int)

    return agg_df


def aggregate_goalie_stats(goalies_df, season="2024-25"):
    """Aggregate goalie statistics."""
    if goalies_df.empty:
        return pd.DataFrame()

    season_df = goalies_df[goalies_df["season"] == season].copy()
    if season_df.empty:
        return pd.DataFrame()

    numeric_cols = ["saves", "goals_against", "minutes_played"]
    for col in numeric_cols:
        season_df[col] = pd.to_numeric(season_df[col], errors="coerce").fillna(0)

    season_df["played_game"] = season_df["saves"] > 0
    games_played = (
        season_df[season_df["played_game"]]
        .groupby("skater")["game_id"]
        .nunique()
        .reset_index(name="games_played")
    )

    agg_df = (
        season_df.groupby("skater")
        .agg(saves=("saves", "sum"), goals_against=("goals_against", "sum"))
        .reset_index()
    )

    if not games_played.empty:
        agg_df = pd.merge(agg_df, games_played, on="skater", how="left")
        agg_df["games_played"] = agg_df["games_played"].fillna(0).astype(int)
    else:
        agg_df["games_played"] = 0

    agg_df["save_percentage"] = agg_df.apply(
        lambda row: row["saves"] / (row["saves"] + row["goals_against"])
        if row["saves"] + row["goals_against"] > 0 else 0,
        axis=1,
    )
    agg_df["goals_against_average"] = agg_df.apply(
        lambda row: row["goals_against"] / row["games_played"] if row["games_played"] > 0 else 0,
        axis=1,
    )

    for col in ["saves", "goals_against"]:
        agg_df[col] = agg_df[col].astype(int)

    agg_df["save_percentage"] = agg_df["save_percentage"].round(3)
    agg_df["goals_against_average"] = agg_df["goals_against_average"].round(2)

    return agg_df


# ============================================================================
# PLAYER CARD COMPONENTS
# ============================================================================

def render_player_card(player_name, player_stats, player_shots, faceoff_data, shootout_data, games_df):
    """Render a complete player card with all stats."""
    st.markdown(f"""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;'>
            <h1 style='color: white; margin: 0;'>{player_name}</h1>
            <p style='color: rgba(255,255,255,0.9); margin: 5px 0 0 0; font-size: 1.1em;'>{player_stats['pos']} • Syracuse Crunch</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
    with col1:
        st.metric("GP", player_stats['games_played'])
    with col2:
        st.metric("G", player_stats['goals'])
    with col3:
        st.metric("A", player_stats['assists'])
    with col4:
        st.metric("PTS", player_stats['points'])
    with col5:
        st.metric("+/-", player_stats['plus_minus'])
    with col6:
        st.metric("PIM", player_stats['penalty_minutes'])
    with col7:
        st.metric("SOG", player_stats['shots'])
    with col8:
        st.metric("Avg xG", f"{player_stats['avg_xg']:.3f}")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Box Score", "🎯 Shot Chart", "🥅 Shootout", "⚔️ Faceoffs"])

    # ------------------------------------------------------------------
    # TAB 1: BOX SCORE
    # ------------------------------------------------------------------
    with tab1:
        st.markdown('<div class="stat-card"><h3>Game-by-Game Stats</h3></div>', unsafe_allow_html=True)

        player_games = st.session_state.players_df[
            st.session_state.players_df["skater"] == player_name
        ].copy()

        if not player_games.empty and not games_df.empty:
            player_games = player_games.merge(games_df, on="game_id", suffixes=("", "_game"))
            player_games["opponent"] = player_games.apply(
                lambda row: row["away_team"] if row["team_name"] == row["home_team"] else row["home_team"],
                axis=1
            )
            player_games["points"] = player_games["goals"] + player_games["assists"]
            player_games = player_games.sort_values("game_id")
            player_games["game_number"] = range(1, len(player_games) + 1)

            display_df = player_games[
                ["game_number", "opponent", "goals", "assists", "points", "plus_minus", "penalty_minutes", "shots"]
            ].iloc[::-1]

            st.dataframe(
                display_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "game_number": "Game #", "opponent": "Opponent", "goals": "G",
                    "assists": "A", "points": "PTS", "plus_minus": "+/-",
                    "penalty_minutes": "PIM", "shots": "SOG",
                },
            )
        else:
            st.info("No game data available")

    # ------------------------------------------------------------------
    # TAB 2: SHOT CHART
    # ------------------------------------------------------------------
    with tab2:
        st.markdown('<div class="stat-card"><h3>Shot Chart</h3></div>', unsafe_allow_html=True)

        if not player_shots.empty:
            available_games = sorted(player_shots["game_id"].unique())
            game_lookup = {}
            player_games = st.session_state.players_df[
                st.session_state.players_df["skater"] == player_name
            ].copy()

            if not player_games.empty and not games_df.empty:
                player_games = player_games.merge(games_df, on="game_id", suffixes=("", "_game"))
                player_games["opponent"] = player_games.apply(
                    lambda row: row["away_team"] if row["team_name"] == row["home_team"] else row["home_team"],
                    axis=1
                )
                player_games = player_games.sort_values("game_id")
                player_games["game_number"] = range(1, len(player_games) + 1)
                for _, row in player_games.iterrows():
                    game_lookup[row["game_id"]] = f"Game {row['game_number']}: {row['opponent']}"

            col1, col2 = st.columns([3, 1])
            with col1:
                game_filter_option = st.radio(
                    "Show shots from:",
                    ["All Games", "Single Game"],
                    horizontal=True,
                    key=f"player_game_filter_{player_name}"
                )

            filtered_shots = player_shots.copy()
            if game_filter_option == "Single Game":
                with col2:
                    selected_game = st.selectbox(
                        "Select Game:",
                        available_games,
                        index=len(available_games) - 1,
                        format_func=lambda x: game_lookup.get(x, f"Game {x}"),
                        key=f"player_single_game_{player_name}"
                    )
                filtered_shots = player_shots[player_shots["game_id"] == selected_game]

            st.markdown("---")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Shots", len(filtered_shots))
            goals = (filtered_shots["is_goal"] == True).sum()
            col2.metric("Goals", goals)
            col3.metric(
                "Shooting %",
                f"{(goals / len(filtered_shots)) * 100:.1f}%" if len(filtered_shots) > 0 else "0.0%"
            )
            avg_xg = (
                filtered_shots["xg"].mean()
                if "xg" in filtered_shots.columns and len(filtered_shots) > 0 else 0
            )
            col4.metric("Avg xG", f"{avg_xg:.3f}")

            fig = create_shot_chart(filtered_shots, player_name, view_type="player")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No shot data available for this player")

    # ------------------------------------------------------------------
    # TAB 3: SHOOTOUT
    # ------------------------------------------------------------------
    with tab3:
        st.markdown('<div class="section-header">🥅 Shootout Performance</div>', unsafe_allow_html=True)

        shootout_ice_data = pd.DataFrame()
        shootout_net_data = pd.DataFrame()

        try:
            ice_file = CRUNCH_DATA_DIR / "Crunch25-26SO.csv"
            if ice_file.exists():
                shootout_ice_data = pd.read_csv(ice_file)
                shootout_ice_data.columns = shootout_ice_data.columns.str.strip()
        except Exception as e:
            logging.warning(f"Could not load shootout ice location data: {e}")

        try:
            net_file = CRUNCH_DATA_DIR / "SO_Goalzone.csv"
            if net_file.exists():
                shootout_net_data = pd.read_csv(net_file)
                shootout_net_data.columns = shootout_net_data.columns.str.strip()
        except Exception as e:
            logging.warning(f"Could not load shootout net location data: {e}")

        # Match player in scouting data
        player_scouting_data = pd.DataFrame()
        if not shootout_data.empty:
            player_scouting_data = shootout_data[shootout_data["player"] == player_name]
            if player_scouting_data.empty and " " in player_name:
                last_name = player_name.split()[-1]
                player_scouting_data = shootout_data[
                    shootout_data["player"].str.lower() == last_name.lower()
                ]

        has_ice_data = not shootout_ice_data.empty
        has_net_data = not shootout_net_data.empty
        has_scouting_data = not player_scouting_data.empty

        if not has_ice_data and not has_scouting_data:
            st.info("No shootout data available for this player")
        else:
            if " " in player_name:
                name_parts = player_name.split()
                last_name = name_parts[-1]
                first_name = name_parts[0]
            else:
                last_name = player_name
                first_name = player_name

            player_ice_data = pd.DataFrame()
            player_net_data = pd.DataFrame()

            if has_ice_data:
                crunch_ice_data = shootout_ice_data[shootout_ice_data["Team"] == "Home"].copy()
                player_ice_data = crunch_ice_data[crunch_ice_data["Player"].str.lower() == last_name.lower()]
                if player_ice_data.empty:
                    player_ice_data = crunch_ice_data[crunch_ice_data["Player"].str.lower() == player_name.lower()]
                if player_ice_data.empty:
                    player_ice_data = crunch_ice_data[
                        crunch_ice_data["Player"].str.lower().str.contains(last_name.lower(), na=False)
                    ]
                if player_ice_data.empty:
                    player_ice_data = crunch_ice_data[
                        crunch_ice_data["Player"].str.lower().str.contains(first_name.lower(), na=False)
                    ]

            if has_net_data:
                crunch_net_data = shootout_net_data[shootout_net_data["Team"] == "Home"].copy()
                player_net_data = crunch_net_data[crunch_net_data["Player"].str.lower() == last_name.lower()]
                if player_net_data.empty:
                    player_net_data = crunch_net_data[
                        crunch_net_data["Player"].str.lower().str.contains(last_name.lower(), na=False)
                    ]

            found_player_data = (
                not player_ice_data.empty or
                not player_net_data.empty or
                not player_scouting_data.empty
            )

            if found_player_data:
                # Stats
                if not player_scouting_data.empty:
                    attempts = len(player_scouting_data)
                    goals = (player_scouting_data["goal"] == "Yes").sum()
                elif not player_ice_data.empty:
                    attempts = len(player_ice_data)
                    goals = (player_ice_data["Type"] == "Goal").sum()
                else:
                    attempts = 0
                    goals = 0

                success_rate = (goals / attempts * 100) if attempts > 0 else 0

                col1, col2, col3 = st.columns(3)
                col1.metric("Shootout Attempts", attempts)
                col2.metric("Goals", goals)
                col3.metric("Success Rate", f"{success_rate:.1f}%")

                st.markdown("---")

                # Ice and Net charts
                if not player_ice_data.empty or not player_net_data.empty:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("🏒 Shot Locations on Ice")
                        if not player_ice_data.empty:
                            fig_rink = create_nhl_rink_shootout()
                            shots = player_ice_data[player_ice_data["Type"] == "Shot"]
                            goals_df = player_ice_data[player_ice_data["Type"] == "Goal"]

                            if not shots.empty:
                                fig_rink.add_trace(go.Scatter(
                                    x=shots["X"], y=shots["Y"], mode='markers',
                                    marker=dict(size=12, color='lightblue', symbol='circle', line=dict(width=2, color='blue')),
                                    name='Miss', text=shots["Player"],
                                    hovertemplate='<b>Miss - %{text}</b><br>Location: (%{x:.1f}, %{y:.1f})<extra></extra>'
                                ))
                            if not goals_df.empty:
                                fig_rink.add_trace(go.Scatter(
                                    x=goals_df["X"], y=goals_df["Y"], mode='markers',
                                    marker=dict(size=16, color='red', symbol='star', line=dict(width=2, color='darkred')),
                                    name='Goal ⭐', text=goals_df["Player"],
                                    hovertemplate='<b>GOAL! - %{text}</b><br>Location: (%{x:.1f}, %{y:.1f})<extra></extra>'
                                ))

                            fig_rink.update_layout(title=f"{player_name} - Shootout Shot Locations")
                            st.plotly_chart(fig_rink, use_container_width=True)
                        else:
                            st.info("Ice location data not available")

                    with col2:
                        st.subheader("🥅 Shot Locations on Net")
                        if not player_net_data.empty:
                            fig_net = create_nhl_goal_net()

                            if 'Type' in player_net_data.columns:
                                net_goals = player_net_data[player_net_data["Type"].str.lower().isin(['goal', 'goals'])]
                                net_saves = player_net_data[player_net_data["Type"].str.lower().isin(['save', 'saves', 'saved'])]
                            else:
                                net_goals = player_net_data.copy()
                                net_saves = pd.DataFrame()

                            if not net_saves.empty:
                                fig_net.add_trace(go.Scatter(
                                    x=net_saves["X"], y=net_saves["Y"], mode='markers',
                                    marker=dict(size=12, color='lightblue', symbol='x', line=dict(width=2, color='blue')),
                                    name='Save', text=net_saves["Player"],
                                    hovertemplate='<b>SAVE - %{text}</b><br>Location: (%{x:.1f}, %{y:.1f})<extra></extra>'
                                ))
                            if not net_goals.empty:
                                fig_net.add_trace(go.Scatter(
                                    x=net_goals["X"], y=net_goals["Y"], mode='markers',
                                    marker=dict(size=16, color='red', symbol='star', line=dict(width=2, color='darkred')),
                                    name='Goal', text=net_goals["Player"],
                                    hovertemplate='<b>GOAL! - %{text}</b><br>Location: (%{x:.1f}, %{y:.1f})<extra></extra>'
                                ))

                            fig_net.update_layout(title=f"{player_name} - Shots on Net")
                            st.plotly_chart(fig_net, use_container_width=True)

                            if not net_goals.empty:
                                st.markdown("**Goal Locations by Zone:**")
                                net_goals_copy = net_goals.copy()
                                net_goals_copy["Zone"] = net_goals_copy.apply(
                                    lambda row: get_net_zone(row["X"], row["Y"]), axis=1
                                )
                                zone_counts = net_goals_copy["Zone"].value_counts().reset_index()
                                zone_counts.columns = ["Zone", "Goals"]
                                st.dataframe(zone_counts, hide_index=True, use_container_width=True)

                            if not net_saves.empty:
                                st.caption(f"💾 {len(net_saves)} save(s) by goalie")

                            if net_goals.empty and net_saves.empty:
                                st.info("No shot locations found on net")
                                st.caption(f"Data available: {len(player_net_data)} row(s)")
                        else:
                            st.info("Net location data not available")

                # Scouting details table
                if not player_scouting_data.empty:
                    st.markdown("---")
                    st.subheader("📋 Shootout Details")
                    st.dataframe(
                        player_scouting_data.head(10),
                        hide_index=True,
                        use_container_width=True,
                    )

                # ── VIDEO CLIPS — always shown when player data exists ──────
                render_video_section(
                    team="Crunch",
                    player_last=player_name.split()[-1],
                    scouting_df=shootout_data,
                    key_suffix=player_name,
                )

            else:
                st.info(f"No shootout data available for {player_name}")
                st.caption("Player must be on the Syracuse Crunch to appear in shootout data")

    # ------------------------------------------------------------------
    # TAB 4: FACEOFFS
    # ------------------------------------------------------------------
    with tab4:
        st.markdown('<div class="section-header">Faceoff Statistics</div>', unsafe_allow_html=True)

        if not faceoff_data.empty:
            player_faceoff = faceoff_data[faceoff_data["player"] == player_name]
            if not player_faceoff.empty:
                row = player_faceoff.iloc[0]
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Total", row.get("total_faceoffs", 0))
                col2.metric("Overall", f"{row.get('overall', 0):.1f}%")
                col3.metric("Offensive", f"{row.get('offensive', 0):.1f}%")
                col4.metric("Defensive", f"{row.get('defensive', 0):.1f}%")
                col5.metric("Neutral", f"{row.get('neutral', 0):.1f}%")
            else:
                st.info("No faceoff data available for this player")
        else:
            st.info("No faceoff data loaded")


def render_goalie_card(goalie_name, goalie_stats, goalie_shots, shootout_data, games_df):
    """Render a complete goalie card."""
    st.markdown(f"""
    <div class="goalie-card">
        <div class="player-name">{goalie_name}</div>
        <div class="player-position">Goalie • Syracuse Crunch</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("GP", goalie_stats['games_played'])
    with col2:
        st.metric("SVS", goalie_stats['saves'])
    with col3:
        st.metric("GA", goalie_stats['goals_against'])
    with col4:
        st.metric("SV%", f"{goalie_stats['save_percentage']:.3f}")
    with col5:
        st.metric("GAA", f"{goalie_stats['goals_against_average']:.2f}")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📊 Box Score", "🎯 Goals Against Map", "🥅 Shootout"])

    # ------------------------------------------------------------------
    # TAB 1: BOX SCORE
    # ------------------------------------------------------------------
    with tab1:
        st.markdown('<div class="section-header">Game-by-Game Stats</div>', unsafe_allow_html=True)

        goalie_games = st.session_state.goalies_df[
            st.session_state.goalies_df["skater"] == goalie_name
        ].copy()

        if not goalie_games.empty and not games_df.empty:
            goalie_games = goalie_games.merge(games_df, on="game_id", suffixes=('', '_game'))
            goalie_games["opponent"] = goalie_games.apply(
                lambda row: row["away_team"] if row["team_name"] == row["home_team"] else row["home_team"],
                axis=1
            )
            goalie_games = goalie_games.sort_values("game_id", ascending=True)
            goalie_games["game_number"] = range(1, len(goalie_games) + 1)

            season_summary = pd.DataFrame([{
                "game_number": "TOTAL",
                "opponent": f"{len(goalie_games)} Games",
                "saves": goalie_games["saves"].sum(),
                "goals_against": goalie_games["goals_against"].sum(),
                "sv_pct": f"{goalie_stats['save_percentage']:.3f}",
                "gaa": f"{goalie_stats['goals_against_average']:.2f}"
            }])

            individual_games = goalie_games[["game_number", "opponent", "saves", "goals_against"]].copy()
            individual_games["sv_pct"] = individual_games.apply(
                lambda row: f"{row['saves'] / (row['saves'] + row['goals_against']):.3f}"
                if (row['saves'] + row['goals_against']) > 0 else "0.000",
                axis=1
            )
            individual_games["gaa"] = "N/A"
            individual_games = individual_games.iloc[::-1]

            display_df = pd.concat([season_summary, individual_games], ignore_index=True)

            st.dataframe(
                display_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "game_number": "Game #", "opponent": "Opponent",
                    "saves": "SVS", "goals_against": "GA", "sv_pct": "SV%", "gaa": "GAA"
                }
            )
        else:
            st.info("No game data available")

    # ------------------------------------------------------------------
    # TAB 2: GOALS AGAINST MAP
    # ------------------------------------------------------------------
    with tab2:
        st.markdown('<div class="stat-card"><h3>Goals Against Map</h3></div>', unsafe_allow_html=True)

        if not goalie_shots.empty:
            available_games = sorted(goalie_shots["game_id"].unique())
            game_lookup = {}
            goalie_games = st.session_state.goalies_df[
                st.session_state.goalies_df["skater"] == goalie_name
            ].copy()

            if not goalie_games.empty and not games_df.empty:
                goalie_games = goalie_games.merge(games_df, on="game_id", suffixes=("", "_game"))
                goalie_games["opponent"] = goalie_games.apply(
                    lambda row: row["away_team"] if row["team_name"] == row["home_team"] else row["home_team"],
                    axis=1
                )
                goalie_games = goalie_games.sort_values("game_id")
                goalie_games["game_number"] = range(1, len(goalie_games) + 1)
                for _, row in goalie_games.iterrows():
                    game_lookup[row["game_id"]] = f"Game {row['game_number']}: {row['opponent']}"

            col1, col2 = st.columns([3, 1])
            with col1:
                game_filter_option = st.radio(
                    "Show shots from:",
                    ["All Games", "Single Game"],
                    horizontal=True,
                    key=f"goalie_game_filter_{goalie_name}"
                )

            filtered_shots = goalie_shots.copy()
            if game_filter_option == "Single Game":
                with col2:
                    selected_game = st.selectbox(
                        "Select Game:",
                        available_games,
                        index=len(available_games) - 1,
                        format_func=lambda x: game_lookup.get(x, f"Game {x}"),
                        key=f"goalie_single_game_{goalie_name}"
                    )
                filtered_shots = goalie_shots[goalie_shots["game_id"] == selected_game]

            st.markdown("---")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Shots Faced", len(filtered_shots))
            goals = (filtered_shots["is_goal"] == True).sum()
            col2.metric("Goals Against", goals)
            col3.metric(
                "Save %",
                f"{((len(filtered_shots) - goals) / len(filtered_shots)) * 100:.1f}%"
                if len(filtered_shots) > 0 else "0.0%"
            )
            avg_xg = (
                filtered_shots["xg"].mean()
                if "xg" in filtered_shots.columns and len(filtered_shots) > 0 else 0
            )
            col4.metric("Avg xG Against", f"{avg_xg:.3f}")

            goals_only = filtered_shots[filtered_shots["is_goal"]]
            fig = create_shot_chart(goals_only, goalie_name, view_type="goalie")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No goals scored against this goalie in selected game(s)")
        else:
            st.info("No shot data available for this goalie")

    # ------------------------------------------------------------------
    # TAB 3: GOALIE SHOOTOUT
    # ------------------------------------------------------------------
    with tab3:
        st.markdown('<div class="section-header">🥅 Shootout Performance</div>', unsafe_allow_html=True)

        shootout_ice_data = pd.DataFrame()
        shootout_net_data = pd.DataFrame()

        try:
            ice_file = CRUNCH_DATA_DIR / "Crunch25-26SO.csv"
            if ice_file.exists():
                shootout_ice_data = pd.read_csv(ice_file)
                shootout_ice_data.columns = shootout_ice_data.columns.str.strip()
        except Exception as e:
            st.warning(f"Could not load shootout ice data: {e}")

        try:
            net_file = CRUNCH_DATA_DIR / "SO_Goalzone.csv"
            if net_file.exists():
                shootout_net_data = pd.read_csv(net_file)
                shootout_net_data.columns = shootout_net_data.columns.str.strip()
        except Exception as e:
            st.warning(f"Could not load shootout net data: {e}")

        # Filter scouting data for this goalie
        # Keep a copy of the full shootout_data for video lookups BEFORE we drop the goalie column
        full_shootout_data = shootout_data.copy() if not shootout_data.empty else pd.DataFrame()

        goalie_scouting_data = pd.DataFrame()
        if not shootout_data.empty and "goalie" in shootout_data.columns and goalie_name:
            sd = shootout_data.copy()
            sd["goalie"] = sd["goalie"].astype(str)

            goalie_scouting_data = sd[sd["goalie"].str.contains(goalie_name, case=False, na=False)]

            if goalie_scouting_data.empty and " " in goalie_name:
                last_name_g = goalie_name.split()[-1]
                goalie_scouting_data = sd[sd["goalie"].str.contains(last_name_g, case=False, na=False)]

            # Drop the goalie column for display purposes only (keep a pre-drop copy for video)
            goalie_scouting_display = goalie_scouting_data.drop(columns=["goalie"], errors="ignore")
        else:
            goalie_scouting_display = pd.DataFrame()

        # Filter ice/net data for this goalie
        goalie_ice_data = pd.DataFrame()
        goalie_net_data = pd.DataFrame()
        last_name = goalie_name.split()[-1] if " " in goalie_name else goalie_name

        if not shootout_ice_data.empty and "Player" in shootout_ice_data.columns:
            away_ice = shootout_ice_data[shootout_ice_data["Team"] == "Away"].copy()
            away_ice["Player"] = away_ice["Player"].astype(str)
            goalie_ice_data = away_ice[away_ice["Player"].str.contains(last_name, case=False, na=False)]

        if not shootout_net_data.empty and "Player" in shootout_net_data.columns:
            away_net = shootout_net_data[shootout_net_data["Team"] == "Away"].copy()
            away_net["Player"] = away_net["Player"].astype(str)
            goalie_net_data = away_net[away_net["Player"].str.contains(last_name, case=False, na=False)]

        found_goalie_data = (
            not goalie_scouting_data.empty or
            not goalie_ice_data.empty or
            not goalie_net_data.empty
        )

        if not found_goalie_data:
            st.info(f"No shootout data available for goalie {goalie_name}")
        else:
            # Stats
            if not goalie_scouting_data.empty and "goal" in goalie_scouting_data.columns:
                attempts = len(goalie_scouting_data)
                goals_against = (goalie_scouting_data["goal"].astype(str).str.lower() == "yes").sum()
                saves = attempts - goals_against
                save_pct = (saves / attempts * 100) if attempts > 0 else 0
            elif not goalie_ice_data.empty and "Type" in goalie_ice_data.columns:
                attempts = len(goalie_ice_data)
                goals_against = (goalie_ice_data["Type"].str.lower() == "goal").sum()
                saves = attempts - goals_against
                save_pct = (saves / attempts * 100) if attempts > 0 else 0
            else:
                attempts = goals_against = saves = save_pct = 0

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Shootout Attempts", attempts)
            col2.metric("Goals Against", goals_against)
            col3.metric("Saves", saves)
            col4.metric("Save %", f"{save_pct:.1f}%")

            st.markdown("---")

            # Ice and net visualizations
            if not goalie_ice_data.empty or not goalie_net_data.empty:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("🏒 Shooter Locations (Ice)")
                    if not goalie_ice_data.empty:
                        fig_rink = create_nhl_rink_shootout()
                        goals_df = goalie_ice_data[goalie_ice_data["Type"].str.lower() == "goal"]
                        saves_df = goalie_ice_data[goalie_ice_data["Type"].str.lower().isin(["shot", "save", "miss"])]

                        if not saves_df.empty:
                            fig_rink.add_trace(go.Scatter(
                                x=saves_df["X"], y=saves_df["Y"],
                                mode="markers", name="Save",
                                marker=dict(color="green", size=10)
                            ))
                        if not goals_df.empty:
                            fig_rink.add_trace(go.Scatter(
                                x=goals_df["X"], y=goals_df["Y"],
                                mode="markers", name="Goal Against",
                                marker=dict(color="blue", size=10)
                            ))

                        fig_rink.update_layout(title=f"{goalie_name} — Shootout Ice Map")
                        st.plotly_chart(fig_rink, use_container_width=True)
                    else:
                        st.info("No ice data available")

                with col2:
                    st.subheader("🥅 Shot Locations (Net)")
                    if not goalie_net_data.empty:
                        fig_net = create_nhl_goal_net()
                        goals_df = goalie_net_data[goalie_net_data["Type"].str.lower() == "goal"]
                        saves_df = goalie_net_data[goalie_net_data["Type"].str.lower() == "save"]

                        if not saves_df.empty:
                            fig_net.add_trace(go.Scatter(
                                x=saves_df["X"], y=saves_df["Y"],
                                mode="markers", name="Save",
                                marker=dict(color="green", size=10)
                            ))
                        if not goals_df.empty:
                            fig_net.add_trace(go.Scatter(
                                x=goals_df["X"], y=goals_df["Y"],
                                mode="markers", name="Goal Against",
                                marker=dict(color="blue", size=10)
                            ))

                        fig_net.update_layout(title=f"{goalie_name} — Shootout Net Map")
                        st.plotly_chart(fig_net, use_container_width=True)
                    else:
                        st.info("No net data available")

            # Detailed scouting table
            if not goalie_scouting_display.empty:
                st.markdown("---")
                st.subheader("📋 Shootout Details — Shots Faced")

                scouting = goalie_scouting_display.copy()

                text_cols = ["player", "shot_location_ice", "shot_location_goal", "move_type", "goal", "date"]
                for col in text_cols:
                    if col in scouting.columns:
                        scouting[col] = scouting[col].astype(str)

                if "date" in scouting.columns:
                    scouting["date"] = pd.to_datetime(scouting["date"], errors="coerce")

                st.dataframe(scouting.head(15), hide_index=True, use_container_width=True)

            # ── VIDEO CLIPS: opponent shooters vs this goalie ─────────────
            st.markdown("---")
            st.markdown("### 📹 Opponent Shooter Clips")

            if not goalie_scouting_data.empty and "player" in goalie_scouting_data.columns:
                opponent_shooters = sorted(
                    goalie_scouting_data["player"].dropna().unique().tolist()
                )
                if opponent_shooters:
                    selected_shooter = st.selectbox(
                        "Select opponent shooter:",
                        options=opponent_shooters,
                        key=f"goalie_vid_shooter_{goalie_name}",
                    )
                    # Resolve shooter's team from the full (unmodified) scouting table
                    shooter_team_rows = full_shootout_data[
                        full_shootout_data["player"].str.strip().str.lower()
                        == selected_shooter.lower()
                    ] if not full_shootout_data.empty else pd.DataFrame()

                    shooter_team = (
                        shooter_team_rows["team"].iloc[0]
                        if not shooter_team_rows.empty else "Unknown"
                    )
                    render_video_section(
                        team=shooter_team,
                        player_last=selected_shooter.split()[-1],
                        scouting_df=full_shootout_data,
                        key_suffix=f"{goalie_name}_{selected_shooter}",
                    )
                else:
                    st.caption("No opponent shooters found in scouting data.")
            else:
                st.caption("No opponent scouting data available to match clips against.")


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables with defaults."""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'players_df' not in st.session_state:
        st.session_state.players_df = pd.DataFrame()
    if 'goalies_df' not in st.session_state:
        st.session_state.goalies_df = pd.DataFrame()
    if 'games_df' not in st.session_state:
        st.session_state.games_df = pd.DataFrame()
    if 'shots_df' not in st.session_state:
        st.session_state.shots_df = pd.DataFrame()
    if 'shots_df_goalies' not in st.session_state:
        st.session_state.shots_df_goalies = pd.DataFrame()
    if 'faceoff_df' not in st.session_state:
        st.session_state.faceoff_df = pd.DataFrame()
    if 'shootout_df' not in st.session_state:
        st.session_state.shootout_df = pd.DataFrame()
    if 'excluded_players' not in st.session_state:
        st.session_state.excluded_players = load_excluded_players()
    if 'current_roster' not in st.session_state:
        st.session_state.current_roster = load_current_roster()
        if st.session_state.current_roster:
            logging.info(f"Current roster loaded: {len(st.session_state.current_roster)} players")
    if 'selected_forward' not in st.session_state:
        st.session_state.selected_forward = None
    if 'selected_defenseman' not in st.session_state:
        st.session_state.selected_defenseman = None
    if 'selected_goalie' not in st.session_state:
        st.session_state.selected_goalie = None


# ============================================================================
# ROSTER MANAGEMENT UI
# ============================================================================

def render_roster_management(player_stats, goalie_stats, excluded_players, current_roster):
    """Render roster management section for excluding traded players."""
    with st.expander("⚙️ Manage Roster (Exclude Non-Roster Players)", expanded=False):
        st.markdown('<div class="manage-roster-section">', unsafe_allow_html=True)
        st.markdown("### Exclude players from roster views")
        st.caption("Use this to hide players who have been traded or are no longer with the team.")

        all_players = []
        if not player_stats.empty:
            all_players.extend(player_stats['skater'].tolist())
        if not goalie_stats.empty:
            all_players.extend(goalie_stats['skater'].tolist())
        all_players = sorted(set(all_players))

        if not all_players:
            st.info("No players found in current dataset")
            st.markdown('</div>', unsafe_allow_html=True)
            return excluded_players

        not_on_roster = [p for p in all_players if p not in current_roster] if current_roster else []

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("**Select players to exclude:**")
            newly_excluded = st.multiselect(
                "Players",
                options=all_players,
                default=list(excluded_players),
                help="Select all players who should be hidden from roster views",
                label_visibility="collapsed"
            )
            st.caption(f"Currently excluding: {len(newly_excluded)} player(s)")

        with col2:
            st.markdown("**Quick Actions:**")
            if st.button("🔄 Clear All Exclusions", use_container_width=True):
                newly_excluded = []
                st.success("Cleared all exclusions!")

            if current_roster and not_on_roster:
                if st.button(
                    f"📋 Exclude {len(not_on_roster)} Non-Roster Players",
                    use_container_width=True,
                    help="Exclude players not in Crunch_Roster.txt"
                ):
                    newly_excluded = list(set(newly_excluded + not_on_roster))
                    st.info(f"Added {len(not_on_roster)} players not on current roster")

            if not player_stats.empty:
                zero_game_players = player_stats[player_stats['games_played'] == 0]['skater'].tolist()
                if zero_game_players:
                    if st.button(f"⚡ Exclude {len(zero_game_players)} player(s) with 0 GP", use_container_width=True):
                        newly_excluded = list(set(newly_excluded + zero_game_players))
                        st.info(f"Added {len(zero_game_players)} players with 0 games played")

        if current_roster:
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📋 Current Roster", len(current_roster))
            with col2:
                st.metric("📊 Players in Data", len(all_players))
            with col3:
                st.metric("⚠️ Not on Roster", len(not_on_roster))

            if not_on_roster:
                with st.expander(f"View {len(not_on_roster)} players not on current roster"):
                    not_roster_df = pd.DataFrame({"Player": sorted(not_on_roster)})
                    st.dataframe(not_roster_df, hide_index=True, use_container_width=True)
                    st.caption("These players appear in your game data but are not in Crunch_Roster.txt")
        else:
            st.info("💡 Place 'Crunch_Roster.txt' in the 'Crunch_Box_and_Shot' folder to automatically detect traded players")

        st.markdown("---")
        if st.button("💾 Save Roster Changes", type="primary", use_container_width=True):
            save_excluded_players(set(newly_excluded))
            st.success(f"✅ Saved! Excluding {len(newly_excluded)} player(s) from roster views.")
            st.rerun()

        if newly_excluded:
            st.markdown("---")
            st.markdown("**Currently Excluded Players:**")
            excluded_df = pd.DataFrame({"Player": sorted(newly_excluded)})
            st.dataframe(excluded_df, hide_index=True, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)
        return set(newly_excluded)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application entry point."""
    initialize_session_state()

    st.markdown('<div class="main-title">🏒 Syracuse Crunch</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Player Scouting Dashboard</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🔄 Reload Data", use_container_width=True):
            st.cache_data.clear()
            st.session_state.clear()
            st.rerun()

    if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
        status_placeholder = st.empty()
        progress_bar = st.progress(0)

        with status_placeholder:
            st.info("🔄 Loading Syracuse Crunch data...")

        try:
            players_df, goalies_df, games_df, shots_df, shots_df_goalies = load_all_data()
            progress_bar.progress(50)

            faceoff_df = load_faceoff_data()
            progress_bar.progress(75)

            shootout_df = load_shootout_data()
            progress_bar.progress(100)

            st.session_state.players_df = players_df
            st.session_state.goalies_df = goalies_df
            st.session_state.games_df = games_df
            st.session_state.shots_df = shots_df
            st.session_state.shots_df_goalies = shots_df_goalies
            st.session_state.faceoff_df = faceoff_df
            st.session_state.shootout_df = shootout_df
            st.session_state.data_loaded = True

            status_placeholder.success(
                f"✅ Loaded: {len(games_df)} games, "
                f"{len(players_df['skater'].unique()) if not players_df.empty else 0} players, "
                f"{len(goalies_df['skater'].unique()) if not goalies_df.empty else 0} goalies"
            )
            progress_bar.empty()

        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            logging.exception("Data loading error")
            status_placeholder.empty()
            progress_bar.empty()
            return
    else:
        with st.expander("📊 Data Summary", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Games", len(st.session_state.games_df))
            with col2:
                unique_players = len(st.session_state.players_df['skater'].unique()) if not st.session_state.players_df.empty else 0
                excluded_count = len(st.session_state.excluded_players)
                st.metric("Active Players", unique_players - excluded_count)
            with col3:
                unique_goalies = len(st.session_state.goalies_df['skater'].unique()) if not st.session_state.goalies_df.empty else 0
                st.metric("Goalies", unique_goalies)
            with col4:
                total_shots = len(st.session_state.shots_df) if not st.session_state.shots_df.empty else 0
                st.metric("Total Shots", total_shots)

            if st.session_state.excluded_players:
                st.caption(f"ℹ️ {len(st.session_state.excluded_players)} player(s) excluded from roster views")

    if not st.session_state.games_df.empty:
        current_season = st.session_state.games_df["season"].max()
    else:
        current_season = "2024-25"

    if st.session_state.players_df.empty and st.session_state.goalies_df.empty:
        st.warning("⚠️ No Syracuse Crunch data found. Please ensure CSV files are in the 'Crunch_Box_and_Shot' folder.")
        st.info("""
        **Expected folder structure:**
        ```
        Crunch_Box_and_Shot/
        ├── ahl_boxscore_*.csv
        ├── ahl_shots_*.csv
        └── Crunch_Roster.txt (optional)
        ```
        """)
        return

    player_stats_full = aggregate_player_stats(st.session_state.players_df, st.session_state.shots_df, current_season)
    goalie_stats_full = aggregate_goalie_stats(st.session_state.goalies_df, current_season)

    st.markdown("---")
    st.session_state.excluded_players = render_roster_management(
        player_stats_full,
        goalie_stats_full,
        st.session_state.excluded_players,
        st.session_state.current_roster
    )

    player_stats = filter_excluded_players(player_stats_full, st.session_state.excluded_players)
    goalie_stats = filter_excluded_players(goalie_stats_full, st.session_state.excluded_players)

    view_mode = st.radio("", ["👥 Players", "🥅 Goalies"], horizontal=True, label_visibility="collapsed")

    st.markdown("---")

    # ========================================================================
    # PLAYERS VIEW
    # ========================================================================
    if view_mode == "👥 Players":
        if player_stats.empty:
            st.info("No active players available (all players may be excluded)")
            return

        forwards = player_stats[player_stats['pos'].isin(['C', 'LW', 'RW'])].sort_values("points", ascending=False)
        defensemen = player_stats[player_stats['pos'] == 'D'].sort_values("points", ascending=False)

        pos_tab1, pos_tab2 = st.tabs(["⚡ Forwards", "🛡️ Defensemen"])

        with pos_tab1:
            if forwards.empty:
                st.info("No forwards available")
            else:
                forward_options = [f"{row['skater']} ({row['pos']}) - {row['points']} PTS"
                                   for _, row in forwards.iterrows()]
                forward_names = forwards['skater'].tolist()

                if 'selected_forward' not in st.session_state or st.session_state.selected_forward not in forward_names:
                    st.session_state.selected_forward = forward_names[0]

                try:
                    current_idx = forward_names.index(st.session_state.selected_forward)
                except ValueError:
                    current_idx = 0
                    st.session_state.selected_forward = forward_names[0]

                selected_option = st.selectbox(
                    "Select Forward:",
                    options=forward_options,
                    index=current_idx,
                    key="forward_select"
                )

                selected_forward = forward_names[forward_options.index(selected_option)]
                st.session_state.selected_forward = selected_forward

                player_row = forwards[forwards["skater"] == selected_forward].iloc[0]
                player_shots = st.session_state.shots_df[
                    st.session_state.shots_df["shooter"] == selected_forward
                ].copy() if not st.session_state.shots_df.empty else pd.DataFrame()

                render_player_card(
                    selected_forward,
                    player_row,
                    player_shots,
                    st.session_state.faceoff_df,
                    st.session_state.shootout_df,
                    st.session_state.games_df
                )

        with pos_tab2:
            if defensemen.empty:
                st.info("No defensemen available")
            else:
                defense_options = [f"{row['skater']} - {row['points']} PTS"
                                   for _, row in defensemen.iterrows()]
                defense_names = defensemen['skater'].tolist()

                if 'selected_defenseman' not in st.session_state or st.session_state.selected_defenseman not in defense_names:
                    st.session_state.selected_defenseman = defense_names[0]

                try:
                    current_idx = defense_names.index(st.session_state.selected_defenseman)
                except ValueError:
                    current_idx = 0
                    st.session_state.selected_defenseman = defense_names[0]

                selected_option = st.selectbox(
                    "Select Defenseman:",
                    options=defense_options,
                    index=current_idx,
                    key="defense_select"
                )

                selected_defenseman = defense_names[defense_options.index(selected_option)]
                st.session_state.selected_defenseman = selected_defenseman

                player_row = defensemen[defensemen["skater"] == selected_defenseman].iloc[0]
                player_shots = st.session_state.shots_df[
                    st.session_state.shots_df["shooter"] == selected_defenseman
                ].copy() if not st.session_state.shots_df.empty else pd.DataFrame()

                render_player_card(
                    selected_defenseman,
                    player_row,
                    player_shots,
                    st.session_state.faceoff_df,
                    st.session_state.shootout_df,
                    st.session_state.games_df
                )

    # ========================================================================
    # GOALIES VIEW
    # ========================================================================
    else:
        if goalie_stats.empty:
            st.info("No active goalies available")
            return

        goalie_stats = goalie_stats.sort_values("save_percentage", ascending=False)
        goalie_list = goalie_stats["skater"].tolist()

        goalie_options = [f"{row['skater']} - SV% {row['save_percentage']:.3f}"
                          for _, row in goalie_stats.iterrows()]

        if 'selected_goalie' not in st.session_state or st.session_state.selected_goalie not in goalie_list:
            st.session_state.selected_goalie = goalie_list[0] if goalie_list else None

        try:
            current_idx = goalie_list.index(st.session_state.selected_goalie)
        except (ValueError, AttributeError):
            current_idx = 0
            st.session_state.selected_goalie = goalie_list[0] if goalie_list else None

        if st.session_state.selected_goalie:
            selected_option = st.selectbox(
                "Select Goalie:",
                options=goalie_options,
                index=current_idx,
                key="goalie_select"
            )

            selected_goalie = goalie_list[goalie_options.index(selected_option)]
            st.session_state.selected_goalie = selected_goalie

            goalie_row = goalie_stats[goalie_stats["skater"] == selected_goalie].iloc[0]

            goalie_shots = st.session_state.shots_df_goalies[
                st.session_state.shots_df_goalies["goalie"] == selected_goalie
            ].copy() if not st.session_state.shots_df_goalies.empty else pd.DataFrame()

            render_goalie_card(
                selected_goalie,
                goalie_row,
                goalie_shots,
                st.session_state.shootout_df,
                st.session_state.games_df
            )


if __name__ == "__main__":
    main()

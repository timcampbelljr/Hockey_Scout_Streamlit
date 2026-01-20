"""
Syracuse Crunch Scouting Dashboard - Single Page with Player Cards
Complete player profiles with box scores, shot charts, shootout, and faceoffs

Run with: streamlit run syracuse_crunch_dashboard.py
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

# Configuration
UPLOAD_DIR = Path("uploaded_data")
ASSETS_DIR = Path("assets")
CRUNCH_DATA_DIR = Path("Crunch_Box_and_Shot")
EXCLUDED_PLAYERS_FILE = CRUNCH_DATA_DIR / "excluded_players.json"
ROSTER_FILE = CRUNCH_DATA_DIR / "Crunch_Roster.txt"

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True)
CRUNCH_DATA_DIR.mkdir(exist_ok=True)

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
            
            # Parse format: "9 – Wojciech Stachowiak"
            if '–' in line or '-' in line:
                # Split on either – or -
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
        """Get all CSV files matching pattern from all directories"""
        upload_files = list(UPLOAD_DIR.glob(pattern)) if UPLOAD_DIR.exists() else []
        assets_files = list(ASSETS_DIR.glob(pattern)) if ASSETS_DIR.exists() else []
        crunch_files = list(CRUNCH_DATA_DIR.glob(pattern)) if CRUNCH_DATA_DIR.exists() else []
        
        # Combine and remove duplicates based on filename
        all_files = {}
        for f in upload_files + assets_files + crunch_files:
            all_files[f.name] = f
        return sorted(all_files.values(), key=lambda x: x.name)

    # Get ALL boxscore and shot files from both directories
    boxscore_files = get_unique_files("ahl_boxscore_*.csv")
    shot_files = get_unique_files("ahl_shots_*.csv")

    # Log what we found
    logging.info(f"Found {len(boxscore_files)} boxscore files")
    logging.info(f"Found {len(shot_files)} shot files")
    
    if boxscore_files:
        game_ids = [f.stem.replace("ahl_boxscore_", "") for f in boxscore_files]
        logging.info(f"Boxscore game IDs: {min(game_ids)} to {max(game_ids)}")
    if shot_files:
        game_ids = [f.stem.replace("ahl_shots_", "") for f in shot_files]
        logging.info(f"Shot file game IDs: {min(game_ids)} to {max(game_ids)}")

    all_shots = []
    
    # Load shots
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
    game_team_ids = []  # Track team IDs per game

    # Load boxscores
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
            
            # Store Crunch team ID for this game
            crunch_row = df[df["team_name"] == TARGET_TEAM]
            if not crunch_row.empty:
                crunch_team_id = int(crunch_row["team_id"].iloc[0])
                # Also get opponent team ID
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

    # Combine all data
    players_df = pd.concat(all_players) if all_players else pd.DataFrame()
    goalies_df = pd.concat(all_goalies) if all_goalies else pd.DataFrame()

    # Handle stat corrections: sum stats for duplicate game_id + skater combinations
    if not players_df.empty:
        # Group by game_id and skater, summing numeric stats
        numeric_cols = ["goals", "assists", "penalty_minutes", "plus_minus", "shots"]
        agg_dict = {col: "sum" for col in numeric_cols}
        agg_dict.update({
            "pos": "first",
            "team_name": "first",
            "team_id": "first",
            "team_side": "first",
            "season": "first",
            "number": "first"
        })
        
        players_df = players_df.groupby(["game_id", "skater"], as_index=False).agg(agg_dict)
    
    if not goalies_df.empty:
        # Group by game_id and skater, summing numeric stats
        numeric_cols = ["saves", "goals_against"]
        agg_dict = {col: "sum" for col in numeric_cols}
        agg_dict.update({
            "pos": "first",
            "team_name": "first",
            "team_id": "first",
            "team_side": "first",
            "season": "first",
            "number": "first",
            "minutes_played": "first"
        })
        
        goalies_df = goalies_df.groupby(["game_id", "skater"], as_index=False).agg(agg_dict)
    
    games_df = pd.DataFrame(all_games).drop_duplicates(subset=["game_id"]) if all_games else pd.DataFrame()
    game_team_df = pd.DataFrame(game_team_ids).drop_duplicates(subset=["game_id"]) if game_team_ids else pd.DataFrame()

    # Filter for Syracuse Crunch only
    if not players_df.empty:
        players_df = players_df[players_df["team_name"] == TARGET_TEAM]
    if not goalies_df.empty:
        goalies_df = goalies_df[goalies_df["team_name"] == TARGET_TEAM]

    # Split shots into two dataframes: shots BY Crunch and shots AGAINST Crunch
    shot_df_players = pd.DataFrame()
    shot_df_goalies = pd.DataFrame()
    
    if not shot_df.empty and not games_df.empty:
        # Add season to shots
        if "season" not in shot_df.columns:
            shot_df = pd.merge(shot_df, games_df[["game_id", "season"]], on="game_id", how="left")
        
        # Add Crunch and opponent team IDs to shot data
        if not game_team_df.empty:
            shot_df = pd.merge(shot_df, game_team_df, on="game_id", how="left")
            
            # SHOTS BY CRUNCH PLAYERS (for player shot charts)
            shot_df_players = shot_df[shot_df["shooter_team_id"] == shot_df["crunch_team_id"]].copy()
            
            # Further filter by Crunch players (in case of name conflicts)
            if not players_df.empty:
                crunch_players = players_df["skater"].unique()
                shot_df_players = shot_df_players[shot_df_players["shooter"].isin(crunch_players)]
            
            # SHOTS AGAINST CRUNCH GOALIES (for goalie charts)
            # These are shots where the shooter_team_id is the OPPONENT
            shot_df_goalies = shot_df[shot_df["shooter_team_id"] == shot_df["opponent_team_id"]].copy()
            
            # Further filter by Crunch goalies
            if not goalies_df.empty:
                crunch_goalies = goalies_df["skater"].unique()
                shot_df_goalies = shot_df_goalies[shot_df_goalies["goalie"].isin(crunch_goalies)]
            
            # Log the filtering results
            logging.info(f"Player shots (by Crunch): {len(shot_df_players)}")
            logging.info(f"Goalie shots (against Crunch): {len(shot_df_goalies)}")
        else:
            # Fallback if no team mapping available
            if not goalies_df.empty:
                crunch_goalies = goalies_df["skater"].unique()
                shot_df_goalies = shot_df[shot_df["goalie"].isin(crunch_goalies)].copy()
                logging.info(f"Using fallback goalie filtering: {len(shot_df_goalies)} shots")

    return players_df, goalies_df, games_df, shot_df_players, shot_df_goalies

    
@st.cache_data
def load_faceoff_data():
    """Load faceoff data."""
    try:
        # Check all possible directories for faceoff data
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
    """Load shootout data."""
    try:
        # Check all possible directories for shootout data
        shootout_files = (
            list(ASSETS_DIR.glob("Shootout*.csv")) +
            list(ASSETS_DIR.glob("Shootout*.xlsx")) +
            list(CRUNCH_DATA_DIR.glob("Shootout*.csv")) +
            list(CRUNCH_DATA_DIR.glob("Shootout*.xlsx"))
        )
        
        if not shootout_files:
            logging.warning("No shootout files found")
            return pd.DataFrame()
        
        logging.info(f"Found shootout file: {shootout_files[0]}")
        
        # Read CSV with proper encoding
        df = pd.read_csv(shootout_files[0], encoding='cp1252')
        
        logging.info(f"Loaded shootout data: {len(df)} rows, columns: {df.columns.tolist()}")
        
        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()
        
        # Rename columns to match expected format
        column_mapping = {
            'where player shot from on ice': 'shot_location_ice',
            'where the shot went on goal': 'shot_location_goal',
            'what move they made': 'move_type',
            'goalie (don\'t worry about this)': 'goalie',
            "goalie (don't worry about this)": 'goalie',
            'date': 'date',
            'team': 'team',
            'player': 'player',
            'goal': 'goal'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Clean up data
        df["player"] = df["player"].fillna("").astype(str).str.strip()
        df["team"] = df.get("team", pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
        
        # Remove rows with empty player names or "idle"
        df = df[df["player"].notna() & (df["player"] != "") & (df["player"].str.lower() != "idle")].copy()
        
        # Clean goal column
        df["goal"] = df["goal"].fillna("No")
        df["goal"] = df["goal"].apply(lambda x: "Yes" if str(x).strip().lower() in ["yes", "y", "goal"] else "No")
        
        # Drop rows where all important columns are NaN
        df = df.dropna(subset=['player', 'goal'], how='all')
        
        logging.info(f"Cleaned shootout data: {len(df)} rows")
        
        return df
    except Exception as e:
        logging.exception(f"Error loading shootout data: {e}")
        return pd.DataFrame()



# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def create_shot_chart(df, player_name, view_type="player"):
    """Create interactive shot chart."""
    if df.empty:
        return None
    
    fig = go.Figure()
    
    # Rink outline
    fig.add_shape(type="rect", x0=0, y0=0, x1=848, y1=400,
                  line=dict(color="#000000", width=3), fillcolor="rgba(0,0,0,0)")
    
    # Goal lines (red)
    fig.add_shape(type="line", x0=50, y0=0, x1=50, y1=400, 
                  line=dict(color="red", width=3))
    fig.add_shape(type="line", x0=798, y0=0, x1=798, y1=400, 
                  line=dict(color="red", width=3))
    
    # Blue lines
    fig.add_shape(type="line", x0=274, y0=0, x1=274, y1=400, 
                  line=dict(color="blue", width=2))
    fig.add_shape(type="line", x0=574, y0=0, x1=574, y1=400, 
                  line=dict(color="blue", width=2))
    
    # Center red line
    fig.add_shape(type="line", x0=424, y0=0, x1=424, y1=400, 
                  line=dict(color="red", width=3))
    
    # Goal nets (left and right)
    goal_width = 60
    goal_height = 15
    
    # Left goal net
    left_goal_x = 50
    left_goal_y_center = 200
    fig.add_shape(type="rect", 
                  x0=left_goal_x - goal_height, y0=left_goal_y_center - goal_width/2,
                  x1=left_goal_x, y1=left_goal_y_center + goal_width/2,
                  line=dict(color="red", width=2), fillcolor="rgba(255,0,0,0.1)")
    
    # Right goal net
    right_goal_x = 798
    right_goal_y_center = 200
    fig.add_shape(type="rect",
                  x0=right_goal_x, y0=right_goal_y_center - goal_width/2,
                  x1=right_goal_x + goal_height, y1=right_goal_y_center + goal_width/2,
                  line=dict(color="red", width=2), fillcolor="rgba(255,0,0,0.1)")
    
    if view_type == "player":
        # Saves (shots)
        saves = df[~df["is_goal"]]
        if not saves.empty:
            fig.add_trace(go.Scatter(
                x=saves["x"], y=saves["y"],
                mode='markers',
                marker=dict(size=8, color='#3b82f6', symbol='circle', 
                           line=dict(width=1, color='#1e3a8a')),
                name='Shot',
                text=saves.apply(lambda r: f"{r.get('shot_type', 'Shot')}", axis=1),
                hovertemplate='<b>Shot</b><br>xG: %{customdata:.2%}<extra></extra>',
                customdata=saves.get('xg', 0)
            ))
        
        # Goals
        goals = df[df["is_goal"]]
        if not goals.empty:
            fig.add_trace(go.Scatter(
                x=goals["x"], y=goals["y"],
                mode='markers',
                marker=dict(size=14, color='#10b981', symbol='star', 
                           line=dict(width=2, color='#065f46')),
                name='Goal',
                text=goals.apply(lambda r: f"⚽ GOAL", axis=1),
                hovertemplate='<b>GOAL</b><br>xG: %{customdata:.2%}<extra></extra>',
                customdata=goals.get('xg', 0)
            ))
    else:  # goalie view
        if not df.empty:
            fig.add_trace(go.Scatter(
                x=df["x"], y=df["y"],
                mode='markers',
                marker=dict(size=10, color='#ef4444', symbol='x', 
                           line=dict(width=2, color='#991b1b')),
                name='Goal Against',
                text=df.apply(lambda r: f"Goal by {r.get('shooter', 'Unknown')}", axis=1),
                hovertemplate='<b>Goal Against</b><br>%{text}<br>xG: %{customdata:.2%}<extra></extra>',
                customdata=df.get('xg', 0)
            ))
    
    fig.update_layout(
        title=f"{player_name} - Shot Chart",
        xaxis=dict(range=[0, 848], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[0, 400], showgrid=False, zeroline=False, showticklabels=False, 
                  scaleanchor="x", scaleratio=1),
        plot_bgcolor='#f8f9fa',
        height=400,
        hovermode='closest',
        showlegend=True,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

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
    
    # Add xG stats
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
        .agg(
            saves=("saves", "sum"),
            goals_against=("goals_against", "sum"),
        )
        .reset_index()
    )
    
    if not games_played.empty:
        agg_df = pd.merge(agg_df, games_played, on="skater", how="left")
        agg_df["games_played"] = agg_df["games_played"].fillna(0).astype(int)
    else:
        agg_df["games_played"] = 0
    
    agg_df["save_percentage"] = agg_df.apply(
        lambda row: row["saves"] / (row["saves"] + row["goals_against"])
        if row["saves"] + row["goals_against"] > 0
        else 0,
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
    # Header
    st.markdown(f"""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;'>
            <h1 style='color: white; margin: 0;'>{player_name}</h1>
            <p style='color: rgba(255,255,255,0.9); margin: 5px 0 0 0; font-size: 1.1em;'>{player_stats['pos']} • Syracuse Crunch</p>
        </div>
    """, unsafe_allow_html=True)

    # Season Stats - Now with 8 columns including +/- and PIM
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
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Box Score", "🎯 Shot Chart", "🥅 Shootout", "⚔️ Faceoffs"]
    )

    with tab1:
        st.markdown(
            '<div class="stat-card"><h3>Game-by-Game Stats</h3></div>',
            unsafe_allow_html=True
        )

        player_games = st.session_state.players_df[
            st.session_state.players_df["skater"] == player_name
        ].copy()

        if not player_games.empty and not games_df.empty:
            player_games = player_games.merge(
                games_df, on="game_id", suffixes=("", "_game")
            )

            player_games["opponent"] = player_games.apply(
                lambda row: row["away_team"]
                if row["team_name"] == row["home_team"]
                else row["home_team"],
                axis=1
            )

            player_games["points"] = (
                player_games["goals"] + player_games["assists"]
            )

            player_games = player_games.sort_values("game_id")
            player_games["game_number"] = range(1, len(player_games) + 1)

            display_df = player_games[
                [
                    "game_number",
                    "opponent",
                    "goals",
                    "assists",
                    "points",
                    "plus_minus",
                    "penalty_minutes",
                    "shots",
                ]
            ].iloc[::-1]

            st.dataframe(
                display_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "game_number": "Game #",
                    "opponent": "Opponent",
                    "goals": "G",
                    "assists": "A",
                    "points": "PTS",
                    "plus_minus": "+/-",
                    "penalty_minutes": "PIM",
                    "shots": "SOG",
                },
            )
        else:
            st.info("No game data available")

    with tab2:
        st.markdown(
            '<div class="stat-card"><h3>Shot Chart</h3></div>',
            unsafe_allow_html=True
        )

        if not player_shots.empty:
            # Game selector for shot chart
            available_games = sorted(player_shots["game_id"].unique())
            
            # Create game lookup dictionary with opponent info
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
            
            # Filter shots based on selection
            filtered_shots = player_shots.copy()
            if game_filter_option == "Single Game":
                with col2:
                    selected_game = st.selectbox(
                        "Select Game:",
                        available_games,
                        index=len(available_games) - 1,  # Default to most recent game
                        format_func=lambda x: game_lookup.get(x, f"Game {x}"),
                        key=f"player_single_game_{player_name}"
                    )
                filtered_shots = player_shots[player_shots["game_id"] == selected_game]
            
            st.markdown("---")
            
            # Metrics
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
                if "xg" in filtered_shots.columns and len(filtered_shots) > 0
                else 0
            )
            col4.metric("Avg xG", f"{avg_xg:.3f}")

            fig = create_shot_chart(filtered_shots, player_name, view_type="player")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No shot data available for this player")

    with tab3:
        st.markdown(
            '<div class="section-header">Shootout Performance</div>',
            unsafe_allow_html=True
        )

        if not shootout_data.empty:
            player_shootout = shootout_data[
                shootout_data["player"] == player_name
            ]

            if player_shootout.empty and " " in player_name:
                last_name = player_name.split()[-1]
                player_shootout = shootout_data[
                    shootout_data["player"].str.lower() == last_name.lower()
                ]

            if not player_shootout.empty:
                attempts = len(player_shootout)
                goals = (player_shootout["goal"] == "Yes").sum()
                success_rate = (goals / attempts * 100) if attempts else 0

                col1, col2, col3 = st.columns(3)
                col1.metric("Attempts", attempts)
                col2.metric("Goals", goals)
                col3.metric("Success Rate", f"{success_rate:.1f}%")

                st.markdown("---")
                st.dataframe(
                    player_shootout.head(10),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.info("No shootout data available for this player")
        else:
            st.info("No shootout data loaded")

    with tab4:
        st.markdown(
            '<div class="section-header">Faceoff Statistics</div>',
            unsafe_allow_html=True
        )

        if not faceoff_data.empty:
            player_faceoff = faceoff_data[
                faceoff_data["player"] == player_name
            ]

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
    
    # Header
    st.markdown(f"""
    <div class="goalie-card">
        <div class="player-name">{goalie_name}</div>
        <div class="player-position">Goalie • Syracuse Crunch</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Season Stats
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
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Box Score", "🎯 Goals Against Map", "🥅 Shootout"])
    
    with tab1:
        st.markdown('<div class="section-header">Game-by-Game Stats</div>', unsafe_allow_html=True)
        
        # Get game-by-game stats
        goalie_games = st.session_state.goalies_df[
            st.session_state.goalies_df["skater"] == goalie_name
        ].copy()
        
        if not goalie_games.empty and not games_df.empty:
            goalie_games = goalie_games.merge(games_df, on="game_id", suffixes=('', '_game'))
            goalie_games["opponent"] = goalie_games.apply(
                lambda row: row["away_team"] if row["team_name"] == row["home_team"] else row["home_team"],
                axis=1
            )
            
            # Sort by game_id to get chronological order
            goalie_games = goalie_games.sort_values("game_id", ascending=True)
            
            # Add game number (1, 2, 3, etc.)
            goalie_games["game_number"] = range(1, len(goalie_games) + 1)
            
            # Create season summary row
            season_summary = pd.DataFrame([{
                "game_number": "TOTAL",
                "opponent": f"{len(goalie_games)} Games",
                "saves": goalie_games["saves"].sum(),
                "goals_against": goalie_games["goals_against"].sum(),
                "sv_pct": f"{goalie_stats['save_percentage']:.3f}",
                "gaa": f"{goalie_stats['goals_against_average']:.2f}"
            }])
            
            # Individual games
            individual_games = goalie_games[["game_number", "opponent", "saves", "goals_against"]].copy()
            individual_games["sv_pct"] = individual_games.apply(
                lambda row: f"{row['saves'] / (row['saves'] + row['goals_against']):.3f}" if (row['saves'] + row['goals_against']) > 0 else "0.000",
                axis=1
            )
            individual_games["gaa"] = "N/A"
            
            # Reverse order for display (most recent first) but keep numbering intact
            individual_games = individual_games.iloc[::-1]
            
            display_df = pd.concat([season_summary, individual_games], ignore_index=True)
            
            st.dataframe(
                display_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "game_number": "Game #",
                    "opponent": "Opponent",
                    "saves": "SVS",
                    "goals_against": "GA",
                    "sv_pct": "SV%",
                    "gaa": "GAA"
                }
            )
        else:
            st.info("No game data available")
    
    with tab2:
        st.markdown(
            '<div class="stat-card"><h3>Goals Against Map</h3></div>',
            unsafe_allow_html=True
        )

        if not goalie_shots.empty:
            # Game selector for shot chart
            available_games = sorted(goalie_shots["game_id"].unique())
            
            # Create game lookup dictionary with opponent info
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
            
            # Filter shots based on selection
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
            
            # Metrics - for goalies, show goals against
            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Shots Faced", len(filtered_shots))
            goals = (filtered_shots["is_goal"] == True).sum()
            col2.metric("Goals Against", goals)
            col3.metric(
                "Save %",
                f"{((len(filtered_shots) - goals) / len(filtered_shots)) * 100:.1f}%" if len(filtered_shots) > 0 else "0.0%"
            )

            avg_xg = (
                filtered_shots["xg"].mean()
                if "xg" in filtered_shots.columns and len(filtered_shots) > 0
                else 0
            )
            col4.metric("Avg xG Against", f"{avg_xg:.3f}")

            # Show only goals against on the chart
            goals_only = filtered_shots[filtered_shots["is_goal"]]
            fig = create_shot_chart(goals_only, goalie_name, view_type="goalie")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No shot data available for this goalie")
    
    with tab3:
        st.markdown('<div class="section-header">Shootout Performance</div>', unsafe_allow_html=True)
        
        if not shootout_data.empty:
            # Filter shootout data for this goalie
            # Try exact match first
            goalie_shootout = shootout_data[shootout_data["goalie"] == goalie_name]
            
            # If no exact match, try matching with different name formats
            if goalie_shootout.empty and " " in goalie_name:
                # Split the goalie name
                name_parts = goalie_name.split()
                first_name = name_parts[0]
                last_name = name_parts[-1]
                
                # Try: "FirstName LastName" or "LastName FirstName" or just "LastName"
                goalie_shootout = shootout_data[
                    (shootout_data["goalie"].str.lower() == goalie_name.lower()) |
                    (shootout_data["goalie"].str.lower() == f"{first_name} {last_name}".lower()) |
                    (shootout_data["goalie"].str.lower() == f"{last_name} {first_name}".lower()) |
                    (shootout_data["goalie"].str.lower() == last_name.lower()) |
                    (shootout_data["goalie"].str.lower() == first_name.lower())
                ]
            
            if not goalie_shootout.empty:
                attempts = len(goalie_shootout)
                goals_allowed = (goalie_shootout["goal"] == "Yes").sum()
                saves = attempts - goals_allowed
                save_rate = (saves / attempts * 100) if attempts > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Attempts", attempts)
                col2.metric("Saves", saves)
                col3.metric("Save %", f"{save_rate:.1f}%")
                
                st.markdown("---")
                
                # Display the shootout attempts table
                st.dataframe(
                    goalie_shootout.head(10),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.info("No shootout data available for this goalie")
        else:
            st.info("No shootout data loaded")

# ============================================================================
# ROSTER MANAGEMENT UI
# ============================================================================

def render_roster_management(player_stats, goalie_stats, excluded_players, current_roster):
    """Render roster management section for excluding traded players."""
    
    with st.expander("⚙️ Manage Roster (Exclude Non-Roster Players)", expanded=False):
        st.markdown('<div class="manage-roster-section">', unsafe_allow_html=True)
        
        st.markdown("### Exclude players from roster views")
        st.caption("Use this to hide players who have been traded or are no longer with the team. Their data remains in the system but won't appear in roster selections.")
        
        # Combine all players
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
        
        # Identify players not on current roster
        not_on_roster = [p for p in all_players if p not in current_roster] if current_roster else []
        
        # Split into two columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Select players to exclude:**")
            
            # Multiselect with currently excluded players pre-selected
            newly_excluded = st.multiselect(
                "Players",
                options=all_players,
                default=list(excluded_players),
                help="Select all players who should be hidden from roster views",
                label_visibility="collapsed"
            )
            
            # Show count
            st.caption(f"Currently excluding: {len(newly_excluded)} player(s)")
        
        with col2:
            st.markdown("**Quick Actions:**")
            
            if st.button("🔄 Clear All Exclusions", use_container_width=True):
                newly_excluded = []
                st.success("Cleared all exclusions!")
            
            # Show button to exclude players not on current roster
            if current_roster and not_on_roster:
                if st.button(f"📋 Exclude {len(not_on_roster)} Non-Roster Players", use_container_width=True, help="Exclude players not in Crunch_Roster.txt"):
                    newly_excluded = list(set(newly_excluded + not_on_roster))
                    st.info(f"Added {len(not_on_roster)} players not on current roster")
            
            if not player_stats.empty:
                # Find players with 0 games (potential trades)
                zero_game_players = player_stats[player_stats['games_played'] == 0]['skater'].tolist()
                if zero_game_players:
                    if st.button(f"⚡ Exclude {len(zero_game_players)} player(s) with 0 GP", use_container_width=True):
                        newly_excluded = list(set(newly_excluded + zero_game_players))
                        st.info(f"Added {len(zero_game_players)} players with 0 games played")
        
        # Show roster analysis
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
        
        # Save button
        st.markdown("---")
        if st.button("💾 Save Roster Changes", type="primary", use_container_width=True):
            save_excluded_players(set(newly_excluded))
            st.success(f"✅ Saved! Excluding {len(newly_excluded)} player(s) from roster views.")
            st.rerun()
        
        # Show excluded players list if any
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
    # Title
    st.markdown('<div class="main-title">🏒 Syracuse Crunch</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Player Scouting Dashboard</div>', unsafe_allow_html=True)
    
    # Add reload button in top right
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🔄 Reload Data", use_container_width=True):
            st.cache_data.clear()
            st.session_state.clear()
            st.rerun()
    
    # Load excluded players
    if 'excluded_players' not in st.session_state:
        st.session_state.excluded_players = load_excluded_players()
    
    # Load current roster
    if 'current_roster' not in st.session_state:
        st.session_state.current_roster = load_current_roster()
        if st.session_state.current_roster:
            logging.info(f"Current roster loaded: {len(st.session_state.current_roster)} players")
    
    # Load data with progress
    if 'data_loaded' not in st.session_state:
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        with status_placeholder:
            st.info("🔄 Loading Syracuse Crunch data...")
        
        try:
            # Load data
            players_df, goalies_df, games_df, shots_df, shots_df_goalies = load_all_data()
            progress_bar.progress(50)
            
            faceoff_df = load_faceoff_data()
            progress_bar.progress(75)
            
            shootout_df = load_shootout_data()
            progress_bar.progress(100)
            
            # Store in session state
            st.session_state.players_df = players_df
            st.session_state.goalies_df = goalies_df
            st.session_state.games_df = games_df
            st.session_state.shots_df = shots_df
            st.session_state.shots_df_goalies = shots_df_goalies
            st.session_state.faceoff_df = faceoff_df
            st.session_state.shootout_df = shootout_df
            st.session_state.data_loaded = True
            
            # Show what was loaded
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
        # Show quick stats in collapsed section
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
    
    # Get current season
    if not st.session_state.games_df.empty:
        current_season = st.session_state.games_df["season"].max()
    else:
        current_season = "2024-25"
    
    # Check if we have data
    if st.session_state.players_df.empty and st.session_state.goalies_df.empty:
        st.warning("⚠️ No Syracuse Crunch data found. Please ensure CSV files are in the 'Crunch_Box_and_Shot' folder.")
        st.info("""
        **Expected folder structure:**
        ```
        Crunch_Box_and_Shot/
        ├── ahl_boxscore_*.csv
        ├── ahl_shots_*.csv
        └── Crunch_Roster.txt (optional - for roster management)
        ```
        """)
        return
    
    # Aggregate stats (before filtering)
    player_stats_full = aggregate_player_stats(st.session_state.players_df, st.session_state.shots_df, current_season)
    goalie_stats_full = aggregate_goalie_stats(st.session_state.goalies_df, current_season)
    
    # Roster Management Section
    st.markdown("---")
    st.session_state.excluded_players = render_roster_management(
        player_stats_full, 
        goalie_stats_full, 
        st.session_state.excluded_players,
        st.session_state.current_roster
    )
    
    # Filter out excluded players for display
    player_stats = filter_excluded_players(player_stats_full, st.session_state.excluded_players)
    goalie_stats = filter_excluded_players(goalie_stats_full, st.session_state.excluded_players)
    
    # View selector
    view_mode = st.radio("", ["👥 Players", "🥅 Goalies"], horizontal=True, label_visibility="collapsed")
    
    st.markdown("---")
    
    # ========================================================================
    # PLAYERS VIEW
    # ========================================================================
    if view_mode == "👥 Players":
        if player_stats.empty:
            st.info("No active players available (all players may be excluded)")
            return
        
        # Categorize players by position
        forwards = player_stats[player_stats['pos'].isin(['C', 'LW', 'RW'])].sort_values("points", ascending=False)
        defensemen = player_stats[player_stats['pos'] == 'D'].sort_values("points", ascending=False)
        
        # Position tabs
        pos_tab1, pos_tab2 = st.tabs(["⚡ Forwards", "🛡️ Defensemen"])
        
        with pos_tab1:
            if forwards.empty:
                st.info("No forwards available")
            else:
                # Dropdown for forward selection
                forward_options = [f"{row['skater']} ({row['pos']}) - {row['points']} PTS" 
                                 for _, row in forwards.iterrows()]
                forward_names = forwards['skater'].tolist()
                
                # Initialize selected forward
                if 'selected_forward' not in st.session_state or st.session_state.selected_forward not in forward_names:
                    st.session_state.selected_forward = forward_names[0]
                
                # Find current index
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
                
                # Extract player name from selection
                selected_forward = forward_names[forward_options.index(selected_option)]
                st.session_state.selected_forward = selected_forward
                
                # Get player data
                player_row = forwards[forwards["skater"] == selected_forward].iloc[0]
                player_shots = st.session_state.shots_df[
                    st.session_state.shots_df["shooter"] == selected_forward
                ].copy() if not st.session_state.shots_df.empty else pd.DataFrame()
                
                # Render card
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
                # Dropdown for defenseman selection
                defense_options = [f"{row['skater']} - {row['points']} PTS" 
                                 for _, row in defensemen.iterrows()]
                defense_names = defensemen['skater'].tolist()
                
                # Initialize selected defenseman
                if 'selected_defenseman' not in st.session_state or st.session_state.selected_defenseman not in defense_names:
                    st.session_state.selected_defenseman = defense_names[0]
                
                # Find current index
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
                
                # Extract player name from selection
                selected_defenseman = defense_names[defense_options.index(selected_option)]
                st.session_state.selected_defenseman = selected_defenseman
                
                # Get player data
                player_row = defensemen[defensemen["skater"] == selected_defenseman].iloc[0]
                player_shots = st.session_state.shots_df[
                    st.session_state.shots_df["shooter"] == selected_defenseman
                ].copy() if not st.session_state.shots_df.empty else pd.DataFrame()
                
                # Render card
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
        
        # Sort goalies by save percentage
        goalie_stats = goalie_stats.sort_values("save_percentage", ascending=False)
        goalie_list = goalie_stats["skater"].tolist()
        
        # Dropdown for goalie selection
        goalie_options = [f"{row['skater']} - SV% {row['save_percentage']:.3f}" 
                         for _, row in goalie_stats.iterrows()]
        
        # Initialize selected goalie
        if 'selected_goalie' not in st.session_state or st.session_state.selected_goalie not in goalie_list:
            st.session_state.selected_goalie = goalie_list[0] if goalie_list else None
        
        # Find current index
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
            
            # Extract goalie name from selection
            selected_goalie = goalie_list[goalie_options.index(selected_option)]
            st.session_state.selected_goalie = selected_goalie
            
            # Get selected goalie stats
            goalie_row = goalie_stats[goalie_stats["skater"] == selected_goalie].iloc[0]
            
            # Get goalie-specific data
            goalie_shots = st.session_state.shots_df_goalies[
                st.session_state.shots_df_goalies["goalie"] == selected_goalie
            ].copy() if not st.session_state.shots_df_goalies.empty else pd.DataFrame()
            
            # Render card
            render_goalie_card(
                selected_goalie,
                goalie_row,
                goalie_shots,
                st.session_state.shootout_df,
                st.session_state.games_df
            )

if __name__ == "__main__":
    main()

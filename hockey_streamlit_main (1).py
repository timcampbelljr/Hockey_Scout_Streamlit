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
</style>
""", unsafe_allow_html=True)

# Configuration
UPLOAD_DIR = Path("uploaded_data")
ASSETS_DIR = Path("assets")
CRUNCH_DATA_DIR = Path("Crunch_Box_and_Shot")  # Your specific folder

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True)
CRUNCH_DATA_DIR.mkdir(exist_ok=True)

# Target team
TARGET_TEAM = "Syracuse Crunch"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
            
            game_info = {
                "game_id": game_id,
                "home_team": home_team,
                "away_team": away_team,
                "season": season,
            }
            all_games.append(game_info)
            
            df["is_goalie"] = df["pos"] == "G"
            df["season"] = season
            
            players_df = df[~df["is_goalie"]].copy()
            goalies_df = df[df["is_goalie"]].copy()
            
            # Add goals from shots data
            if not shot_df.empty:
                game_shots = shot_df[shot_df["game_id"] == game_id]
                player_goals = (
                    game_shots[game_shots["is_goal"]]
                    .groupby("shooter")
                    .size()
                    .reset_index(name="goals_from_shots")
                )
                players_df = pd.merge(
                    players_df, player_goals, left_on="skater", right_on="shooter", how="left"
                )
                players_df["g"] = players_df["goals_from_shots"].fillna(0).astype(int)
                players_df.drop(columns=["shooter", "goals_from_shots"], inplace=True, errors='ignore')
            
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
    players_df = pd.concat(all_players).drop_duplicates(subset=["game_id", "skater"]) if all_players else pd.DataFrame()
    goalies_df = pd.concat(all_goalies).drop_duplicates(subset=["game_id", "skater"]) if all_goalies else pd.DataFrame()
    games_df = pd.DataFrame(all_games).drop_duplicates(subset=["game_id"]) if all_games else pd.DataFrame()
    
    # Filter for Syracuse Crunch only
    if not players_df.empty:
        players_df = players_df[players_df["team_name"] == TARGET_TEAM]
    if not goalies_df.empty:
        goalies_df = goalies_df[goalies_df["team_name"] == TARGET_TEAM]
    
    # Add season to shots and filter by Crunch players
    if not shot_df.empty and not games_df.empty:
        if "season" not in shot_df.columns:
            shot_df = pd.merge(shot_df, games_df[["game_id", "season"]], on="game_id", how="left")
        
        # Filter shots for Crunch players only
        if not players_df.empty:
            crunch_players = players_df["skater"].unique()
            shot_df = shot_df[shot_df["shooter"].isin(crunch_players)]
        
        # Filter shots for Crunch goalies
        if not goalies_df.empty:
            crunch_goalies = goalies_df["skater"].unique()
            shot_df_goalies = shot_df[shot_df["goalie"].isin(crunch_goalies)]
        else:
            shot_df_goalies = pd.DataFrame()
    else:
        shot_df_goalies = pd.DataFrame()
    
    return players_df, goalies_df, games_df, shot_df, shot_df_goalies

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
            return pd.DataFrame()
        
        # Read CSV with proper encoding
        df = pd.read_csv(shootout_files[0], encoding='cp1252')
        
        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()
        
        # Rename columns to match expected format
        column_mapping = {
            'where player shot from on ice': 'shot_location_ice',
            'where the shot went on goal': 'shot_location_goal',
            'what move they made': 'move_type',
            'goalie (don\'t worry about this)': 'goalie',
            'date': 'date'
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
    <div class="player-card">
        <div class="player-name">{player_name}</div>
        <div class="player-position">{player_stats['pos']} • Syracuse Crunch</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Season Stats
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("GP", player_stats['games_played'])
    with col2:
        st.metric("G", player_stats['goals'])
    with col3:
        st.metric("A", player_stats['assists'])
    with col4:
        st.metric("PTS", player_stats['points'])
    with col5:
        st.metric("SOG", player_stats['shots'])
    with col6:
        st.metric("Avg xG", f"{player_stats['avg_xg']:.3f}")
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Box Score", "🎯 Shot Chart", "🥅 Shootout", "⚔️ Faceoffs"])
    
    with tab1:
        st.markdown('<div class="section-header">Game-by-Game Stats</div>', unsafe_allow_html=True)
        
        # Get game-by-game stats
        player_games = st.session_state.players_df[
            st.session_state.players_df["skater"] == player_name
        ].copy()
        
        if not player_games.empty and not games_df.empty:
            player_games = player_games.merge(games_df, on="game_id", suffixes=('', '_game'))
            player_games["opponent"] = player_games.apply(
                lambda row: row["away_team"] if row["team_name"] == row["home_team"] else row["home_team"],
                axis=1
            )
            player_games["points"] = player_games["goals"] + player_games["assists"]
            
            # Create season summary row
            season_summary = pd.DataFrame([{
                "game_id": "SEASON TOTAL",
                "opponent": f"{len(player_games)} Games",
                "goals": player_games["goals"].sum(),
                "assists": player_games["assists"].sum(),
                "points": player_games["points"].sum(),
                "plus_minus": player_games["plus_minus"].sum(),
                "penalty_minutes": player_games["penalty_minutes"].sum(),
                "shots": player_games["shots"].sum()
            }])
            
            # Sort individual games and combine with summary
            individual_games = player_games[["game_id", "opponent", "goals", "assists", "points", "plus_minus", "penalty_minutes", "shots"]].sort_values("game_id", ascending=False)
            display_df = pd.concat([season_summary, individual_games], ignore_index=True)
            
            st.dataframe(
                display_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "game_id": "Game",
                    "opponent": "Opponent",
                    "goals": "G",
                    "assists": "A",
                    "points": "PTS",
                    "plus_minus": "+/-",
                    "penalty_minutes": "PIM",
                    "shots": "SOG"
                }
            )
        else:
            st.info("No game data available")
    
    with tab2:
        st.markdown('<div class="section-header">Shot Chart</div>', unsafe_allow_html=True)
        
        if not player_shots.empty:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Shots", len(player_shots))
            with col2:
                goals = len(player_shots[player_shots["is_goal"] == True])
                st.metric("Goals", goals)
            with col3:
                if len(player_shots) > 0:
                    shooting_pct = (goals / len(player_shots)) * 100
                    st.metric("Shooting %", f"{shooting_pct:.1f}%")
            with col4:
                avg_xg = player_shots["xg"].mean() if "xg" in player_shots.columns else 0
                st.metric("Avg xG", f"{avg_xg:.3f}")
            
            fig = create_shot_chart(player_shots, player_name, view_type="player")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No shot data available for this player")
    
    with tab3:
        st.markdown('<div class="section-header">Shootout Performance</div>', unsafe_allow_html=True)
        
        if not shootout_data.empty:
            player_shootout = shootout_data[shootout_data["player"] == player_name]
            
            if not player_shootout.empty:
                attempts = len(player_shootout)
                goals = (player_shootout["goal"] == "Yes").sum()
                success_rate = (goals / attempts * 100) if attempts > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Attempts", attempts)
                with col2:
                    st.metric("Goals", goals)
                with col3:
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                
                st.markdown("---")
                
                # Display shootout details
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Shot Locations (Ice):**")
                    if "shot_location_ice" in player_shootout.columns:
                        locations = player_shootout["shot_location_ice"].value_counts()
                        for loc, count in locations.items():
                            if pd.notna(loc) and str(loc).lower() not in ['nan', 'n/a', '']:
                                st.write(f"• {loc}: {count}")
                    
                    st.markdown("**Move Types:**")
                    if "move_type" in player_shootout.columns:
                        moves = player_shootout["move_type"].value_counts()
                        for move, count in moves.items():
                            if pd.notna(move) and str(move).lower() not in ['nan', 'n/a', '']:
                                st.write(f"• {move}: {count}")
                
                with col2:
                    st.markdown("**Shot Locations (Goal):**")
                    if "shot_location_goal" in player_shootout.columns:
                        goal_locs = player_shootout["shot_location_goal"].value_counts()
                        for loc, count in goal_locs.items():
                            if pd.notna(loc) and str(loc).lower() not in ['nan', 'n/a', '']:
                                st.write(f"• {loc}: {count}")
                
                # Show recent attempts
                st.markdown("---")
                st.markdown("**Recent Attempts:**")
                display_cols = ['date', 'shot_location_ice', 'shot_location_goal', 'move_type', 'goal']
                available_cols = [col for col in display_cols if col in player_shootout.columns]
                
                if available_cols:
                    recent = player_shootout[available_cols].head(10)
                    st.dataframe(
                        recent,
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            'date': 'Date',
                            'shot_location_ice': 'Shot From',
                            'shot_location_goal': 'Shot To',
                            'move_type': 'Move',
                            'goal': 'Result'
                        }
                    )
            else:
                st.info("No shootout data available for this player")
        else:
            st.info("No shootout data loaded")
    
    with tab4:
        st.markdown('<div class="section-header">Faceoff Statistics</div>', unsafe_allow_html=True)
        
        if not faceoff_data.empty:
            player_faceoff = faceoff_data[faceoff_data["player"] == player_name]
            
            if not player_faceoff.empty:
                row = player_faceoff.iloc[0]
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Total", row.get("total_faceoffs", 0))
                with col2:
                    st.metric("Overall", f"{row.get('overall', 0):.1f}%")
                with col3:
                    st.metric("Offensive", f"{row.get('offensive', 0):.1f}%")
                with col4:
                    st.metric("Defensive", f"{row.get('defensive', 0):.1f}%")
                with col5:
                    st.metric("Neutral", f"{row.get('neutral', 0):.1f}%")
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
        # ... box score code ...
    
    with tab2:
        st.markdown('<div class="section-header">Goals Against Chart</div>', unsafe_allow_html=True)
        # ... goals against code ...
    
    with tab3:  # <-- Make sure this aligns with tab1 and tab2 above
        st.markdown('<div class="section-header">Shootout Performance</div>', unsafe_allow_html=True)
        
        if not shootout_data.empty:
            # For goalies, we need to check the "goalie" column if it exists
            # Otherwise we can't match goalies to shootout data
            if "goalie" in shootout_data.columns:
                goalie_shootout = shootout_data[shootout_data["goalie"] == goalie_name]
            else:
                goalie_shootout = pd.DataFrame()
            
            if not goalie_shootout.empty:
                shots_faced = len(goalie_shootout)
                goals_against = (goalie_shootout["goal"] == "Yes").sum()
                save_pct = ((shots_faced - goals_against) / shots_faced * 100) if shots_faced > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Shots Faced", shots_faced)
                with col2:
                    st.metric("Goals Against", goals_against)
                with col3:
                    st.metric("Save %", f"{save_pct:.1f}%")
                
                st.markdown("---")
                
                # Display breakdown by shooter
                st.markdown("**Goals Against by Shooter:**")
                if "player" in goalie_shootout.columns:
                    goals_df = goalie_shootout[goalie_shootout["goal"] == "Yes"]
                    if not goals_df.empty:
                        shooter_goals = goals_df["player"].value_counts()
                        for shooter, count in shooter_goals.items():
                            st.write(f"• {shooter}: {count}")
                    else:
                        st.success("No goals allowed!")
            else:
                st.info("No shootout data available for this goalie")
        else:
            st.info("No shootout data loaded")

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
                st.metric("Players", unique_players)
            with col3:
                unique_goalies = len(st.session_state.goalies_df['skater'].unique()) if not st.session_state.goalies_df.empty else 0
                st.metric("Goalies", unique_goalies)
            with col4:
                total_shots = len(st.session_state.shots_df) if not st.session_state.shots_df.empty else 0
                st.metric("Total Shots", total_shots)
    
    # Get current season
    if not st.session_state.games_df.empty:
        current_season = st.session_state.games_df["season"].max()
    else:
        current_season = "2024-25"
    
    # Check if we have data
    if st.session_state.players_df.empty and st.session_state.goalies_df.empty:
        st.warning("⚠️ No Syracuse Crunch data found. Please ensure CSV files are in the 'assets' folder.")
        st.info("""
        **Expected folder structure:**
        ```
        assets/
        ├── ahl_boxscore_*.csv
        └── ahl_shots_*.csv
        ```
        """)
        return
    
    # Aggregate stats
    player_stats = aggregate_player_stats(st.session_state.players_df, st.session_state.shots_df, current_season)
    goalie_stats = aggregate_goalie_stats(st.session_state.goalies_df, current_season)
    
    # View selector
    view_mode = st.radio("", ["👥 Players", "🥅 Goalies"], horizontal=True, label_visibility="collapsed")
    
    st.markdown("---")
    
    # ========================================================================
    # PLAYERS VIEW
    # ========================================================================
    # ========================================================================
    # PLAYERS VIEW
    # ========================================================================
    if view_mode == "👥 Players":
        if player_stats.empty:
            st.info("No player data available for Syracuse Crunch")
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
                if 'selected_forward' not in st.session_state:
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
                if 'selected_defenseman' not in st.session_state:
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
            else:
                st.info("Select a player from the roster")
    
    # ========================================================================
    # GOALIES VIEW
    # ========================================================================
    else:
        if goalie_stats.empty:
            st.info("No goalie data available for Syracuse Crunch")
            return
        
        # Sort goalies by save percentage
        goalie_stats = goalie_stats.sort_values("save_percentage", ascending=False)
        goalie_list = goalie_stats["skater"].tolist()
        
        # Create two columns: goalie list and goalie card
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown("### 🥅 Goalies")
            
            # Display goalie selection buttons
            if 'selected_goalie' not in st.session_state:
                st.session_state.selected_goalie = goalie_list[0] if goalie_list else None
            
            for goalie_name in goalie_list:
                goalie_row = goalie_stats[goalie_stats["skater"] == goalie_name].iloc[0]
                
                # Button styling based on selection
                button_type = "primary" if st.session_state.selected_goalie == goalie_name else "secondary"
                
                # Create button with goalie stats preview
                if st.button(
                    f"{goalie_name}\nSV% {goalie_row['save_percentage']:.3f}",
                    key=f"goalie_{goalie_name}",
                    use_container_width=True,
                    type=button_type
                ):
                    st.session_state.selected_goalie = goalie_name
                    st.rerun()
        
        with col2:
            if st.session_state.selected_goalie and st.session_state.selected_goalie in goalie_list:
                # Get selected goalie stats
                goalie_row = goalie_stats[goalie_stats["skater"] == st.session_state.selected_goalie].iloc[0]
                
                # Get goalie-specific data
                goalie_shots = st.session_state.shots_df_goalies[
                    st.session_state.shots_df_goalies["goalie"] == st.session_state.selected_goalie
                ].copy() if not st.session_state.shots_df_goalies.empty else pd.DataFrame()
                
                # Render card
                render_goalie_card(
                    st.session_state.selected_goalie,
                    goalie_row,
                    goalie_shots,
                    st.session_state.shootout_df,
                    st.session_state.games_df
                )
            else:
                st.info("Select a goalie from the list")

if __name__ == "__main__":
    main()

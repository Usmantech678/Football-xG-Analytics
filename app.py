
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Football xG Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "shots_with_context.csv"
MODEL_PATH = BASE_DIR / "xgboost_model.joblib"

NUMERIC_FEATURES = [
    "x", "y", "distance", "angle",
    "goalkeeper_distance", "goalkeeper_goal_distance",
    "nearest_defender_distance", "defenders_within_5m",
    "defenders_in_shot_lane", "visible_defender_count",
    "visible_teammate_count"
]
CATEGORICAL_FEATURES = [
    "shot_type", "body_part", "technique", "under_pressure"
]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CONTEXT_FEATURES = [
    "goalkeeper_distance", "goalkeeper_goal_distance",
    "nearest_defender_distance", "defenders_within_5m",
    "defenders_in_shot_lane", "visible_defender_count",
    "visible_teammate_count"
]

st.markdown("""
<style>
.stApp {
    background: linear-gradient(145deg, #071321 0%, #0a192a 55%, #07111d 100%);
    color: #edf4ff;
}
[data-testid="stSidebar"] {
    background: #081321;
    border-right: 1px solid #1b3550;
}
h1, h2, h3 {
    color: #f2f7ff !important;
}
p, label, .stMarkdown {
    color: #d1def0;
}
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #10263d, #0b1b2d);
    border: 1px solid #214363;
    padding: 18px;
    border-radius: 14px;
}
div[data-testid="stMetric"] label {
    color: #9eb8d7 !important;
}
div[data-testid="stMetricValue"] {
    color: #47e5b1 !important;
}
.panel {
    background: linear-gradient(135deg, #10243a, #0a192a);
    border: 1px solid #1c3a57;
    border-radius: 15px;
    padding: 18px;
    margin-bottom: 12px;
}
.hero {
    padding: 26px;
    border-radius: 18px;
    background: linear-gradient(110deg, #102b42, #075b52);
    border: 1px solid #28665f;
    margin-bottom: 22px;
}
.hero h1 {
    margin: 0;
    font-size: 2.35rem;
}
.hero p {
    color: #c5e5e8;
    margin-bottom: 0;
}
.small-note {
    color: #8faac7;
    font-size: 0.85rem;
}
div.stButton > button[kind="primary"] {
    background: #0dbb8c;
    color: #061b20;
    border: none;
    border-radius: 10px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

PLOT_TEMPLATE = "plotly_dark"

def find_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    for col in ["x", "y"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    dx = 120 - df["x"]
    dy = 40 - df["y"]

    df["distance"] = np.sqrt(dx**2 + dy**2)
    df["angle"] = np.arctan2(
        abs(8 * dx),
        dx**2 + (36 - df["y"]) * (44 - df["y"])
    )

    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = df[col].fillna("Unknown").astype(str)

    df = df.dropna(subset=["x", "y"]).copy()
    return df


@st.cache_resource
def load_model():
    saved = joblib.load(MODEL_PATH)

    if not isinstance(saved, dict):
        raise ValueError(
            "Unexpected model file format. Expected a dictionary "
            "containing preprocessor and model."
        )

    return saved["preprocessor"], saved["model"]


def add_pitch(fig):
    """Draw a simple football pitch using StatsBomb's 120 x 80 coordinates."""
    line_color = "#6f9d91"

    fig.add_shape(
        type="rect", x0=0, y0=0, x1=120, y1=80,
        line=dict(color=line_color, width=2)
    )
    fig.add_shape(
        type="line", x0=60, y0=0, x1=60, y1=80,
        line=dict(color=line_color, width=1)
    )
    fig.add_shape(
        type="circle", x0=49, y0=29, x1=71, y1=51,
        line=dict(color=line_color, width=1)
    )
    for x0, x1 in [(0, 18), (102, 120)]:
        fig.add_shape(
            type="rect", x0=x0, y0=18, x1=x1, y1=62,
            line=dict(color=line_color, width=1)
        )
    for x0, x1 in [(0, 6), (114, 120)]:
        fig.add_shape(
            type="rect", x0=x0, y0=30, x1=x1, y1=50,
            line=dict(color=line_color, width=1)
        )

    fig.update_xaxes(range=[0, 120], visible=False)
    fig.update_yaxes(range=[0, 80], visible=False, scaleanchor="x")
    fig.update_layout(
        template=PLOT_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#10432f",
        margin=dict(l=8, r=8, t=10, b=8),
        height=450,
        showlegend=True
    )
    return fig


# ---------- Load files and model ----------
st.markdown("""
<div class="hero">
    <h1>⚽ Football <span style="color:#47e5b1">xG</span> Analytics</h1>
    <p>Explore shot quality, compare teams and players, and estimate goal probability with XGBoost.</p>
</div>
""", unsafe_allow_html=True)

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.info("Make sure data/shots_with_context.csv exists.")
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"Model not found: {MODEL_PATH}")
    st.info("Run python train_xgboost.py first.")
    st.stop()

try:
    df = load_data()
    preprocessor, model = load_model()
except Exception as exc:
    st.error(f"Could not load the dataset or model: {exc}")
    st.stop()

missing = [col for col in MODEL_FEATURES if col not in df.columns]
if missing:
    st.error(f"Required model features are missing: {missing}")
    st.stop()

features = df[MODEL_FEATURES].copy()
df["xG"] = model.predict_proba(
    preprocessor.transform(features)
)[:, 1]

goal_col = find_column(df, ["goal", "is_goal", "shot_goal"])
player_col = find_column(df, ["player", "player_name", "name"])
team_col = find_column(df, ["team", "team_name"])
match_col = find_column(df, ["match_id"])

if goal_col:
    df[goal_col] = pd.to_numeric(df[goal_col], errors="coerce")
    df["Actual Goal"] = df[goal_col].fillna(0).astype(int)
    df["Actual Goal"] = df["Actual Goal"].clip(0, 1)

# ---------- Sidebar ----------
st.sidebar.markdown("## ⚽ Match Centre")
st.sidebar.caption("Football xG Analytics")

if team_col:
    teams = sorted(df[team_col].dropna().astype(str).unique())
    selected_teams = st.sidebar.multiselect(
        "Filter teams",
        teams,
        default=[],
        help="Leave empty to include all teams."
    )
else:
    selected_teams = []

if "shot_type" in df.columns:
    shot_types = sorted(df["shot_type"].dropna().unique())
    selected_shot_types = st.sidebar.multiselect(
        "Shot type",
        shot_types,
        default=[]
    )
else:
    selected_shot_types = []

min_xg, max_xg = st.sidebar.slider(
    "xG probability range",
    min_value=0.0,
    max_value=1.0,
    value=(0.0, 1.0),
    step=0.05
)

filtered = df.copy()

if selected_teams and team_col:
    filtered = filtered[
        filtered[team_col].astype(str).isin(selected_teams)
    ]

if selected_shot_types:
    filtered = filtered[
        filtered["shot_type"].isin(selected_shot_types)
    ]

filtered = filtered[
    filtered["xG"].between(min_xg, max_xg)
].copy()

st.sidebar.divider()
st.sidebar.caption("Model: XGBoost")
st.sidebar.caption("Data: StatsBomb open event data")

# ---------- Navigation ----------
page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Shot Analysis",
        "Team Analysis",
        "Player Analysis",
        "Shot Predictor"
    ],
    label_visibility="collapsed"
)

# ---------- Overview ----------
if page == "Overview":
    st.subheader("📊 Performance Overview")
    st.caption(f"Showing {len(filtered):,} shots after applying your filters.")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Shots", f"{len(filtered):,}")
    c2.metric(
        "Average xG",
        f"{filtered['xG'].mean():.1%}" if len(filtered) else "N/A"
    )
    c3.metric(
        "High-Quality Chances",
        f"{(filtered['xG'] >= 0.30).sum():,}"
    )

    if goal_col:
        c4.metric("Actual Goals", f"{filtered['Actual Goal'].sum():,}")
    else:
        c4.metric("Actual Goals", "Unavailable")

    left, right = st.columns([1.1, 1])

    with left:
        st.markdown("### 🎯 xG Distribution")
        fig = px.histogram(
            filtered,
            x="xG",
            nbins=30,
            labels={"xG": "Expected Goals", "count": "Shots"},
            color_discrete_sequence=["#36a2ff"]
        )
        fig.update_layout(
            template=PLOT_TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=15, b=10),
            height=360
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("### ⚽ Shot Map")
        pitch_df = filtered.dropna(subset=["x", "y"]).copy()

        if len(pitch_df):
            pitch_df["Outcome"] = "Unknown"
            if goal_col:
                pitch_df["Outcome"] = np.where(
                    pitch_df["Actual Goal"] == 1, "Goal", "No goal"
                )

            fig = px.scatter(
                pitch_df,
                x="x",
                y="y",
                color="Outcome",
                size="xG",
                size_max=11,
                hover_data=["xG"],
                color_discrete_map={
                    "Goal": "#41efad",
                    "No goal": "#329bff",
                    "Unknown": "#f4b942"
                }
            )
            fig = add_pitch(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No shots match the current filters.")

    if goal_col and len(filtered):
        st.markdown("### 📈 Predicted xG vs Actual Goals")

        bins = np.linspace(0, 1, 11)
        comparison = filtered.copy()
        comparison["xG Range"] = pd.cut(
            comparison["xG"],
            bins=bins,
            include_lowest=True
        )

        summary = comparison.groupby(
            "xG Range", observed=False
        ).agg(
            Shots=("xG", "size"),
            Average_xG=("xG", "mean"),
            Actual_Goals=("Actual Goal", "sum")
        ).reset_index()

        summary["xG Range"] = summary["xG Range"].astype(str)

        fig = go.Figure()
        fig.add_bar(
            x=summary["xG Range"],
            y=summary["Shots"],
            name="Shots",
            marker_color="#329bff"
        )
        fig.add_bar(
            x=summary["xG Range"],
            y=summary["Actual_Goals"],
            name="Actual goals",
            marker_color="#18c99a"
        )
        fig.update_layout(
            barmode="group",
            template=PLOT_TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Predicted xG range",
            yaxis_title="Count",
            height=380
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔎 Recent Shot Records")
    display_cols = [
        c for c in [
            "match_id", team_col, player_col, "x", "y",
            "shot_type", "body_part", "xG", "Actual Goal"
        ] if c and c in filtered.columns
    ]
    st.dataframe(
        filtered[display_cols].head(100),
        use_container_width=True,
        hide_index=True
    )

# ---------- Shot Analysis ----------
elif page == "Shot Analysis":
    st.subheader("🎯 Shot Analysis")
    st.caption("Explore the location and predicted quality of individual shots.")

    if len(filtered) == 0:
        st.warning("No shots match your filters.")
    else:
        pitch_df = filtered.dropna(subset=["x", "y"]).copy()
        color_by = st.selectbox(
            "Colour shots by",
            ["xG"] + (["Actual Goal"] if goal_col else [])
        )

        if color_by == "Actual Goal" and goal_col:
            pitch_df["Outcome"] = pitch_df["Actual Goal"].map(
                {1: "Goal", 0: "No goal"}
            )
            fig = px.scatter(
                pitch_df, x="x", y="y", color="Outcome",
                hover_data=["xG", "shot_type"],
                color_discrete_map={
                    "Goal": "#41efad", "No goal": "#329bff"
                }
            )
        else:
            fig = px.scatter(
                pitch_df, x="x", y="y",
                color="xG", size="xG", size_max=13,
                color_continuous_scale=["#2678d4", "#35e6ad"],
                hover_data=["shot_type", "body_part"]
            )

        fig = add_pitch(fig)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Highest-xG shots")
        st.dataframe(
            filtered.sort_values("xG", ascending=False).head(20),
            use_container_width=True,
            hide_index=True
        )

# ---------- Team Analysis ----------
elif page == "Team Analysis":
    st.subheader("🏆 Team Analysis")

    if not team_col:
        st.warning("A team-name column was not found in this dataset.")
    elif len(filtered) == 0:
        st.warning("No shots match your filters.")
    else:
        team_stats = filtered.groupby(team_col).agg(
            Shots=("xG", "size"),
            Total_xG=("xG", "sum"),
            Average_xG=("xG", "mean")
        ).reset_index()

        if goal_col:
            goals = filtered.groupby(team_col)["Actual Goal"].sum().reset_index()
            team_stats = team_stats.merge(goals, on=team_col)

        team_stats = team_stats.sort_values("Total_xG", ascending=False)

        metric = st.selectbox(
            "Rank teams by",
            ["Total_xG", "Average_xG", "Shots"]
        )

        fig = px.bar(
            team_stats.head(20).sort_values(metric),
            x=metric,
            y=team_col,
            orientation="h",
            color=metric,
            color_continuous_scale=["#1b6db1", "#37e6ad"],
            labels={
                "Total_xG": "Total xG",
                "Average_xG": "Average xG per shot"
            }
        )
        fig.update_layout(
            template=PLOT_TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=550
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(team_stats, use_container_width=True, hide_index=True)

# ---------- Player Analysis ----------
elif page == "Player Analysis":
    st.subheader("🥇 Player Analysis")

    if not player_col:
        st.warning("A player-name column was not found in this dataset.")
    elif len(filtered) == 0:
        st.warning("No shots match your filters.")
    else:
        player_stats = filtered.groupby(player_col).agg(
            Shots=("xG", "size"),
            Total_xG=("xG", "sum"),
            Average_xG=("xG", "mean")
        ).reset_index()

        if goal_col:
            goals = filtered.groupby(player_col)["Actual Goal"].sum().reset_index()
            player_stats = player_stats.merge(goals, on=player_col)

        player_stats = player_stats.sort_values(
            "Total_xG", ascending=False
        )

        top_n = st.slider("Number of players", 5, 30, 15)

        fig = px.bar(
            player_stats.head(top_n).sort_values("Total_xG"),
            x="Total_xG",
            y=player_col,
            orientation="h",
            color="Total_xG",
            color_continuous_scale=["#1b6db1", "#37e6ad"]
        )
        fig.update_layout(
            template=PLOT_TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=max(350, top_n * 27)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            player_stats.head(top_n),
            use_container_width=True,
            hide_index=True
        )

# ---------- Interactive Shot Predictor ----------
elif page == "Shot Predictor":
    st.subheader("🧮 Interactive Shot xG Predictor")
    st.write(
        "Adjust the shot location and situation to estimate the probability "
        "of scoring. Context defaults use typical values from your dataset."
    )

    st.info(
        "The prediction depends on goalkeeper and defender context as well "
        "as shot location. Adjust those estimates to explore different situations."
    )

    context_defaults = {}
    for col in CONTEXT_FEATURES:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        context_defaults[col] = float(values.median()) if len(values) else 0.0

    with st.form("xg_predictor"):
        left, right = st.columns(2)

        with left:
            shot_x = st.slider("Shot X coordinate", 0.0, 120.0, 110.0)
            shot_y = st.slider("Shot Y coordinate", 0.0, 80.0, 40.0)

            shot_type = st.selectbox(
                "Shot type",
                sorted(df["shot_type"].dropna().unique())
            )
            body_part = st.selectbox(
                "Body part",
                sorted(df["body_part"].dropna().unique())
            )

        with right:
            technique = st.selectbox(
                "Technique",
                sorted(df["technique"].dropna().unique())
            )
            under_pressure = st.selectbox(
                "Under pressure",
                sorted(df["under_pressure"].dropna().unique())
            )

            with st.expander("Advanced defensive context"):
                goalkeeper_distance = st.slider(
                    "Distance to goalkeeper (m)", 0.0, 30.0,
                    float(np.clip(context_defaults["goalkeeper_distance"], 0, 30)),
                    0.5
                )
                goalkeeper_goal_distance = st.slider(
                    "Goalkeeper distance to goal (m)", 0.0, 30.0,
                    float(np.clip(context_defaults["goalkeeper_goal_distance"], 0, 30)),
                    0.5
                )
                nearest_defender_distance = st.slider(
                    "Nearest defender distance (m)", 0.0, 30.0,
                    float(np.clip(context_defaults["nearest_defender_distance"], 0, 30)),
                    0.5
                )
                defenders_within_5m = st.slider(
                    "Defenders within 5m", 0, 10,
                    int(np.clip(round(context_defaults["defenders_within_5m"]), 0, 10))
                )
                defenders_in_shot_lane = st.slider(
                    "Defenders in shot lane", 0, 10,
                    int(np.clip(round(context_defaults["defenders_in_shot_lane"]), 0, 10))
                )
                visible_defender_count = st.slider(
                    "Visible defenders", 0, 20,
                    int(np.clip(round(context_defaults["visible_defender_count"]), 0, 20))
                )
                visible_teammate_count = st.slider(
                    "Visible teammates", 0, 20,
                    int(np.clip(round(context_defaults["visible_teammate_count"]), 0, 20))
                )

        submitted = st.form_submit_button(
            "⚡ Predict xG",
            type="primary",
            use_container_width=True
        )

    if submitted:
        dx = 120 - shot_x
        dy = 40 - shot_y

        distance = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(
            abs(8 * dx),
            dx**2 + (36 - shot_y) * (44 - shot_y)
        )

        shot = pd.DataFrame([{
            "x": shot_x,
            "y": shot_y,
            "distance": distance,
            "angle": angle,
            "goalkeeper_distance": goalkeeper_distance,
            "goalkeeper_goal_distance": goalkeeper_goal_distance,
            "nearest_defender_distance": nearest_defender_distance,
            "defenders_within_5m": defenders_within_5m,
            "defenders_in_shot_lane": defenders_in_shot_lane,
            "visible_defender_count": visible_defender_count,
            "visible_teammate_count": visible_teammate_count,
            "shot_type": shot_type,
            "body_part": body_part,
            "technique": technique,
            "under_pressure": under_pressure
        }])[MODEL_FEATURES]

        probability = model.predict_proba(
            preprocessor.transform(shot)
        )[0, 1]

        result_col, pitch_col = st.columns([1, 1.4])

        with result_col:
            st.markdown("### Predicted goal probability")
            st.metric("Expected Goals (xG)", f"{probability:.1%}")
            st.progress(float(probability))
            st.caption(
                "For example, 0.20 xG means an estimated 20% scoring "
                "probability for a shot with these inputs."
            )

            if probability >= 0.30:
                st.success("High-quality chance")
            elif probability >= 0.10:
                st.info("Moderate-quality chance")
            else:
                st.warning("Low-probability chance")

        with pitch_col:
            shot_plot = pd.DataFrame([{"x": shot_x, "y": shot_y, "xG": probability}])
            fig = px.scatter(
                shot_plot,
                x="x",
                y="y",
                size="xG",
                size_max=28,
                color="xG",
                color_continuous_scale=["#2684ff", "#37e6ad"],
                hover_data={"xG": ":.1%"}
            )
            fig = add_pitch(fig)
            fig.update_layout(height=330, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

st.divider()
st.markdown(
    '<p class="small-note">Model: XGBoost · Data: StatsBomb open event data · '
    'xG values are estimates, not guarantees of scoring. '
    'Predictions depend on the supplied shot and defensive context.</p>',
    unsafe_allow_html=True
)

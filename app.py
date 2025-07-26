import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
import numpy as np

@st.cache_data
def load_data():
    file_path = "2025-swing-data-2.csv"
    df = pd.read_csv(file_path)
    df = df.dropna()
    df.set_index("last_name, first_name", inplace=True)
    df = df.drop(columns=[col for col in ["player_id", "year"] if col in df.columns], errors="ignore")
    features = df.select_dtypes(include=["float64", "int64"])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return df, features.columns.tolist(), scaled, df.index.tolist()

df, feature_cols, scaled_data, player_list = load_data()

st.title("MLB Swing Similarity Tool")

selected_player = st.selectbox("Select a player:", player_list)

max_results = min(150, len(player_list) - 1)
num_results = st.slider("Number of similar players to show:", min_value=1, max_value=max_results, value=10)

custom_weights = {
    "avg_swing_speed": 0.7,
    "attack_direction": 0.6,
    "avg_swing_length": 1.0,
    "attack_angle": 1.1,
    "vertical_swing_path": 1.1
}
weight_vector = [custom_weights.get(col, 1.0) for col in feature_cols]
weighted_data = scaled_data * weight_vector

cov_matrix = np.cov(weighted_data, rowvar=False)
cov_matrix_inv = np.linalg.inv(cov_matrix)

if selected_player:
    idx = player_list.index(selected_player)
    selected_vec = weighted_data[idx]

    distances = np.array([
        mahalanobis(selected_vec, weighted_data[i], cov_matrix_inv)
        for i in range(len(weighted_data))
    ])
    similarities = 1 / (1 + distances)

    # Create dataframe with all players, similarity, and stats
    all_stats = df.reset_index()
    similarity_df = pd.DataFrame({
        "Player": player_list,
        "Similarity Score (Higher = More Similar)": similarities
    })
    # Merge stats into similarity_df
    full_df = pd.merge(similarity_df, all_stats, left_on="Player", right_on="last_name, first_name", how="left")
    full_df = full_df.drop(columns=["last_name, first_name"])

    # Move selected player to top and show only top N similar
    selected_row = full_df[full_df["Player"] == selected_player].copy()
    selected_row.iloc[0, full_df.columns.get_loc("Similarity Score (Higher = More Similar)")] = 1.0  # Ensure 1.0 for selected
    others = full_df[full_df["Player"] != selected_player]
    top_similar = others.sort_values(by="Similarity Score (Higher = More Similar)", ascending=False).head(num_results)
    result_table = pd.concat([selected_row, top_similar], ignore_index=True)

    st.subheader(f"Top {num_results} players similar to **{selected_player}** (showing stats):")
    st.dataframe(result_table)

    csv = result_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Similar Players and Stats as CSV",
        data=csv,
        file_name=f"{selected_player}_similar_players_with_stats.csv",
        mime="text/csv"
    )

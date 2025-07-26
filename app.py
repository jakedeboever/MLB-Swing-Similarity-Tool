import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

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


if selected_player:
    idx = player_list.index(selected_player)

    distances = euclidean_distances([weighted_data[idx]], weighted_data)[0]

    similarity_df = pd.DataFrame({
        "Player": player_list,
        "Similarity Score (Lower = More Similar)": distances
    })

    similarity_df = similarity_df[similarity_df["Player"] != selected_player]
    similarity_df = similarity_df.sort_values(by="Similarity Score (Lower = More Similar)")
    top_similar = similarity_df.head(num_results).reset_index(drop=True)


    st.subheader(f"Top {num_results} players similar to **{selected_player}**:")
    st.dataframe(top_similar)


    csv = top_similar.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Similar Players as CSV",
        data=csv,
        file_name=f"{selected_player}_similar_players.csv",
        mime="text/csv"
    )


    with st.expander("See Selected Player's Stats"):
        st.dataframe(df.loc[selected_player].to_frame().T)

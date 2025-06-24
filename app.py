import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

# ---- Load and prepare your data from a local CSV ----
@st.cache_data
def load_data():
    file_path = "2025-swing-data.csv"  # Replace with your file name if needed
    df = pd.read_csv(file_path)
    df = df.dropna()

    # Set player name as index
    df.set_index("Player Name", inplace=True)

    # Drop 'player_id' and 'year' if they exist
    df = df.drop(columns=[col for col in ["player_id", "year"] if col in df.columns], errors="ignore")

    # Use only numerical columns for similarity
    features = df.select_dtypes(include=["float64", "int64"])

    # Normalize stats
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    return df, features.columns, scaled, df.index.tolist()

# Load data
df, feature_cols, scaled_data, player_list = load_data()

# ---- Streamlit App UI ----
st.title("MLB Hitter Swing Similarity Tool")

# Player selection
selected_player = st.selectbox("Select a player:", player_list)

# Slider to choose number of similar players to display
max_results = min(100, len(player_list) - 1)
num_results = st.slider("Number of similar players to show:", min_value=1, max_value=max_results, value=10)

if selected_player:
    idx = player_list.index(selected_player)
    
    # Compute similarity scores
    distances = euclidean_distances([scaled_data[idx]], scaled_data)[0]
    
    similarity_df = pd.DataFrame({
        "Player": player_list,
        "Similarity Score (Lower = More Similar)": distances
    })

    # Exclude the selected player from results
    similarity_df = similarity_df[similarity_df["Player"] != selected_player]
    
    # Sort and limit the number of rows shown
    similarity_df = similarity_df.sort_values(by="Similarity Score (Lower = More Similar)")
    top_similar = similarity_df.head(num_results).reset_index(drop=True)
    
    st.subheader(f"Top {num_results} players similar to **{selected_player}**:")
    st.dataframe(top_similar)

    # ---- Download button ----
    csv = top_similar.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Similar Players as CSV",
        data=csv,
        file_name=f"{selected_player}_similar_players.csv",
        mime="text/csv"
    )

    with st.expander("See Selected Player's Stats"):
        st.dataframe(df.loc[selected_player].to_frame().T)

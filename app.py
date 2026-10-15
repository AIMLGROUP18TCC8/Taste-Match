import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
BASE = Path(__file__).parent

st.set_page_config(page_title="TasteMatch", page_icon="🎧", layout="centered",)


# --- USER LOGIN ---
from supabase import create_client

if "supabase" not in st.session_state:
    st.session_state.supabase = create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )

db = st.session_state.supabase

if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    st.title("Welcome to TasteMatch")

    login_tab, signup_tab = st.tabs(["Log in", "Create account"])

    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            login = st.form_submit_button("Log in")
        

        if login:
            try:
                result = db.auth.sign_in_with_password({
                    "email": email,
                    "password": password
                })
                st.session_state.user = result.user
                st.rerun()
            except Exception:
                st.error("Login failed. Check your details.")

        if st.button("Forgot password?"):
            st.session_state["show_forgot_password"] = True
        if st.session_state.get("show_forgot_password", False):
            st.subheader("Reset your password")

            email = st.text_input(
                "Enter your registered email address",
                key="recovery_email"
            )

            if st.button("Send recovery email"):
                if email.strip():
                    try:
                        db.auth.reset_password_for_email(
                            email.strip()
                        )

                        st.success(
                            "If an account exists for that email, "
                            "you will receive password recovery instructions."
                        )

                    except Exception:
                        st.error(
                            "We couldn't send the recovery email. "
                            "Please try again."
                        )
    with signup_tab:
        with st.form("signup_form"):
            display_name = st.text_input("Your name")
            signup_email = st.text_input("Email", key="signup_email")
            signup_password = st.text_input(
                "Create a password",
                type="password",
                key="signup_password"
            )
            signup = st.form_submit_button("Create account")

        # Process the form AFTER it has been submitted
        if signup:
            if not display_name.strip() or not signup_email.strip():
                st.warning("Enter your name and email.")

            elif len(signup_password) < 8:
                st.warning("Use a password of at least 8 characters.")

            else:
                try:
                    result = db.auth.sign_up({
                        "email": signup_email.strip(),
                        "password": signup_password,
                        "options": {
                            "data": {
                                "display_name": display_name.strip()
                            }
                        }
                    })

                    if result.session:
                        st.session_state.user = result.user
                        st.rerun()
                    else:
                        st.success(
                            "Account created! Check your email "
                            "to confirm your account, then log in."
                        )

                except Exception as e:
                    st.error(f"Signup failed: {e}")
    st.stop()
user = st.session_state.user
user_id = user.id

st.sidebar.write(f"Signed in as **{user.email}**")

if st.sidebar.button("Log out"):
    db.auth.sign_out()
    st.session_state.user = None
    st.session_state.pop("selected_songs", None)
    st.rerun()

# --- LOAD USER PROFILE ---
profile_result = (
    db.table("user_profiles")
    .select("*")
    .eq("user_id", user_id)
    .execute()
)

if profile_result.data:
    saved_profile = profile_result.data[0]
else:
    saved_profile = {
        "user_id": user_id,
        "display_name": user.user_metadata.get(
            "display_name", user.email
        ),
        "movie_preferences": {},
        "music_preferences": {}
    }

    db.table("user_profiles").insert(saved_profile).execute()

movie_preferences = saved_profile.get("movie_preferences") or {}
music_preferences = saved_profile.get("music_preferences") or {}
with st.sidebar:
    st.header("GROUP 18,AI/ML TCC8")
    st.title("About this model")

    st.subheader("Movie Model")
    st.write("**Algorithm:** K-Means Clustering")
    st.write("**Input:** Movie genres")
    st.write("**Output:** Movie taste groups")
    st.write("**Purpose:** Groups movies with similar genres.")

    st.divider()

    st.subheader("Music Model")
    st.write("**Algorithm:** K-Means Clustering")
    st.write("**Input:** Music features")
    st.write("**Output:** Music taste groups")
    st.write("**Purpose:** Groups songs with similar characteristics.")

    st.info(
        "This app explores similarities in movie and music preferences."
        " It uses K-Means clustering to group movies and songs based on their features."
        "It Recommends based on the features used by each model."
    )
st.title("🎬 TasteMatch 🎧")
st.subheader("Discover movies and music that match your taste.")

# Create the two tabs
movie_tab, music_tab = st.tabs(["🎬 Movies", "🎵 Music"], width= 700)

# Movie tab
with movie_tab:
    st.subheader("Movie Taste Finder")
    st.write("Find your movie taste")
    movie_data = joblib.load(BASE / "movies" / "movie_model.pkl")
    movies = pd.read_csv(BASE / "movies" / "clustered_movies.csv")
   

    kmeans = movie_data["kmeans"]
    scaler = movie_data["scaler"]
    features = movie_data["features"]
    cluster_names = movie_data["cluster_names"]
    genre_columns = movie_data["genre_columns"]

    genre_options = sorted(genre_columns)
     

    chosen = st.multiselect(
        "Choose genres you enjoy",
        genre_options
    )

   
    if st.button("Find my movie group", key="movie_button"):
        if len(chosen) < 3:
            st.warning("Please select at least three genres.")
        else:
                # Load the original movie features
                original_data = pd.read_csv(
                    BASE / "movies" / "data.csv"
                )

                # Create a profile using numerical averages
                profile = {}

                for feature in features:
                    if feature in genre_columns:
                        profile[feature] = int(feature in chosen)
                    elif feature in original_data.columns:
                        profile[feature] = original_data[feature].mean()
                    else:
                        profile[feature] = 0

                # Arrange features in the training order
                user_data = pd.DataFrame(
                    [[profile[f] for f in features]],
                    columns=features
                )

                # Use the original fitted scaler and K-Means
                user_scaled = scaler.transform(user_data)
                cluster = int(kmeans.predict(user_scaled)[0])

                name = cluster_names.get(
                    cluster, f"Cluster {cluster}"
                )

                st.success(f"Your movie group: {name}")

                recommendations = movies[
                    movies["Cluster"] == cluster
                ]

                display_columns = [
                    col for col in [
                        "Movie Name", "Genre",
                        "Movie Rating", "Year of Release"
                    ] if col in recommendations.columns
                ]

                st.dataframe(
                    recommendations[display_columns]
                    .head(25),
                    hide_index=True
                )


with music_tab:
    st.subheader("Music Taste Finder")
    st.write("Find Your Music Taste 🎵")
    st.write("Add 3–5 songs you love, and we'll discover your music taste!")

    # Load your music dataset and trained model
    music = pd.read_csv(BASE / "models" / "clustered_music.csv")

    kmeans = joblib.load(BASE / "models" / "music_kmeans.pkl")
    scaler = joblib.load(BASE / "models" / "music_scaler.pkl")
    features = joblib.load(BASE / "models" / "music_features.pkl")
    cluster_names = joblib.load(BASE / "models" / "music_cluster_names.pkl")

    # Create labels people can recognize
    music["song_choice"] = (
        music["track_name"].fillna("Unknown") + " — " +
        music["artists"].fillna("Unknown") + " (" +
        music["track_genre"].fillna("Unknown") + ")"
    )

    # Remember selected songs when the app refreshes
    if "selected_songs" not in st.session_state:
        st.session_state.selected_songs = []

    st.write(
        f"Songs selected: {len(st.session_state.selected_songs)}/5"
    )

    # Search by song title or artist
    search = st.text_input(
        "Search for a song or artist",
        placeholder="e.g. Wizkid, Adele, Calm Down..."
    )

    if search:
        results = music[
            music["song_choice"].str.contains(
                search, case=False, na=False, regex=False
            )
        ]["song_choice"].drop_duplicates().head(100).tolist()

        if results:
            chosen_song = st.selectbox(
                "Choose a matching song",
                results,
                key="music_search_result"
            )

            if st.button("Add song"):
                if chosen_song in st.session_state.selected_songs:
                    st.warning("You've already added that song.")
                elif len(st.session_state.selected_songs) >= 5:
                    st.warning("You can select a maximum of 5 songs.")
                else:
                    st.session_state.selected_songs.append(chosen_song)
                    st.rerun()
        else:
            st.info("No matches found. Try another song or artist.")

    # Display selected songs
    if st.session_state.selected_songs:
        st.subheader("Your selected songs")

        for song in st.session_state.selected_songs:
            st.write(f"🎵 {song}")

        if st.button("Clear selected songs"):
            st.session_state.selected_songs = []
            st.rerun()

    # Predict taste from all selected songs
    if st.button("Find My Music Taste", key="music_button"):
        if len(st.session_state.selected_songs) < 3:
            st.warning("Please add at least 3 songs first.")
        else:
            selected = music[
                music["song_choice"].isin(
                    st.session_state.selected_songs
                )
            ].drop_duplicates("song_choice")

            # Create one profile from the selected songs
            profile = selected[features].mean().to_frame().T

            # Predict the user's overall music group
            scaled_profile = scaler.transform(profile)
            cluster = int(kmeans.predict(scaled_profile)[0])

            name = cluster_names.get(cluster, f"Group {cluster}")
            st.success(f"Your music taste: {name}")

            # Recommend songs from the predicted group
            
# Recommend songs based on the selected songs' audio features
            selected_features = selected[features]

            # Scale the selected songs using the existing scaler
            selected_scaled = scaler.transform(selected_features)

            # Find the user's average music profile
            user_profile = selected_scaled.mean(axis=0)

            # Scale all songs using the same scaler
            all_scaled = scaler.transform(music[features])

            # Calculate how close each song is to the user's profile
            distances = np.linalg.norm(
                all_scaled - user_profile,
                axis=1
            )

            music["distance"] = distances

            # Remove songs the user already selected
            recommendations = music[
                ~music["song_choice"].isin(
                    st.session_state.selected_songs
                )
            ].copy()

            # Closest songs first
            recommendations = recommendations.sort_values(
                "distance",
                ascending=True
            )

            st.subheader("Songs picked for your taste 🎶")

            columns = [
                col for col in
                ["track_name", "artists", "track_genre"]
                if col in recommendations.columns
            ]

            st.dataframe(
                recommendations[columns].drop_duplicates().head(25),
                hide_index=True
            )

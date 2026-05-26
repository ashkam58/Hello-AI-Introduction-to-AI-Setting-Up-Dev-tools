import time
import random
import pandas as pd
from textblob import TextBlob
from colorama import init, Fore, Style

# Initialize colorama for colored terminal text
init(autoreset=True)

# 1. Safely load the dataset
try:
    df = pd.read_csv("imdb_top_1000.csv")
except FileNotFoundError:
    print(Fore.RED + "Error: The file 'imdb_top_1000.csv' was not found. Please make sure it's in your folder!")
    raise SystemExit

# 2. Extract a clean list of unique genres
genres = sorted({g.strip() for xs in df["Genre"].dropna().str.split(", ") for g in xs})

# Helper function for a quick processing animation
def dots():
    for _ in range(3):
        print(Fore.YELLOW + ".", end="", flush=True)
        time.sleep(0.5)

# Helper function to convert polarity into a readable sentiment emoji
def senti(p):
    return "Positive 😊" if p > 0 else "Negative 😞" if p < 0 else "Neutral 😐"

# 3. AI Recommendation Logic (Filters by Genre, Rating, and Mood)
def ai_recommend(genre=None, mood=None, rating=None, n=5):
    d = df
    if genre:
        d = d[d["Genre"].str.contains(genre, case=False, na=False)]
    if rating is not None:
        d = d[d["IMDB_Rating"] >= rating]
        
    if d.empty:
        return "No suitable movie recommendations found."
        
    # Shuffle for variety
    d = d.sample(frac=1).reset_index(drop=True)
    need_nonneg = bool(mood)
    out = []
    
    for _, r in d.iterrows():
        ov = r.get("Overview")
        if pd.isna(ov):
            continue
            
        pol = TextBlob(ov).sentiment.polarity
        # If user entered a mood, try to match with non-negative overview sentiment
        if (not need_nonneg) or pol >= 0:
            out.append((r["Series_Title"], r["Genre"], r["IMDB_Rating"], ov, pol))
            
        if len(out) == n:
            break
            
    return out if out else "No suitable movie recommendations found."

# 4. Random Recommendation Logic
def random_recommend():
    r = df.sample(n=1).iloc[0] # Pick 1 random row
    ov = r.get("Overview")
    pol = TextBlob(ov).sentiment.polarity if not pd.isna(ov) else 0
    return [(r["Series_Title"], r["Genre"], r["IMDB_Rating"], ov, pol)]

# 5. Display function to format the output nicely
def show(recs, name):
    print(Fore.YELLOW + f"\n🍿 Movie Recommendations for {name}:")
    for i, (title, genre, rating, overview, pol) in enumerate(recs, 1):
        print(f"\n{Fore.CYAN}{i}. 🎥 {title}")
        print(f"{Fore.GREEN}Genre:{Style.RESET_ALL} {genre} | {Fore.GREEN}Rating:{Style.RESET_ALL} {rating}")
        print(f"{Fore.WHITE}Overview: {overview}")
        print(f"{Fore.MAGENTA}Sentiment: {senti(pol)} (Polarity: {pol:.2f})")

# Helper function to get valid genre input
def get_genre():
    print(Fore.GREEN + "\nAvailable Genres: ", end="")
    for i, g in enumerate(genres, 1):
        if i % 5 == 0: # Line break every 5 genres for neatness
            print(f"{Fore.CYAN}{i}. {g}")
        else:
            print(f"{Fore.CYAN}{i}. {g}", end="  ")
    print("\n")
    
    while True:
        x = input(Fore.YELLOW + "Enter genre number or name: ").strip()
        if x.isdigit() and 1 <= int(x) <= len(genres):
            return genres[int(x)-1]
        x = x.title()
        if x in genres:
            return x
        print(Fore.RED + "Invalid input. Try again.\n")

# Helper function to get valid rating input
def get_rating():
    while True:
        x = input(Fore.YELLOW + "Enter minimum IMDB rating (7.6-9.3) or 'skip': ").strip()
        if x.lower() == "skip":
            return None
        try:
            r = float(x)
            if 7.6 <= r <= 9.3:
                return r
            print(Fore.RED + "Rating out of range. Try again.\n")
        except ValueError:
            print(Fore.RED + "Invalid input. Try again.\n")

# 6. Main execution loop
def main():
    print(Fore.BLUE + "🎥 Welcome to your Personal Movie Recommendation Assistant! 🎥\n")
    name = input(Fore.YELLOW + "What's your name? ").strip()
    print(f"\n{Fore.GREEN}Great to meet you, {name}!\n")
    
    while True:
        choice = input(Fore.YELLOW + "\nWould you like an (1) AI-based recommendation or (2) Random recommendation? (Enter 1 or 2): ").strip()
        
        if choice == "1":
            genre = get_genre()
            mood = input(Fore.YELLOW + "How do you feel today? (Describe your mood): ").strip()
            
            print(Fore.BLUE + "\nAnalyzing mood", end="", flush=True)
            dots()
            mp = TextBlob(mood).sentiment.polarity
            md = "positive 😊" if mp > 0 else "negative 😞" if mp < 0 else "neutral 😐"
            print(f"\n{Fore.GREEN}Your mood is {md} (Polarity: {mp:.2f}).\n")
            
            rating = get_rating()
            
            print(f"{Fore.BLUE}\nFinding movies for {name} ", end="", flush=True)
            dots()
            
            recs = ai_recommend(genre=genre, mood=mood, rating=rating, n=3)
            if isinstance(recs, str):
                print(Fore.RED + "\n" + recs + "\n")
            else:
                show(recs, name)
                
        elif choice == "2":
            print(Fore.BLUE + "\nFetching a random movie ", end="", flush=True)
            dots()
            recs = random_recommend()
            show(recs, name)
            
        else:
            print(Fore.RED + "Invalid choice. Please enter 1 or 2.")
            continue
            
        # Check if the user wants to go again
        a = input(Fore.YELLOW + "\nWould you like more recommendations? (yes/no): ").strip().lower()
        if a != "yes":
            print(Fore.GREEN + f"\nEnjoy your movie picks, {name}! 🎬🍿\n")
            break

if __name__ == "__main__":
    main()
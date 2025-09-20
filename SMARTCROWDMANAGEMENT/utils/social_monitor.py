import tweepy
import time

# -----------------------------
# --- Twitter API keys ---
# -----------------------------
API_KEY = "YOUR_API_KEY"
API_SECRET = "YOUR_API_SECRET"
ACCESS_TOKEN = "YOUR_ACCESS_TOKEN"
ACCESS_TOKEN_SECRET = "YOUR_ACCESS_TOKEN_SECRET"

auth = tweepy.OAuth1UserHandler(API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

# -----------------------------
# --- Fetch trending hashtags ---
# -----------------------------
def fetch_trending_hashtags(woeid=1, top_n=10):
    """
    Get top N trending hashtags
    woeid=1 -> worldwide
    """
    trends = api.get_place_trends(id=woeid)
    hashtags = []
    for t in trends[0]["trends"]:
        if t["name"].startswith("#"):
            hashtags.append((t["name"], t["tweet_volume"] or 0))
    hashtags.sort(key=lambda x: x[1], reverse=True)
    return hashtags[:top_n]

# -----------------------------
# --- Filter hashtags related to gatherings ---
# -----------------------------
def filter_gathering_hashtags(hashtags, keywords=None):
    """
    keywords = list of words like ["match", "concert", "festival"]
    """
    if not keywords:
        keywords = ["match", "concert", "festival", "crowd", "event", "gathering"]

    filtered = []
    for tag, volume in hashtags:
        if any(word.lower() in tag.lower() for word in keywords):
            filtered.append((tag, volume))
    return filtered

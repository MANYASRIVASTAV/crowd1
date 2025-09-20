// backend_connector.js

const videoFeedURL = "http://127.0.0.1:5000/video_feed";
const countURL = "http://127.0.0.1:5000/count";

// Set video feed
document.getElementById("videoFeed").src = videoFeedURL;

// Update people count every second
setInterval(async () => {
    try {
        let res = await fetch(countURL);
        let data = await res.json();
        document.getElementById("peopleCount").innerText = data.peopleCount;
    } catch (err) {
        console.error("Error fetching count:", err);
    }
}, 1000);

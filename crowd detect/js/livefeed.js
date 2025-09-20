const videoInput = document.getElementById("videoInput");
const videoPlayer = document.getElementById("videoPlayer");
const peopleCountElement = document.querySelector(".people-count");
const generateHeatmapBtn = document.getElementById("generateHeatmap");

// Handle video upload and preview
videoInput.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (file) {
        const url = URL.createObjectURL(file);
        videoPlayer.src = url;
        videoPlayer.play();

        // Placeholder: simulate backend processing
        peopleCountElement.textContent = "Processing video...";
        setTimeout(() => {
            const simulatedCount = Math.floor(Math.random() * 100);
            peopleCountElement.textContent = `People Count: ${simulatedCount}`;
        }, 3000);
    }
});

// Navigate to Heatmap page after processing
generateHeatmapBtn.addEventListener("click", () => {
    // TODO: In future, send backend request to generate heatmap
    window.location.href = "heatmap.html";
});

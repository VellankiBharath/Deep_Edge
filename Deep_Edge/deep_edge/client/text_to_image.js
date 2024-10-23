document.getElementById('promptForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    // Get the user's prompt input
    const prompt = document.getElementById('prompt').value;

    // Prepare the request data
    const data = {
        prompt: prompt
    };

    try {
        // Send the request to the Flask API
        const response = await fetch('http://127.0.0.1:5000/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
            
        });

        // Parse the response
        const result = await response.json();

        // Display the generated image
        const imageElement = document.getElementById('generatedImage');
        imageElement.src = 'data:image/png;base64,' + result.generated_image;
        imageElement.style.display = 'block';
// Add this inside the try block in text_to_image.js
        console.log("Response received:", result);

        // Display CLIP results
        const clipResultsElement = document.getElementById('clipScores');
        clipResultsElement.innerHTML = '';  // Clear previous results
        for (let concept in result.clip_results) {
            const score = result.clip_results[concept];
            const listItem = document.createElement('li');
            listItem.textContent = `${concept}: ${score.toFixed(4)}`;
            clipResultsElement.appendChild(listItem);
        }

        // Display SAM results
        const samResultsElement = document.getElementById('samScores');
        samResultsElement.textContent = `Segmentation Scores: ${result.sam_scores.join(', ')}`;

    } catch (error) {
        console.error('Error generating image:', error);
        alert('Failed to generate image. Please try again.');
    }
});

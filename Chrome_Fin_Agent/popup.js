document.getElementById('getAnalysis').addEventListener('click', async () => {
    const stockName = document.getElementById('stockName').value.trim();
    const resultContainer = document.getElementById('result');
    
    if (!stockName) {
        resultContainer.textContent = "Please enter a stock name.";
        return;
    }

    resultContainer.textContent = "Fetching data...";
    
    try {
        const response = await fetch(`http://0.0.0.0:8000/analysis/financial`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ stock_name: stockName })
        });
        
        const data = await response.json();
        resultContainer.textContent = JSON.stringify(data, null, 2);
    } catch (error) {
        resultContainer.textContent = "Error fetching data. Make sure FastAPI is running.";
    }
});

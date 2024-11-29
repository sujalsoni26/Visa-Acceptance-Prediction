import express from "express";
import bodyParser from "body-parser";
import { spawn } from "child_process";

const app = express();

app.use(bodyParser.urlencoded({ extended: true })); // Ensure body is parsed as JSON

app.get('/', (req, res) => {
    res.sendFile("C:/Users/acer/Desktop/ML_Project/front.html");
});

app.post("/predict", async (req, res) => {
    const data = req.body; // Get the JSON data from the request body
    console.log('Received data:', data);
    
    // Spawn the Python process and pass the input data via stdin
    const python = spawn('python3', ['model_website.py']);
    
    python.stdin.write(JSON.stringify(data)); // Write JSON data to stdin
    python.stdin.end();  // Close stdin

    let result = '';
    let error = '';

    // Introduce a delay before handling the stdout data
    setTimeout(() => {
        python.stdout.on('data', (data) => {
            console.log('Data from Python:', data.toString()); // Log the output from Python
            result += data.toString();
        });

        python.stderr.on('data', (data) => {
            console.error('Python stderr:', data.toString()); // Log any error from Python
            error += data.toString();
        });

        python.on('close', (code) => {
            if (code !== 0) {
                console.error(`Python process exited with code ${code}`);
                console.error(`stderr: ${error}`);
                return res.status(500).send('Internal Server Error');
            }
            console.log('Prediction result:', result); // Log the final result
            res.send(result);  // Send the result back to the client
        });
    }, 0);  // Delay for 2 seconds (2000 ms)

});

app.listen(8000, () => {
    console.log("Listening on port localhost:8000");
});

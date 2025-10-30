# k-NN Ad Click Predictor

This project is an implementation of the k-Nearest Neighbors (k-NN) algorithm using Python, NumPy, and OOP principles. It predicts whether a user will click an ad (1) or not (0) based on their Age, Time on Site, and Estimated Income.

## How to Run

1.  Clone the repository and set up a virtual environment:
    ```bash
    git clone https://github.com/mitali-dxt/Ad-Click-Predictor.git
    cd Ad-Click-Predictor
    python -m venv venv
    venv\Scripts\activate (Windows)
    ```

2.  Install requirements:
    ```bash
    pip install -r requirements.txt
    ```

3.  (Optional) View the data:
    ```bash
    python visualize_data.py
    ```

4.  Run the main application:
    ```bash
    python -m src.main
    ```

## Output

Running `main.py` will execute the tests. The console will show the final accuracy and the results.

A `project.log` file will be created, containing a detailed, timestamped log of the entire execution.

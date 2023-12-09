import subprocess

# List of Python files in sequence
python_files = [
    "peak_detection.py",
    "technical_indicators.py",
    "feature_parameter_tuning.py",
    "hurst_kalman_fft_derivatives.py",
    "feature_evaluation.py",
    "feature_evaluation_vis"
]

# Run each Python file in sequence
for file in python_files:
    try:
        subprocess.run(["python", file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {file}: {e}")

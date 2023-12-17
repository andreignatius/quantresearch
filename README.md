# quantresearch

This repository is an exploration of signal processing indicators if they could yield superior prediction signals for detecting peaks/troughs in forex trading (USD / JPY)

To run this project, create your virtual environment via conda (optional):
```
conda create --prefix ./env python=3.8
conda activate ./env
```

If you are using Mac, install all necessary packages via:
```
python -m pip install -r requirements_mac.txt
```

Otherwise if you are using Windows, install via:
```
python -m pip install -r requirements_win.txt
```

To run baseline model, go to backtest/ folder and run:
```
python benchmark.py
```

To run logistic regression model, go to backtest/ folder and run:
```
python logreg_oop.py
```

To run gradient boosted tree model, go to backtest/ folder and run:
```
python gbt_oop.py
```

To run tensorflow neural network model, go to backtest/ folder and run:
```
python tf_oop.py
```

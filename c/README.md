# Description
Logistic regression, despite its name, is a linear model for classification rather than regression. Logistic regression is also known in the literature as logit regression, maximum-entropy classification (MaxEnt) or the log-linear classifier. In this model, the probabilities describing the possible outcomes of a single trial are modeled using a [logistic function](http://en.wikipedia.org/wiki/Logistic_function).

# Usage
1. Enter Paracel's home directory  
```cd paracel;``` 
1. Generate test dataset for classification
```python ./tool/datagen.py -m classification -o data.csv -n 10000 -k 100```
2. Set up link library path:  
```export LD_LIBRARY_PATH=your_paracel_install_path/lib```    
3. Create a json file named `cfg.json`, see example in [Parameters](#parameters) section below.  
4. Run (4 workers, local mode in the following example)  
```./prun.py -w 4 -p 2 -c cfg.json -m local your_paracel_install_path/bin/lr```

# Parameters

# Data Format

# Notes

# Reference

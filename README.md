# Descripton
Calcute the highest number of occurrences of k words in input dataset.

# Usage
1. Enter Paracel's home directory  
```cd paracel;``` 
1. Generate test dataset  
```python ./tool/datagen.py -m wc -o data.txt```
2. Set up link library path:  
```export LD_LIBRARY_PATH=your_paracel_install_path/lib```  
3. Create a json file named `cfg.json`, see example in [Parameters](#parameters) section below.
4. Run (4 workers, local mode in the following example)  
  ```./prun.py -w 4 -p 1 -c cfg.json -m local your_paracel_install_path/bin/wc```

# Parameters
  Default parameters are set in a JSON format file. For example, we create a
  cfg.json as below(modify `your_paracel_install_path`):

{    
    "input" : "data.txt",    
    "output" : "./wc_output/",    
    "topk" : 10,    
    "handle_file" : "your_paracel_install_path/lib/libwc_update.so",   
    "update_function" : "wc_updater",   
    "filter_function" : "wc_filter"   
}

In the configuration file, `handle_file`, `update_function` and `filter_function` is the information of the registry function needed in word count program.

# Data Format
Free-format text.

# Notes


# Reference

# CE7454_Project_Fall2018_NTU
CE7454 Project for Deep Learning for Data Science  Fall 2018 NTU 


### Setup

1. Clone this repo into your local PC
2. Download carvana-image-masking-challenge dataset into `/dataset` ([kaggle API](https://github.com/Kaggle/kaggle-api) is recommended)
  - a. `pip install kaggle`
  - b. Add API credentials 
  From https://www.kaggle.com/<username>/account -> select 'Create API Token' -> Place this file in the location `~/.kaggle/kaggle.json` -> 
  In `~/.bashrc`, add this three line: 
  
  ```bash
export KAGGLE_CONFIG_DIR=$KAGGLE_CONFIG_DIR/kaggle.json
export KAGGLE_USERNAME=<username>
export KAGGLE_KEY=your_key
```
  ->`source ~/.bashrc`
  - c. kaggle competitions download -c carvana-image-masking-challenge


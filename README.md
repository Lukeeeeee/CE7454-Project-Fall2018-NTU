# CE7454_Project_Fall2018_NTU
CE7454 Project for Deep Learning for Data Science Fall 2018 NTU


[![Binder](https://mybinder.org/badge.svg)][binder]

[binder]: https://mybinder.org/v2/gh/Lukeeeeee/CE7454_Project_Fall2018_NTU/master

### Environment Set-up

```bash
git clone https://github.com/Lukeeeeee/CE7454_Project_Fall2018_NTU
cd CE7454_Project_Fall2018_NTU
conda env create environment.yml
source activate dlproject
```

### Download dataset

1. Install [kaggle API](https://github.com/Kaggle/kaggle-api): `pip install kaggle`
3. Create API token from `https://www.kaggle.com/<username>/account` and save the file to `~/.kaggle/kaggle.json` 
4. Download the Carvana dataset: `kaggle competitions download -c carvana-image-masking-challenge`
5. Unzip the dataset into to `train.csv` `valid.csv` `test.csv`
  
### Running the crawler

```bash
cd crawler
scrapy crawl cars
```

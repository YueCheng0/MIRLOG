# MIRLOG: Multi-omics integration and regulariza-tion via logistic regression for cancer subtype classi-fication and biomarker identification
![MIRLOG](https://github.com/YueCheng0/MIRLOG/blob/main/data/MIRLOG.png)

# Requirements 
```bash
#Creating a virtual environment
conda create -n MIRLOG python-3.7

#Activate the virtual environment
conda activate MIRLOG

#Installation requirements packages
pip install -r requirments.txt

```


# Usage
```bash

python "Multi-omics integration.py" --moicsdatas ./data/BRCARNA.csv ./data/BRCADNAmeth.csv ./data/BRCAmiRNA.csv --label ./data/BRCAlabel.csv

python model.py --integration_data ./integration_data/Diffusion_enhanced_feature_matrix.csv --label ./data/BRCAlabel.csv

```
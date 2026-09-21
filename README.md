# Hybrid Wildfire Detection Model

This is a stub design for a hybrid wildfire detection model that seeks to predict an area's risk of wildfire a 30 day window. The model processes and aggregates data from three sources (two satellite and one weather data), feeds this aggregated data into a hybrid convolutional-transformer deep learning model, and then output a fire risk score within the previously mentioned time horizon. All data required to perform inference for this model can be freely obtained online! 

## Data Sources & Related Resources 

Sentinel-1 is a satellite that uses Synthetic Aperture Radar (SAR) to image the ground at 20m resolution. You can obtain a Sentinel-1 scene through the Copernicus Open Access Hub (https://dataspace.copernicus.eu/). 

Sentinel-2 is a satellite that uses high-res optics to image the ground at 10m resolution. You can obtain a Sentinel-2 scene through the Copernicus Open Access Hub (https://dataspace.copernicus.eu/). 

ERA5 uses ECMWF reanalysis to collect periodic weather data, such as temperature, humidity, windspeed, etc. You can obtain an ERA5 data package from the Copernicus Climate Data Store (https://cds.climate.copernicus.eu/). 

MTBS (Monitoring Trends in Burn Severity) is a joint USGS/USDA Forest Service program that maps and assesses the burn severity of large fires across the United States. It's the ground truth source of each fire's ignition date, acreage, and burn severity assessment used to define and label the fires in this dataset. You can browse the MTBS data here (https://www.mtbs.gov/).

The EPA (United States Environmental Protection Agency) sets standards for environmental responsibility and human health. The EPA Level III eco-region map identifies 120 unique regions within the United States that vary by ecosystem. It was used to match fires to controls, ensuring environmental region consistency between fire/control pairs. You can browse the level III map here (https://www.epa.gov/eco-research/level-iii-and-iv-ecoregions-continental-united-states).

## Repository Structure 
```
wildfire-pred/
├── data/
│   ├── aggregator/
│   │   └── zonal_aggregator.py
│   └── loaders/
│       ├── era5_preprocessing.py
│       ├── sentinel1_preprocessing.py
│       └── sentinel2_preprocessing.py
├── docs/
│   ├── dataset_README.md
│   ├── devlog.md
│   └── project_state.md
├── model/
│   ├── architecture.py
│   ├── augmentation.py
│   └── dataset.py
├── scripts/
│   └── build_tile_cache.py
│   └── fwi_calculator.py
│   └── hyperparameter_sweep.py
│   └── nbr_burn_severity_calculator.py
│   └── run_inference.py
├── tests/
│   ├── conftest.py
│   ├── test_burn_severity_classification.py
│   ├── test_dataset_imputation.py
│   ├── test_fwi_calculator.py
│   ├── test_sentinel2_indices.py
│   └── test_zonal_aggregator.py
├── training/
│   └── train.py
├── utils/
│   └── geo_utils.py
├── .gitignore
├── LICENSE
├── README.md
└── env.yml
```

## Setup 

This project was built and tested on Python 3.11. Check the env.yml for environment dependencies, conda is the recommended package manager. Run the following in your terminal once conda is installed and working: 
```bash
conda env create -f env.yml
conda activate wildfire-pred
```

## Included Data Set 

A full scene includes Sentinel-1 pre and post fire SAFE files, Sentinel-2 pre and post SAFE files, an ERA-5 grib file, and a metadata.json with details of the contents of each asset within the scene.
Each fire is paired with three controls that match the fires EPA Level III Eco-region of the fire. 125 fires and 375 control scenes collected over seven US states (Alaska, California, Idaho, Montana, Nevada, Oregon, Washington) comprise the full dataset. 
*Important Note:* All scripts described below assume a fire scene follows the flat structure within my dataset, as in all source files are on the 
same directory level that is one level below the scene directory itself. You can find the full dataset I used for model training and download individual scenes here: (https://huggingface.co/datasets/aroon-sankoh/wildfire-prediction).

## Usage 

Once you have identified a dataset for analysis, it is highly recommended to build and cache aggregated tiles before anything else. This way it won't be necessary to repeatedly load and aggregate source data whenever performing model training 
or inference. Investigate and edit the global variables within `scripts/build_tile_cache.py` to ensure the correct source data is used for tiling. Note that this script (and a few others, like the FWI/dataset collection scripts) pulls directly from my own S3 bucket, so you'll need your own AWS credentials and bucket configured if you want to run it yourself against your own scenes. If you just want to run inference or the calculators against the published HuggingFace dataset, you don't need any AWS access at all.

### Training 

After building and caching your ERA-5 grid tiles, you can train a WildfireModel with `training/train.py`. Since only local tensors are used, 
training can efficiently be completed using just CPU. All fire tiles are labeled 1.0 and control tiles are labeled 0.0, the wildfire detection model
was originally built to predict across 3 seperate time horizons so (1 month, 3 month, and 6 months) but due to dataset constraints, all heads 
output the same labels. I decided to leave the functionality of the 3 heads in the model architecture in the case that I decide to add multi-horizon
prediction in the future. Training should take <2 hours and it's generally not recommended to train for longer than 12 epochs for risk of over-fitting. 

### Hyperparameter Sweeps

The hyperparameter sweep script does a random search of different hyperparameter combinations over WildfireModel training config. Each 
trial gets the same split, normalization, and class weighting. The model runs --trial-epochs epochs and allows the user to specify the 
fraction of the dataset used for validation and testing. --n-trials also allows the user to decide how many configurations will be 
tested, with configuration parameters including embedding dimension, temporal hidden dimension, \# of heads, batch size, and more (check the
global variables at the top of the script: `scripts/hyperparameter_sweep.py`). The script outputs the top performing configurations and it is
recommended the user re-train the best config with actual trainings script for proper checkpointing.

### Inference 

Once a single model has completed training, the `scripts/run_inference.py` script allows you 'predict' on a single, unlabeled fire scene. It 
outputs a per-tile fire probability grid that represents the chance of a wildfire occurring within a 1 month period of each ERA-5 tile. The 
output is a full risk grid csv file that is saved to an inference/ directory and is tied to the specific model checkpoint and scene that 
inference was performed on. Summary statistics, such as the \# of the tiles with >0.5 probability of a fire occurring, the top 5 highest risk 
tiles, and the full-scene aggregated mean chance of a fire occurring are printed. 

### FWI Calculator 

The Canadian Fire Weather Index is a useful tool for calculating the chance of a fire occurring in boreal forests, such as those found in Canada, 
Alaska, and parts of the northwestern United States. I re-implemented the FWI calculator to use a benchmark for my full wildfire model performance
in `data/loaders/era5_preprocessing.py`, and used it to validate that my model identified more subtle patterns in time-series and satellite 
data to wildfires. FWI only uses ERA-5 time-series data though and so is less costly to run. See `scripts/fwi_calculator.py` to run a single scene.

### Burn Severity Calculator 

The delta Normalized Burn Ratio is a useful tool for quantifying the burn severity of an area affected by a wildfire. It works by diffing the NBR
of Sentinel-2 pre fire and Sentinel-2 post fire. I implemented a dNBR calculator in `scripts/nbr_burn_severity_calculator.py` and classified it 
using the fire's own MTBS (Monitoring Trends in Burn Severity) calibrated dNBR thresholds. See the aforementioned script for requirements and 
instructions on how to use. 

## Model Performance & Limitations

The model is still a work in progress as its' performance is modest. The current best checkpoint sits around 0.69-0.73 balanced accuracy on my own test split, meaning the model currectly classifies ~70% of scenes correctly. 

Right now the model is doing same-day classification, not true forward forecasting. The ERA-5 window each tile sees runs right up through the ignition/control date itself with zero forecast lead time, so the question it's answering is "does this look like a fire day or a control day" rather than "will this area burn in the next 30 days." I ran an ablation study that truncated the last 7 days of ERA-5 data from every tile to see how much of that ~0.70-0.73 number depended on near-ignition weather specifically, and balanced accuracy only dropped marginally (still 0.70-0.72 range), which suggests accuracy is not related to a same-day-weather signal, but I cannot disprove the framing caveat still stands until I retrain with a real embargoed window.

Also worth noting that I tested the functionality of my inference and burn severity scripts on a retained sample of fire/control pairs that the model was already trained on. So if you decide to run inference on one of the dataset scenes and get a suspiciously strong result, that's the model recognizing data it has already seen. I hope to curate and add a testing dataset that can truly evaluate my model's prediction powers soon. 


## Future Work

With every passing year, larger and more deadly wildfires ravage more of the world. Due to how different eco-systems contribute to different conditions for wildfire ignitions, I limited the region of my model to North America but I'd like to extend it to be able to predict on 
more fire-prone regions globally. Sub-saharan Africa, Australia, and Southern Europe are three regions I'd like to incorporate into future 
training sets for my wildfire prediction model. More immediately, Northern Canada is the first area I would focus on due to how similar the
eco-systems of Alaska and the Northwestern United States are to it. But before that, there are more immediate project tickets to address. See 
`docs/project_state.md` for more.

## AI Assistance
Portions of this project's scripts, tests, and debugging were developed with assistance from Claude (Anthropic), specifically Sonnet 5, used as a coding assistant throughout development. All outputs were reviewed and verified by @AroonSankoh.

## License

This code is released under the MIT License, see the LICENSE file for the full text. The underlying dataset (Sentinel-1/Sentinel-2/ERA5/MTBS-derived) has its own license and attribution terms, laid out separately on the HuggingFace dataset card (https://huggingface.co/datasets/aroon-sankoh/western-us-wildfire-prediction).

## Citation

If you use this code or model in your own work, please cite it as:
```
@software{sankoh2026wildfire,
  author = {Sankoh, Aroon},
  title = {Hybrid Wildfire Detection Model},
  year = {2026},
  url = {https://github.com/aroon-sankoh/western-us-wildfire-pred}
}
```

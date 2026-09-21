---
license: cc-by-4.0
---
# Curated Wildfire Detection & Analysis Dataset
This dataset consists of 125 fire and 375 control 'scenes', where every scene includes 
a Sentinel-1 pre image, Sentinel-1 post image, Sentinel-2 pre image, Sentinel-2 post image, 
ERA5 re-analysis data series, and a json file with metadata on each piece of data. Each fire 
is paired with three controls that match the fires EPA Level III Eco-region of the fire. 
125 fires and 375 control scenes are collected over seven United States states (Alaska, California, 
Idaho, Montana, Nevada, Oregon, Washington) comprise the full dataset. The full dataset totals 
roughly 2.1 TB, so plan accordingly before downloading -- see the Dataset Structure section below 
if you only need a subset.
## Data Sources 
Sentinel-1 is a satellite that uses Synthetic Aperture Radar (SAR) to image the ground at 20m 
resolution. You can obtain a Sentinel-1 scene through the Copernicus Open Access Hub (https://dataspace.copernicus.eu/).
Sentinel-2 is a satellite that uses high-res optics to image the ground at 10m resolution. 
You can obtain a Sentinel-2 scene through the Copernicus Open Access Hub (https://dataspace.copernicus.eu/).
ERA5 uses ECMWF reanalysis to collect periodic weather data, such as temperature, humidity, windspeed, etc. 
You can obtain an ERA5 data package from the Copernicus Climate Data Store (https://cds.climate.copernicus.eu/).
MTBS (Monitoring Trends in Burn Severity) is a joint USGS/USDA Forest Service program that maps and 
assesses the burn severity of large fires across the United States. It's the source of each fire's 
ignition date, acreage, and burn severity assessment used to define and label the fires in this dataset. 
You can browse MTBS data at https://www.mtbs.gov/.
## License & Attribution
This dataset (the compilation, processing, and derived FWI/model-ready structure) is released under 
CC-BY-4.0. That license covers my own added work; it doesn't replace the terms of the underlying sources, 
which require their own attribution regardless of how the derived dataset is licensed:
Sentinel-1 and Sentinel-2 imagery is provided under the Copernicus Sentinel Data Terms and Conditions. 
Required notice: "Contains modified Copernicus Sentinel data [Year]."
ERA5 data is provided under the CC-BY-4.0 licence used by the Copernicus Climate Data Store. Required 
notice: "Contains modified Copernicus Climate Change Service information [Year]. Neither the European 
Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data 
it contains."
MTBS data is a US federal government product and is in the public domain. No formal citation is legally 
required, but MTBS requests acknowledgment; see https://www.mtbs.gov/ for suggested citation language.
## Scene Constraints 
All fires have a total acreage >10k acres. Fire and control ignition dates range from 2016 to present. 
Orbital direction between Sentinel-1 pre and post images is consistent (eg. either both ascending or descending). 
Temporal gaps, or the time difference between images of the same Sentinel type, are limited to 180 days or less.
Sensor time difference hours, or the time difference between a Sentinel-1 pre and a Sentinel-2 pre image (and 
likewise for a Sentinel-1 post image and a Sentinel-2 post image), are limited to 18 hours or less. 
The fire spatial coverage of each source is at least 70%, meaning that each source covers a geographical area 
that accounts for at least 70% of the fire's boundary box. A scene's ERA5 package includes at least 30 days 
of reanalysis data leading up to, but not including, the fire ignition date. The reanalysis data includes 
2-meter temperature (t2m), 2-meter dewpoint (d2m), eastward wind speed (u10), northward wind speed (v10), and 
total precipitation (tp) variables. All specific information on a scene can be found in it's metadata.json. 
## Dataset Structure
Scenes are organized as fires/{state}/{scene_id}/ and controls/{state}/{scene_id}/, where {state} is the 
two-letter US state abbreviation and {scene_id} is the fire or control's name. Each scene folder contains 
five files: a Sentinel-1 pre-fire SAFE product, a Sentinel-1 post-fire SAFE product, a Sentinel-2 pre-fire 
SAFE product, a Sentinel-2 post-fire SAFE product, an ERA5 grib file, and a metadata.json describing all of 
the above. Sentinel files are named {sensor}_{state}_{grid_coordinates}_{date}_{pre_or_post}.SAFE (eg. 
S1_CA_34N117W_20200728_pre.SAFE), and the ERA5 file is named ERA5_{state}_{grid_coordinates}_{ignition_date}.grib.
Each metadata.json includes: event_id and mtbs_post_id (identifiers tying the scene back to its MTBS record), 
fire_name, state, ignition_date, total_acres, days_to_post (days between ignition_date and the post-fire 
reference date used for assessment), orbit_direction, post_source, temporal_gaps (days between a sensor's own 
pre- and post-fire acquisition dates, S1_days and S2_days), sensor_time_difference_hours (hours between the 
Sentinel-1 and Sentinel-2 acquisitions for the pre and post pairs), spatial_bounds (the scene's bounding box), 
spatial_coverage (percent of the fire's boundary box covered by each source), and contents (the filenames of 
each of the five files in the scene folder).
## Control Selection & Validation
Each fire's three controls are drawn from locations sharing the fire's EPA Level III eco-region, so that 
vegetation, climate, and terrain are broadly comparable to the paired fire rather than an arbitrary site 
elsewhere in the country.
<!-- TODO: describe how it was verified that a control site did not itself experience an undocumented fire
during the collection window (e.g. cross-referenced against MTBS records, visual inspection of imagery, etc). -->
## Limitations
This dataset only covers seven western/northwestern US states (Alaska, California, Idaho, Montana, Nevada, 
Oregon, Washington), so it doesn't generalize to other fire-prone regions (the Southeast, the Mediterranean, 
Australia, etc.) without further validation. Fires are restricted to a >10k acre minimum, so the dataset is 
biased toward large fires and won't reflect the risk profile of smaller ones. The fire-to-control ratio is 
1:3, an intentional class imbalance rather than a natural base rate. There is no pre-defined train/validation/test 
split; users should partition the data themselves, ideally by fire event and eco-region to avoid spatial leakage 
between splits.
## Motivation
For a summer project, I decided to build a wildfire detection model that sought to predict an area's risk of wildfire 
within 1, 3, and 6 month periods. The model processes and aggregates data from each of the three data sources above, 
feeds this aggregated data into a hybrid convolutional-transformer and then outputs a risk score. After collecting and 
curating such a large dataset for model training and inference over ~1 month, I decided it'd be a waste not to make my 
data publicly available for other ML practitioners and climate scientists. You can find the repository for my model 
here (https://github.com/AroonSankoh/alaska-wildfire-pred/). I still have some finishing touches to add, which you can
read about in that project's README.MD. I plan to write a full substack article about my project's design process soon, 
I'll link it here when it's finished.
## Citation
If you use this dataset, please cite it as:
```
@dataset{sankoh2026wildfire,
  author = {Sankoh, Aroon},
  title = {Curated Wildfire Detection & Analysis Dataset},
  year = {2026},
  publisher = {Hugging Face},
  url = {https://huggingface.co/datasets/aroon-sankoh/wildfire-prediction}
}
```

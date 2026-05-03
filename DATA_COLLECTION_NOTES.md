# Image2GPS Data Collection Notes

This just summarizes the data collection and preprocessing setup that I've done so far. Note that the cis5190-project-resources contains all the reference code provided to us in the assignment and on ed.

I downloaded our initial image batch from our shared google photos album into:

```text
data/raw/initial_batch/
```

The repo tracks empty placeholder folders using `.gitkeep` files so everyone gets the expected data structure after cloning. The actual image files, generated split images, and generated metadata files are still ignored by Git because they are large/binary data.

To reproduce the local setup:

1. download the images from the shared Google Photos album - I also shared these with you guys;
2. put the JPEGs in `data/raw/initial_batch/`;
3. run `notebooks/01_data_collection_preprocessing.ipynb` from the top;
4. the notebook will regenerate:

```text
data/image2gps_dataset/train/
data/image2gps_dataset/validation/
```

with images and `metadata.csv` files inside each split.


## Notebook 1: Data Collection Preprocessing

Use:

```text
notebooks/01_data_collection_preprocessing.ipynb
```

This notebook is pretty similar to the reference post_process notebook. It is designed to load the raw images, extract the gps metadata and create the train and validation data folders. It:

1. reads JPEG images from `data/raw/initial_batch/`;
2. randomly shuffles the image list;
3. creates `train` and `validation` folders under `data/image2gps_dataset/`;
4. uses `exifread` to extract GPS EXIF tags;
5. writes one `metadata.csv` file per split with:

```csv
file_name,Latitude,Longitude
```

For now, we are using an 85/15 train/validation split and no local test set. Since the dataset is still small, the professor leaderboard/test set will serve as the true held-out test. Also I plan to gather more data tomorrow, and the dataloader pipeline uses transformations to augment the dataset with too.

## Notebook 2: Baseline Data Loaders

Use:

```text
notebooks/02_baseline_data_loaders.ipynb
```

This notebook mirrors the data-loading section of the provided baseline model notebook. It:

1. loads our local split folders with Hugging Face `load_dataset("imagefolder", ...)`;
2. defines a `GPSImageDataset` class in the same style as the baseline;
3. creates PyTorch transforms;
4. creates `train_dataloader` and `val_dataloader`;
5. includes batch shape checks and image visualization.

This notebook stops before any actual model training code, but the dataloaders should be pretty much good to go for modeling work. Feel free to adjust them however you think is best though too.
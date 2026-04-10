# 2026 3rd Infodengue-Mosqlimate Dengue Challenge (IMDC)

## Team and Contributors

Epidemáticos - School of Applied Mathematics - Getulio Vargas Foundation (FGV/EMAp)

- [Eduardo Adame, M.Sc.](https://github.com/adamesalles)
- [Ezequiel Braga, M.Sc.](https://github.com/ezequielebs)
- [Isaque Pim, Ph.D.](https://github.com/isaquepim)

## Summary




## Repository Structure:

<!-- - `data/`
  1. `1_raw/` (raw data)
  2. `2_inter/` (intermediate data processed)
  3. `3_primary/` (primary data to be used to train the model)
  4. `4_model_output/` (predictions from the model without post-processing)
  5. `5_predictions/` (predictions to be submitted to the competition)
- `dataprep/` (notebooks to process the data)
  1. `1_enso_interpolation.ipynb`
  2. `2_geographic_uf.ipynb`
  3. `3_join_aggregate_data.ipynb`
- `train_model/` (notebooks to train the model)
  1. `train.ipynb`
  2. `SprintModels/` (trained models)
- `forecast_evaluation/` (notebooks to evaluate the model)
  1. `evaluate.ipynb`
- `post_processing/` (notebooks to post-process the predictions)
  1. `post_processing.ipynb`
  2. `submission.ipynb`
- `run_pipeline.py` (script to run the notebooks in the correct order to reproduce the results)
- `pyproject.toml` (project dependencies)
- `uv.lock` (dependency lock file) -->

## Libraries and dependencies

<!-- This project uses `uv` for dependency management. The main dependencies are defined in `pyproject.toml`, yet,
when it comes to training the model, the main library was `autogluon`, while `polars` and `pandas` were used for data manipulation. -->

## Data and Variables

<!-- The project uses the following datasets:
- **`dengue.csv.gz`**: Dengue cases by municipalities and week
- **`ocean_climate_oscillations.csv.gz`**: Climate indices including ENSO (El Niño-Southern Oscillation)
- **`geodata_uf.geojson`**: Shapefiles for Brazilian states -->

### Processed Variables:
<!-- The ENSO data was interpolated to obtain values for the same weeks as the dengue cases.
The shapefile was used to obtain latitude and longitude coordinates to represent each state.
We aggregated the dengue cases by state and week, and joined the ENSO data, while the 
longitude and latitude were used as static features.
We also added the log of the dengue cases to the model as a covariate.

The variables were selected by evaluating model performance on the validation set
for different combinations of variables. The best model was the one that used the ENSO index as a covariate. -->


## Model Training

<!-- The model used was Chronos, a probabilistic time-series forecasting model from Amazon, more specifically,
the `bolt_small` variant. We used the `autogluon` library,
which provides a high-level interface for training and evaluating time-series models.
The models were evaluated using the WQL (Weighted Quantile Loss) metric.

The training used two Chronos configurations with different fine-tuning strategies:
1. **Fine-tuned model**: `fine_tune=True` - model weights are updated during training
2. **Zero-shot model**: `fine_tune=False` - model uses pre-trained weights without fine-tuning

While developing the model, we have tested different configurations, including different models.

The training code is available in the `train_model/train.ipynb` file, and the post-processing
code is available in the `post_processing/post_processing.ipynb` file. -->

## Post-processing
<!-- After training, `post_processing.ipynb` is used to post-process the predictions to
align them with the competition requirements. -->

### Data Usage Restrictions

<!-- The competition required using only data up to Epidemiological Week (EW) 25 of the current year to
generate predictions from EW 41 of the same year to EW 40 of the next year.
To deal with this, our model predicts the entire time-series starting from EW 26 until EW 40 of the next year,
and then we post-process the data to cover only the required period. -->

### Predictive Uncertainty
<!-- The Chronos model within `autogluon` is already trained to provide quantile predictions,
yet, the model is tuned for quantiles `[0.1,0.2,0.5,0.8,0.9]`,
which did not match the requirements of the competition. Hence, we used interpolation and 
extrapolation to obtain the quantiles `[0.05,0.1,0.2,0.5,0.8,0.9,0.95]`, in order to
produce the required interval predictions.

During post-processing, we sorted quantile predictions to ensure monotonicity,
and we set predictions to zero when the value was negative. -->

## References

<!-- @article{ansari2024chronos,
  title={Chronos: Learning the language of time series},
  author={Ansari, Abdul Fatir and Stella, Lorenzo and Turkmen, Caner and Zhang, Xiyuan and Mercado, Pedro and Shen, Huibin and Shchur, Oleksandr and Rangapuram, Syama Sundar and Arango, Sebastian Pineda and Kapoor, Shubham and others},
  journal={arXiv preprint arXiv:2403.07815},
  year={2024}
}

@inproceedings{shchur2023autogluon,
  title={AutoGluon--TimeSeries: AutoML for probabilistic time series forecasting},
  author={Shchur, Oleksandr and Turkmen, Ali Caner and Erickson, Nick and Shen, Huibin and Shirkov, Alexander and Hu, Tony and Wang, Bernie},
  booktitle={International Conference on Automated Machine Learning},
  pages={9--1},
  year={2023},
  organization={PMLR}
} -->

Follow the [R demo rmd](/Demo%20Notebooks/R%20demo.Rmd) or [Python demo notebook](/Demo%20Notebooks/Python%20demo.ipynb) to learn of the essential steps you must follow to complete a submission of your work. For more details check the [mosqlient documentation](https://mosqlimate-client.readthedocs.io/en/latest/tutorials/API/registry/).  Video tutorials for prediction submission are available here: one for [R](https://www.youtube.com/watch?v=57hM-dVY4hA&list=PLh4FLfhFN5irN_IoZvy4c3cf4ZWrSWrwF&index=3) and another for [Python](https://www.youtube.com/watch?v=YorYQ6phAfw&list=PLh4FLfhFN5irN_IoZvy4c3cf4ZWrSWrwF&index=2). If you run into dificulties, please reach out fo help at our [discord server](https://discord.gg/yqtgW4TC). 

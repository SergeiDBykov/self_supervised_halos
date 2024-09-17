# Self-supervised contrastive learning for cosmological simulations

An ongoing project with the application of self-supervised learning on (3D,2D) image and time series modalities and their alignment from cosmological simulation data.

work in progress...


Stucture:

- `data`- here you will find the halo catalogue and merger histories. The original TNG halo snapshots are found on freya or elsewhere.
- `freya_runs` - sample scripts to run `Pytorch` code on freya GPUs.
- `notebooks` - mostly model training script to debug the loops, used for a couple of epochs to see that the loss goes down and so on. The filenames reflect which part of the model is trained and in which fashion.
- `results/plots` - preliminary plots of data and models.
- `scripts` - here scripts are stored.
  - `base_model.py` - base class for model training.
  - `classification_2d.py`, `classification_3d.py` - models to deal with 2d and 3d snapshot histograms.
  - `halo_mass_embeddings.py` - transformer encoder for mass accretion history.
  - `contrastive_learning_2d.py` - attempted model for contrastive learning on 2d projection histograms.
- `utils` - data preparation and TNG analysis.
- - `dataloader.py` - class for data loader in Pytorch and data transform/normalisation.
  - `tng.py` - functions for working with TNG data, producing density maps from dark matter particle positions.

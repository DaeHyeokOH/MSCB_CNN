# EEG Motor Imagery Classification

This repository contains code related to EEG-based motor imagery classification using a multi-scale convolutional neural network (MS-CNN).

## Original Article

The code in this repository is inspired by the following article:

**Title:** An efficient multi-scale CNN model with intrinsic feature integration for motor imagery EEG subject classification in brain-machine interfaces
**Author:** Arunabha M. Roy
**Journal:** Biomedical Signal Processing and Control
**DOI:** [10.1016/j.bspc.2022.103496](https://doi.org/10.1016/j.bspc.2022.103496)

Please note that while the methodology is influenced by the above article, the codes in this repository are not authored by Arunabha M. Roy.

## Contents

1. [dat_to_image.py](dat_to_image.py): Python script to convert raw data to spectrogram.
2. [dat_to_statistic_feature.py](dat_to_statistic_feature.py): Python script to convert raw data to statistical features such as differential entropy and neural power spectra, and save them in a .pkl file.
3. [MSCB_CNN_Aro.ipynb](MSCB_CNN_Aro.ipynb): Jupyter Notebook containing the implementation of the multi-scale CNN based on the original article.
4. [reboot.py](reboot.py): Script to address the increased computing time issue in `dat_to_image.py`. Automatically executes `dat_to_image.py`.
5. [statistic_feature_error_check.py](statistic_feature_error_check.py): Python script to open a .pkl file generated by `dat_to_statistic_feature.py` and check for NaN or INF values.

## Usage

1. Run `dat_to_image.py` to convert raw data to spectrogram.
2. Run `dat_to_statistic_feature.py` to convert raw data to statistical features.
3. Open [MSCB_CNN_Aro.ipynb](MSCB_CNN_Aro.ipynb) for the implementation of the MS-CNN model.
4. Execute `reboot.py` to address computing time issues in `dat_to_image.py`.
5. Use `statistic_feature_error_check.py` to check for NaN or INF values in the generated statistical feature file.

## Acknowledgment

The methodology and inspiration for this work come from the research conducted by Arunabha M. Roy. However, please note that the specific code implementations in this repository are not authored by Arunabha M. Roy.

## Citation

If you find this work helpful, please consider citing the original article:

Roy, A. M. (2022). An efficient multi-scale CNN model with intrinsic feature integration for motor imagery EEG subject classification in brain-machine interfaces. *Biomedical Signal Processing and Control*, 74, 103496. [DOI](https://doi.org/10.1016/j.bspc.2022.103496).

And my repository Either!!!

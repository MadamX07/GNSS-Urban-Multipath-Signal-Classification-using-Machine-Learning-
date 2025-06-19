# GNSS Urban Multipath Classification using Machine Learning

This project uses machine learning to classify **Line-of-Sight (LOS)** and **Non-Line-of-Sight (NLOS)** satellite signals in urban environments. The goal is to improve GNSS positioning accuracy by identifying and filtering out unreliable signals affected by multipath and obstructions.

## Overview

Urban environments cause GNSS signal degradation due to building reflections and obstructions. This project uses real GNSS datasets and supervised ML models to classify signal quality and filter NLOS signals.

## Features Extracted
From raw RINEX/observation data, the following features were extracted:
- Signal-to-Noise Ratio (SNR)
- Azimuth angle
- Satellite elevation

## Tech Stack

- **Languages:** Python
- **Libraries:** `scikit-learn`, `pandas`, `NumPy`, `matplotlib`
- **Models:** Random Forest, Support Vector Machine (SVM)

## Results

- Dataset: 60,000+ labeled urban signal samples
- Accuracy: **98.66%**
- Outcome: Successfully filtered NLOS signals to enhance GNSS accuracy on smart devices in urban areas.



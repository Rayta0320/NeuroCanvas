# NeuroCanvas

NeuroCanvas is a real-time EEG-driven generative art system that transforms brainwave activity into visual expression.

The system receives live EEG data from a Muse headset, extracts signal features related to frequency activity and hemispheric asymmetry, estimates emotional state as valence and arousal, and uses those values to control generative visuals in TouchDesigner.

Rather than treating EEG as a direct emotion-reading tool, NeuroCanvas uses brainwave patterns as an expressive input for responsive digital artwork.

![NeuroCanvas visual output](Images/Visual.png)

## Overview

NeuroCanvas connects neuroscience, machine learning, and generative visual design.

```text
Muse EEG Headset
-> Mind Monitor / OSC
-> EEG Feature Extraction
-> Valence-Arousal Prediction Model
-> TouchDesigner
-> Real-Time Generative Visuals
```

The project uses 4-channel EEG data from Muse:

```text
TP9, AF7, AF8, TP10
```

From this signal, the system extracts a 61-dimensional feature vector from each EEG window and predicts two affective values:

- Valence: pleasant / unpleasant emotional direction
- Arousal: calm / activated emotional intensity

These values are then mapped to visual parameters such as color, motion, speed, distortion, and form.

## Features

- Real-time EEG input through OSC
- Muse 4-channel EEG support
- EEG feature extraction from frequency bands
- Valence-arousal prediction using a LightGBM model
- TouchDesigner integration
- Optional user calibration
- Real-time visual control from predicted emotional state

## EEG Features

NeuroCanvas extracts features from standard EEG frequency bands:

| Band | Frequency |
| --- | --- |
| Delta | 1-4 Hz |
| Theta | 4-8 Hz |
| Alpha | 8-13 Hz |
| Beta | 13-30 Hz |
| Gamma | 30-45 Hz |

The feature vector includes:

- Differential Entropy
- Frontal Alpha Asymmetry
- DASM / RASM hemispheric asymmetry features
- Beta/Alpha and Theta/Beta ratios
- Hjorth activity, mobility, and complexity

Each EEG window is converted into a 61-dimensional feature vector.

## Machine Learning

The prediction model is trained on NeuroSense EEG data.

The current real-time model uses:

- 16-second EEG windows
- 2-second window step
- 61 EEG features
- Per-subject normalization during training
- LightGBM regression for valence and arousal prediction

The trained model is saved as:

```text
va_prediction_model/preprocessed_data_neurosense/va_window_model.joblib
```

This model is loaded in TouchDesigner for real-time inference.

## TouchDesigner Integration

TouchDesigner receives the predicted valence and arousal values as CHOP channels:

```text
valence
arousal
```

These channels can be connected to visual parameters to control the generative system in real time.

Example mappings:

| Signal | Visual Mapping |
| --- | --- |
| Valence | Color palette, brightness, harmony |
| Arousal | Motion speed, intensity, distortion |
| Low arousal | Slower, softer visuals |
| High arousal | Faster, more energetic visuals |
| High valence | Warmer or more expansive visuals |
| Low valence | Cooler or more compressed visuals |

The TouchDesigner project files are stored in the project root and backup folder.

## Project Structure

```text
VA2Art/
  EEG2Emotion_Visual.toe
  Images/
  requirements.txt
  va_prediction_model/
    neurosense.py
    train_window.py
    osc_server.py
    touchdesigner/
```

Important files:

- `va_prediction_model/neurosense.py`: NeuroSense data loading and EEG feature extraction.
- `va_prediction_model/train_window.py`: trains and saves the real-time valence-arousal model.
- `va_prediction_model/touchdesigner/va_infer_ext.py`: TouchDesigner inference extension.
- `va_prediction_model/touchdesigner/va_features.py`: 61-dimensional EEG feature extraction for runtime inference.
- `va_prediction_model/touchdesigner/va_infer_scriptCHOP_callback.py`: Script CHOP callback that outputs valence and arousal.

## Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Train the NeuroSense window model:

```bash
python va_prediction_model/train_window.py
```

Run OSC-related scripts only after confirming the Muse headset and Mind Monitor are sending data to the correct IP address and port.

## Project Goal

NeuroCanvas explores how physiological signals can become part of an artistic feedback system.

The goal is not to perfectly classify emotions, but to create a meaningful interaction between the nervous system and digital visual expression.

It treats EEG as a living input signal: noisy, personal, unstable, and expressive.

## Tech Stack

- Python
- NumPy
- SciPy
- scikit-learn
- LightGBM
- python-osc
- Muse / Mind Monitor
- TouchDesigner

## Status

This project is currently a working prototype for real-time EEG-to-visual interaction.

The system supports EEG signal monitoring, feature extraction, model training, real-time inference, and TouchDesigner visual output.

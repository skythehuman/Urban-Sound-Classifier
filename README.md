# Urban Sound Classifier

Browser-based urban noise classification using a 4-fold ensemble of ResNet models. Upload a WAV file and get instant classification across 17 urban noise types.

**[Go to Demo](https://urban-sound-classifier.streamlit.app)**

## Features

- **Audio upload** with waveform playback
- **Feature visualization** : Mel-spectrogram and MFCC
- **Ensemble inference** : 8 models (4 Mel + 4 MFCC) averaged
- **Top-k predictions** with confidence bar chart
- **Noise taxonomy** : 5 categories, 17 subcategories
- **Model architecture** and performance summary

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model Details

| Spec | Value |
|---|---|
| Architecture | ResNet-based DCNN |
| Features | Mel-spectrogram (n=40) + MFCC (n=40) |
| Ensemble | 4-fold CV × 2 features = 8 models |
| Accuracy | **96.2%** (ensemble) |
| Parameters | 84,755 per model (~1.3 MB) |
| Sample Rate | 20,000 Hz |
| Input Shape | (40, 2344, 1) |
| Output | 17 classes (softmax) |

## Environmental Noise Classes

| Category | Classes |
|---|---|
| Transportation | Airplane, Helicopter, Train, Car Driving, Car Horn, Motorcycle, Motorcycle Horn, Engine Idling, Siren |
| Industrial | Blast Fan, Compressor, Rack Cutting Machine | 
| Construction | Pile Driver, Concrete Pump, Jackhammer |
| Music | Street Music |
| Animal | Dog Barking |

## References

- Lee, H. (2026). *Noise Source Classification and Environmental Pattern Analysis: An Integrated Approach using Deep Learning and K-means Clustering*. In Ph.D. Dissertation.
- Salamon, J., Jacoby, C., & Bello, J. P. (2014). UrbanSound8K (1.0.0) [Data set]. *Zenodo*. https://doi.org/10.5281/zenodo.1203745
- Piczak, K. J. (2015, October). ESC: Dataset for environmental sound classification. In *Proceedings of the 23rd ACM international conference on Multimedia* (pp. 1015-1018). http://dx.doi.org/10.1145/2733373.2806390

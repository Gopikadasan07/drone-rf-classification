# drone-rf-classification
RF-based drone type classification from noisy IQ signals using a spectrogram CNN (7 classes, 85.5% test accuracy, macro F1 0.83)
# Drone RF Signal Classification

Classifies the type of drone from noisy radio (IQ) signals using a spectrogram CNN.

## Dataset
Noisy Drone RF Signal Classification v2 (Kaggle, by sgluege): 17,744 IQ recordings, 7 drone classes, SNR from -20 to +30 dB.
Split: 12,420 train / 2,662 validation / 2,662 test.

## Method
Each IQ recording is converted to a 128x128 spectrogram. A CNN with four convolution blocks (Conv, BatchNorm, ReLU, MaxPool), global average pooling and a 7-way linear layer is trained with Adam and class-weighted cross-entropy. The best epoch is chosen by validation loss.

## Results (test set)
| Model | Accuracy | Macro recall | Macro F1 |
|---|---|---|---|
| Unweighted | 0.827 | 0.702 | 0.771 |
| Inverse-frequency weights | 0.792 | 0.796 | 0.760 |
| Square-root weights (final) | 0.855 | 0.805 | 0.830 |
| Square-root weights + augmentation | 0.860 | 0.802 | 0.836 |

Accuracy is 90 to 98% at 0 dB SNR and above, and 44 to 65% between -20 and -12 dB, where the spectrogram shows mostly noise.

## Limitations
- Four models were compared on the same test set, so the numbers are slightly optimistic.
- The model classifies drone types only. It cannot tell a drone from a bird or from no drone.
- Trained on one public dataset, not tested on real hardware.

## Future work
- Raw-IQ model using the highest-energy window of each recording (not run yet)
- Higher-resolution spectrograms
- Cross-validation with a fixed random seed
- Multi-sensor fusion (radar, acoustic, camera)

## Files
- notebook (`.ipynb`): full code
- `best_model_soft.pth`: final model weights
- `test_results_soft.csv`: per-sample test predictions

# Sign Language Recognition Model

This repository focuses on sign language recognition using deep learning models and practical training workflows. The primary notebook `CNN_Training.ipynb` contains an optimized Colab pipeline for training, evaluating, and saving CNN-based models on a processed image dataset of Malaysian sign language gestures.

## Overview

The Colab notebook compares three model approaches and includes memory/GPU optimizations to run reliably on Google Colab (T4 GPU):

- Custom CNN (from-scratch)
- MobileNetV2 (transfer learning) — selected as the best-performing model in the notebook
- ResNet50 (transfer learning)

## Repository structure

- CNN_Training.ipynb - Colab notebook with dataset extraction, model building, training, evaluation, and model saving.
- README.md - This file.
- requirements.txt (optional) - Python package dependencies.

## Dataset (expected by the notebook)

Place a zipped, processed dataset in your Google Drive at:

- `/content/drive/MyDrive/processed_dataset.zip`

When the notebook runs it extracts the zip to:

- `/content/dataset/processed_dataset`

The dataset should be organized into subfolders (one per class). During development the notebook used 25 classes and 22,500 images (18,000 training / 4,500 validation). Example class names observed in the notebook:

1, 2, 3, 4, 5, 6, 7, 8, 9, A, AWAK, B, C, D, E, F, G, H, I, MAAF, MAKAN, MINUM, SALAH, SAYA, TOLONG

Adjust the paths or dataset layout as needed for your environment.

## Key training settings (from the notebook)

- Batch size: 16
- Image size: 224 x 224
- Validation split: 20% (image_dataset_from_directory with validation_split)
- Seed for splits: 123
- Mixed precision: Enabled via `tf.keras.mixed_precision.set_global_policy('mixed_float16')` to reduce GPU memory usage on compatible GPUs (e.g. T4)

## Colab & memory optimizations included in the notebook

The notebook includes multiple practical optimizations to prevent OOM crashes on Colab:

- Enable mixed precision for faster training and lower GPU memory usage.
- Use a conservative batch size (16). Lower further (e.g., 8) if you get OOMs.
- Avoid `cache()` to reduce system RAM usage; use `prefetch(buffer_size=tf.data.AUTOTUNE)` instead.
- Clear Keras backend and run garbage collection between model runs:
  - `tf.keras.backend.clear_session()`
  - `gc.collect()`
- When using mixed precision, set the final Dense layer dtype to `float32` for numerical stability: `layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')`.

## Models implemented in the notebook

1) Custom CNN
- Sequential Conv2D / MaxPooling / Flatten / Dense architecture.
- Compiled with Adam optimizer and `sparse_categorical_crossentropy` loss.

2) MobileNetV2 (Transfer Learning)
- Pretrained MobileNetV2 as a frozen base (weights='imagenet', include_top=False).
- Preprocessing used: `Rescaling(1./127.5, offset=-1)` to match MobileNetV2 expected input range.
- Added GlobalAveragePooling2D, Dense(256, relu), Dropout(0.5), final Dense(NUM_CLASSES, softmax, dtype='float32').
- Saved as the best model in the notebook.

3) ResNet50 (Transfer Learning)
- Pretrained ResNet50 as a frozen base (weights='imagenet', include_top=False).
- Preprocessing: `Rescaling(1./255)`.

Note: In the notebook runs, MobileNetV2 achieved very high validation accuracy while ResNet50 underfit with the same default training settings — differences in preprocessing, learning rate, or fine-tuning strategy can explain this.

## Evaluation

The notebook prints classification reports (precision, recall, f1-score) for each model on the validation set. It also plots training/validation accuracy and loss for fit diagnostics and computes a confusion matrix visualized with seaborn to inspect common class confusions.

## Saved artifacts (from the notebook)

- Best model (MobileNetV2) saved to Google Drive at: `/content/drive/MyDrive/best_model.keras`
- Class names JSON saved to: `/content/drive/MyDrive/class_names.json`

To load the saved model elsewhere:

```python
import tensorflow as tf
import json

model = tf.keras.models.load_model('/path/to/best_model.keras')
with open('/path/to/class_names.json') as f:
    class_names = json.load(f)
```

## How to run the Colab notebook

1. Open `CNN_Training.ipynb` in Google Colab (the notebook includes a Colab badge link at the top).
2. Mount Google Drive and upload `processed_dataset.zip` to your Drive (e.g., `/content/drive/MyDrive/`).
3. Run cells in order. The notebook extracts the dataset, builds and trains three models, evaluates them, and saves the chosen best model to Drive.

Notes:
- Pretrained model weights for MobileNetV2/ResNet50 are downloaded from the internet — ensure connectivity in the Colab environment.
- If you encounter GPU OOM errors, try reducing BATCH_SIZE and/or disabling mixed precision.

## Dependencies

The notebook uses common ML libraries; install with:

```bash
pip install tensorflow scikit-learn matplotlib seaborn
```

(You can also provide a `requirements.txt` with pinned versions for reproducibility.)

## Reproducibility

- The notebook sets a fixed seed (123) for the `image_dataset_from_directory` calls to ensure consistent train/validation splits.

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.

## License

This project is provided for research and educational purposes. See the LICENSE file for details.

## Contact

If you have questions about the code or models, open an issue or contact the repository owner.

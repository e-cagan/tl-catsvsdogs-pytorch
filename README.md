# tl-catsvdogs-pytorch

A PyTorch transfer-learning classifier for the classic "Cats vs Dogs" task. This repository contains training and evaluation code (classifier.py) that fine-tunes a convolutional model and reports training loss and validation accuracy.

## Features
- Transfer learning with a pretrained CNN (configurable).
- Training loop with per-epoch validation.
- Simple CLI usage: run training / evaluation with a single script.
- Small, easy-to-read codebase intended for learning and experimentation.

## Repository structure
- classifier.py         - Main training / evaluation script
- requirements.txt      - Python dependencies (create if missing)
- README.md             - This file
- data/                 - (expected) dataset root (train / val / test folders)
- models/               - directory for saving checkpoints (created at runtime)

Adjust paths and filenames if your project uses a different layout.

## Requirements
- Python 3.8+
- PyTorch / torchvision compatible with your CUDA (or CPU) setup
- Pillow
- numpy

Example (create virtualenv and install):
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
If you don't have a requirements.txt, install minimal deps:
```
pip install torch torchvision pillow numpy tqdm
```

## Quick start

1. Prepare dataset
    - Arrange images under `data/train/<class>/*` and `data/val/<class>/*` (e.g., `cats`, `dogs`).
    - Typical structure:
      - data/train/cats/*.jpg
      - data/train/dogs/*.jpg
      - data/val/cats/*.jpg
      - data/val/dogs/*.jpg

2. Run training
```
source venv/bin/activate
python3 classifier.py
```

3. Checkpoints / outputs
- Model checkpoints (if enabled in the script) are saved to `models/`.
- Training prints per-epoch loss and validation accuracy to console.

## Example training log (from a run)
```
Epoch: 0, Loss: 0.13622072049247
Validation Accuracy: 0.9199679871948779

Epoch: 1, Loss: 0.09612227126676291
Validation Accuracy: 0.9407763105242097

Epoch: 2, Loss: 0.0795709973903367
Validation Accuracy: 0.9443777511004402

Epoch: 3, Loss: 0.06052084976045516
Validation Accuracy: 0.9613845538215287

Epoch: 4, Loss: 0.06118398645636849
Validation Accuracy: 0.9597839135654261

Epoch: 5, Loss: 0.049987467444730926
Validation Accuracy: 0.9601840736294518

Epoch: 6, Loss: 0.04139840891848827
Validation Accuracy: 0.9615846338535414

Epoch: 7, Loss: 0.03385254905779307
Validation Accuracy: 0.9653861544617847

Epoch: 8, Loss: 0.03251782841586207
Validation Accuracy: 0.9623849539815926

Epoch: 9, Loss: 0.026175950955884635
Validation Accuracy: 0.9571828731492596
```
This shows rapid convergence and high validation accuracy on the dataset used for these runs.

## Troubleshooting
- Pillow warning: `UserWarning: Truncated File Read` often appears when an image file is partially corrupted or truncated. To handle or suppress:
  - Inspect or remove problematic files.
  - Allow loading truncated images (may hide issues):
     ```python
     from PIL import ImageFile
     ImageFile.LOAD_TRUNCATED_IMAGES = True
     ```
- If CUDA is not available, ensure the script falls back to CPU, or set `device = torch.device('cpu')`.
- If you run out of GPU memory, reduce batch size or image resolution.

## Recommendations and tips
- Use torchvision transforms including RandomResizedCrop, RandomHorizontalFlip for augmentation.
- Freeze pretrained backbone layers initially, train classifier head, then fine-tune with a lower learning rate.
- Track experiments with TensorBoard, Weights & Biases, or simple CSV logs.
- Validate on a hold-out set and consider cross-validation for robust estimates.

## Contributing
Contributions are welcome. Open an issue to discuss changes, or submit a pull request with a clear description and tests/examples.

## License
MIT License — see LICENSE file or add one if needed.
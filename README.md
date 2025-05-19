# Plant Disease Classifier

A lightweight implementation of a Vision Transformer (ViT) combined with pretrained VGG-16 using TensorFlow and keras to classify diseases in apple leaves. Loading and preprocessing raw image and label data occurs from binary files.
A compact ViT model implemented using Keras that works by splitting each image into patches, applying transformer encoders, and attaching a lightweight MLP head.
Centralising all paths and hyperparameters in a config.py for easy tweaks.

## File Structure


## Model Performance

After training for 50 epochs on a dataset of ~2,000 apple‑leaf images (split 80/20 train/val), the Vision Transformer achieved:

- **Training accuracy:** 95.2%  
- **Validation accuracy during Training:** 92.3%  
- **After Training:** 96.9%  

### Detailed Metrics

| Class                | Precision | Recall | F1‑score |
|----------------------|----------:|-------:|---------:|
| Apple Scab           |     0.93  |   0.90 |    0.92  |
| Cedar Apple Rust     |     0.94  |   0.90 |    0.92  |
| Frogeye Spot         |     0.96  |   0.94 |    0.95  |
| Healthy              |     0.85  |   0.93 |    0.89  |


### Notes

- val loss plateaued around epoch 30
- Overall F1‑score of **0.92** indicates balanced performance across categories.
- Future work: export and quantize it to an integer‐only TensorFlow Lite file.

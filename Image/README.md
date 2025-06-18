# Image-Based Detection

1. Weather Detection

Classifies images into weather categories: cloudy, shine, rain, sunrise.

Uses pretrained CNN embeddings via **img2vec** and a **Random Forest classifier**.

```bash
python weather_detection.py
```

2. Card Classifier

Classifies Deck of Cards using **Pytorch**

Uses the **timm library** to build an EfficientNet-B0 model and includes a validation loop to monitor performance and reduce overfitting.


3. CIFAR10

Classifies CIFAR10 Dataset using a **CNN classifier model**

Added some explaining about CNN (output size), Normalization and Optimization (SGD and Momentum).

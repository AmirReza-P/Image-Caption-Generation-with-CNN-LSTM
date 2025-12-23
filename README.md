---

# Image Caption Generator with CNN-LSTM

A deep learning model that automatically generates descriptive captions for images. This project implements an encoder-decoder architecture, utilizing a pre-trained **Xception** CNN for visual feature extraction and a **Bidirectional LSTM** network for text sequence generation.

## 📌 Overview

Image captioning is a challenging task that intersects Computer Vision and Natural Language Processing (NLP). This model learns to recognize objects and their relationships within an image and generates a syntactically correct description.

The system was trained and evaluated on the **Flickr8k dataset** using standard metrics like **BLEU** and **CIDEr** to benchmark performance.

## 🚀 Features

* **Visual Encoder:** Uses **Xception** (pre-trained on ImageNet) to extract high-level feature vectors from images.
* **Text Decoder:** Implements a **Bidirectional LSTM** to handle sequence generation, allowing the model to capture context from both past and future states during training.
* **Embeddings:** Utilizes **GloVe (Global Vectors for Word Representation)** 200-dimensional embeddings for dense word representation.
* **Optimization:** Trained using the **AdamW** optimizer with a learning rate of `1e-4` for better regularization.

## 📂 Dataset

The project uses the **Flickr8k dataset**, which contains:

* 8,091 images.
* 5 captions per image (totaling ~40,000 captions).

*Note: The dataset paths in the notebook are configured for the Kaggle environment (`/kaggle/input/flickr8k-dataset/`). If running locally, please update the paths in the configuration section.*

## 🏗️ Model Architecture

The architecture follows a standard "Merge" model approach:

1. **Image Feature Extractor:**
* Input: Images resized to `299x299`.
* Model: Xception (top layer removed, average pooling enabled).
* Output: 2048-dimensional feature vector, projected to a 512-unit dense layer.


2. **Sequence Processor:**
* Input: Text sequences padded to a maximum length of **34**.
* Embedding Layer: Pre-trained GloVe vectors (200 dim).
* LSTM Layer: Bidirectional LSTM with 512 units and dropout (0.3).


3. **Decoder:**
* The outputs from the CNN and LSTM are merged using an `Add` layer.
* Passed through Dense layers with Batch Normalization and ReLU activation.
* Final Output: Softmax probability distribution over the vocabulary size.



## 🛠️ Installation & Requirements

1. Clone the repository:
```bash
git clone https://github.com/yourusername/image-caption-generator.git
cd image-caption-generator

```


2. Install the required dependencies:
```bash
pip install numpy pandas tensorflow matplotlib nltk opencv-python pycocoevalcap

```



## 💻 Usage

### 1. Preprocessing

The notebook handles the preprocessing of images and text:

* Images are normalized using Xception's `preprocess_input`.
* Captions are cleaned (punctuation removed, lowercased) and tokenized.
* `<startseq>` and `<endseq>` tokens are added to guide the generation process.

### 2. Training

Run the training cells in the notebook. The model uses `categorical_crossentropy` loss and trains for **100 epochs** with a batch size of **192**.

```python
# Sample Training Call
model.fit([X1, X2], y, epochs=100, batch_size=192)

```

### 3. Inference / Evaluation

To generate a caption for a new image, the system uses a **Greedy Search** algorithm:

```python
caption = greedy_search(image_feature)

```

This iteratively predicts the most likely next word until the `<endseq>` token matches or the max length is reached.

## 📊 Evaluation

The model is evaluated using the following metrics:

* **BLEU-1, BLEU-2, BLEU-3, BLEU-4:** Measures n-gram overlap between generated and reference captions.
* **METEOR:** Aligns generated captions with references.
* **CIDEr:** Captures human consensus.


## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements.


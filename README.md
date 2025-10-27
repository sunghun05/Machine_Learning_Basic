# Machine_Learning_Basic

This repository includes basic machine learning examples such as classification, linear regression, SVM, and so on.
All of code samples are from Machine Learning Lecture in Sooncheonhyang University, and lecture CS229

---

## 🚀 Getting Started

This project uses Python 3.x and relies on a virtual environment (`venv`) for dependency management.

### 1. Clone the Repository

```bash
git clone [Your Git Repository URL]
cd [Your-Project-Folder-Name]
```

### 2. Create and Activate the Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the environment
# On macOS / Linux
source venv/bin/activate

# On Windows (Command Prompt / PowerShell)
.\venv\Scripts\activate
```

### 3. Install Dependencies

Install all required packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

---

## ⚠️ A Note on PyTorch Installation

This project uses `pytorch`. The `requirements.txt` file might install a **CPU-only** version of PyTorch by default.

If you require **GPU (CUDA) support**, it is strongly recommended that you install the correct PyTorch version manually *before* or *after* running `pip install -r requirements.txt`.

Please visit the [Official PyTorch Website](https://pytorch.org/get-started/locally/) to find the correct command for your specific environment (OS, Package Manager, CUDA version).

*Example command for CUDA 11.8:*
```bash
# You can run this before installing the other requirements
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
```

---

## 💻 Usage

The modules will make sample dataset automatically.

```bash
# Example: Run the classification model
python -m classification.classification
```

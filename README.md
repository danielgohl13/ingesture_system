# InGesture: Gesture Recognition System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning-based system for recognizing hand gestures using wrist-worn IMU data. This project utilizes the InGesture dataset for training and evaluating gesture recognition models. The system is designed to classify various hand gestures including fluid intake, phone answering, and other common hand-to-head movements.

## 📋 Overview

This system processes accelerometer data to detect and classify eating and drinking gestures. It implements various deep learning architectures for time-series classification and includes tools for data preprocessing, model training, and evaluation.

## 🎯 Dataset: InGesture

The system is built on the [InGesture Dataset](https://data.mendeley.com/datasets/fdxst56tcj/3), which includes:

- **65 sessions** from multiple participants
- **8 distinct gesture classes**
- **200Hz sampling rate** from WT901BLECL5 IMU
- **Dual data formats**:
  - Segmented gesture files (individual gestures)
  - Continuous session recordings (~10 minutes each)

### Gesture Classes
0. Free activity
1. Fluid intake
2. Answering phone
3. Scratching head
4. Passing hand over face
5. Adjusting glasses/touching temples
6. Holding chin
7. Stretching hands behind neck


```

## 🚀 Features

- Multiple deep learning architectures (CNN, CNN-LSTM)
- Signal processing pipeline for IMU data (200Hz)
- Support for both segmented and continuous data formats
- Feature extraction methods including spectrograms and time-series transformations
- Model training and evaluation tools
- Confusion matrix analysis

## 📚 Citation

If you use this dataset or code in your research, please cite:

```
Gohl, Pedro Daniel; Spellen, Amanda Nicole; Queiroz, Laura Isabelle; Souto, Eduardo James (2025), 
"InGesture Dataset", Mendeley Data, V3, doi: 10.17632/fdxst56tcj.3
```

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/danielgohl13/ingesture_system.git
   cd ingesture_system
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Usage

1. Configure your dataset paths in `src/config.py`
2. Run the training script:
   ```bash
   python src/train.py
   ```
3. Evaluate the models:
   ```bash
   python src/evaluate.py
   ```

## 📊 Project Structure

```
├── src/                     # Source code
│   ├── architectures/      # Model architectures
│   ├── config.py           # Configuration settings
│   ├── datasets.py         # Data loading and preprocessing
│   ├── evaluate.py         # Model evaluation
│   ├── metrics.py          # Custom metrics
│   ├── model_trainer.py    # Training utilities
│   ├── train.py            # Main training script
│   └── utils.py            # Utility functions
├── experiments/            # Experiment results 
├── .gitignore
└── README.md
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

Pedro Daniel Gohl - [pedro.gohl@icomp.ufa.edu.br](mailto:pedro.gohl@icomp.ufa.edu.br)

Project Link: [https://github.com/danielgohl13/ingesture_system](https://github.com/danielgohl13/ingesture_system)

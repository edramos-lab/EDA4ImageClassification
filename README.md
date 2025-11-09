# EDA4ImageClassification

A web-based tool for analyzing image datasets, providing insights into class distribution, image dimensions, RGB histograms, and more.

## Features

- Upload and analyze CSV files containing image paths and class labels
- Visualize class distribution and dataset metrics
- Analyze image dimensions and properties
- Generate RGB histograms and insights
- Perform cross-validation visualization (K-Fold and Stratified K-Fold)
- Dimensionality reduction visualization (PCA and T-SNE)

## Requirements

- Python 3.7+
- Flask
- NumPy
- Pandas
- Pillow
- scikit-learn
- Matplotlib
- Seaborn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/edramos-lab/EDA4ImageClassification.git
cd EDA4ImageClassification
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Prepare your CSV file with the following format:
```
path/to/image1.jpg,class1
path/to/image2.png,class2
path/to/image3.jpeg,class1
```

4. Upload your CSV file through the web interface

5. Explore the various visualizations and metrics:
   - Dataset metrics (class distribution, imbalance ratio, etc.)
   - Image analysis (dimensions, sizes, etc.)
   - RGB histogram analysis
   - Cross-validation visualization
   - Dimensionality reduction visualization

## CSV File Format

The CSV file should have at least two columns:
1. First column: Path to the image file (must be JPG/JPEG/PNG)
2. Second column: Class label

Example:
```
data/images/cat1.jpg,cat
data/images/dog1.png,dog
data/images/cat2.jpeg,cat
```

## Notes

- Only JPG, JPEG, and PNG image files are supported
- The application will ignore any invalid image paths
- Make sure the image paths in your CSV file are accessible to the application
- The dataset multiplier feature allows you to simulate larger datasets by duplicating the data

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
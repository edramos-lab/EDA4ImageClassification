import os
import io
import base64
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
from flask_cors import CORS
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import cv2
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
matplotlib.use('Agg')

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

def calculate_metrics(df):
    try:
        # Calculate class distribution
        class_counts = df['class'].value_counts()
        total_samples = len(df)
        class_distribution = (class_counts / total_samples * 100).round(2)
        
        # Calculate entropy
        class_probs = class_counts / total_samples
        entropy = -np.sum(class_probs * np.log2(class_probs))
        max_entropy = np.log2(len(class_counts))
        entropy_balance = int((entropy / max_entropy * 100) * 10000) / 10000
        
        # Calculate balancing efficiency
        num_classes = len(class_counts)
        ideal_count = total_samples / num_classes
        # BE = (1 - (1/C) * Σ|n_i - n̄|/n̄) × 100
        sum_abs_deviations = sum(abs(count - ideal_count) for count in class_counts)
        balancing_efficiency = int(((1 - (1 / num_classes) * (sum_abs_deviations / ideal_count)) * 100) * 10000) / 10000
        
        # Calculate class imbalance ratio
        max_class_count = class_counts.max()
        min_class_count = class_counts.min()
        imbalance_ratio = (max_class_count / min_class_count).round(2)
        
        # Calculate dataset size
        dataset_size = total_samples
        
        # Calculate number of classes
        num_classes = len(class_counts)
        
        # Calculate average samples per class
        avg_samples_per_class = (total_samples / num_classes).round(2)
        
        # Calculate standard deviation of class distribution
        std_dev = class_counts.std().round(2)
        
        # Calculate coefficient of variation
        cv = (std_dev / class_counts.mean() * 100).round(2)
        
        # Calculate Gini coefficient
        sorted_counts = np.sort(class_counts)
        n = len(sorted_counts)
        index = np.arange(1, n + 1)
        gini = ((2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n) * 100
        
        return {
            'class_distribution': class_distribution.to_dict(),
            'entropy_balance': entropy_balance,
            'balancing_efficiency': balancing_efficiency,
            'imbalance_ratio': imbalance_ratio,
            'dataset_size': dataset_size,
            'num_classes': num_classes,
            'avg_samples_per_class': avg_samples_per_class,
            'std_dev': std_dev,
            'cv': cv,
            'gini': gini
        }
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return None

def analyze_dataset(df, multiplier=1):
    # Calculate basic metrics
    class_counts = df['class'].value_counts()
    max_class = class_counts.index[0]
    min_class = class_counts.index[-1]
    max_samples = class_counts.iloc[0]
    min_samples = class_counts.iloc[-1]
    ir = max_samples / min_samples
    be = (1 - (max_samples - min_samples) / (max_samples + min_samples)) * 100
    eb = (1 - (class_counts.std() / class_counts.mean())) * 100

    # Calculate train/test/validation split from paths
    train_count = 0
    test_count = 0
    validation_count = 0
    other_count = 0
    
    for _, row in df.iterrows():
        path = row['path'].lower()
        if '/train/' in path:
            train_count += 1
        elif '/test/' in path:
            test_count += 1
        elif '/validation/' in path or '/val/' in path:
            validation_count += 1
        else:
            other_count += 1

    # Image analysis
    valid_images = 0
    invalid_images = 0
    invalid_paths = []
    total_width = 0
    total_height = 0
    total_size = 0
    min_width = float('inf')
    min_height = float('inf')
    max_width = 0
    max_height = 0
    image_records = []

    for _, row in df.iterrows():
        try:
            img = Image.open(row['path']).convert('RGB')
            width, height = img.size
            size = os.path.getsize(row['path']) / 1024  # Size in KB
            
            total_width += width
            total_height += height
            total_size += size
            min_width = min(min_width, width)
            min_height = min(min_height, height)
            max_width = max(max_width, width)
            max_height = max(max_height, height)
            
            image_records.append({
                'path': row['path'],
                'class': row['class'],
                'width': width,
                'height': height,
                'size_kb': round(size, 2)
            })
            valid_images += 1
        except Exception as e:
            invalid_images += 1
            invalid_paths.append(row['path'])

    metrics = {
        'max_samples': int(max_samples),
        'max_class': max_class,
        'min_samples': int(min_samples),
        'min_class': min_class,
        'ir': round(ir, 2),
        'be': round(be, 2),
        'eb': round(eb, 2),
        'avg_width': round(total_width / valid_images if valid_images > 0 else 0),
        'avg_height': round(total_height / valid_images if valid_images > 0 else 0),
        'dataset_size': round(total_size, 2),
        'min_width': min_width if min_width != float('inf') else 0,
        'min_height': min_height if min_height != float('inf') else 0,
        'max_width': max_width,
        'max_height': max_height,
        'valid_images': valid_images,
        'invalid_images': invalid_images,
        'invalid_paths': invalid_paths,
        'train_count': train_count,
        'test_count': test_count,
        'validation_count': validation_count,
        'other_count': other_count
    }

    return metrics, image_records

def generate_plots(df: pd.DataFrame, dataset_path: str, font_sizes: dict = None) -> Dict[str, str]:
    """Generate plots for the dataset analysis."""
    plots = {}
    
    # Default font sizes
    default_font_sizes = {
        'title': 14,
        'label': 12,
        'pie_labels': 18,
        'pie_title': 18,
        'sample_title': 26,
        'sample_subtitle': 26,
        'rgb_title': 14,
        'rgb_axis': 12
    }
    
    # Use provided font sizes or defaults
    if font_sizes:
        default_font_sizes.update(font_sizes)
    
    fs = default_font_sizes
    
    # Set style
    plt.style.use('seaborn-v0_8')
    colors = sns.color_palette('husl', n_colors=len(df['class'].unique()))
    
    # Distribution plot
    plt.figure(figsize=(12, 6))
    ax = sns.countplot(data=df, x='class', palette=colors)
    plt.title('Class Distribution', pad=20, fontsize=fs['title'], fontweight='bold')
    plt.xlabel('Class', fontsize=fs['label'])
    plt.ylabel('Count', fontsize=fs['label'])
    plt.xticks(rotation=45, ha='right', fontsize=fs['label'])
    plt.yticks(fontsize=fs['label'])

    # Add value labels on top of bars
    for i in ax.containers:
        ax.bar_label(i, padding=3, fontsize=fs['label'])

    # Add Entropy Balance (percentage) as a blue dotted reference line
    # Compute class counts and ideal count (perfect balance) for reference
    class_counts_for_entropy = df['class'].value_counts()
    total_samples_for_entropy = len(df)
    num_classes_for_entropy = len(class_counts_for_entropy)
    if num_classes_for_entropy > 0 and total_samples_for_entropy > 0:
        ideal_count_reference = total_samples_for_entropy / num_classes_for_entropy
        class_probs = (class_counts_for_entropy / total_samples_for_entropy).values
        # Entropy balance in percentage
        entropy_value = -np.sum(class_probs * np.log2(class_probs)) if np.all(class_probs > 0) else 0.0
        max_entropy_value = np.log2(num_classes_for_entropy) if num_classes_for_entropy > 1 else 1.0
        entropy_balance_pct = int((entropy_value / max_entropy_value * 100.0) * 10000) / 10000 if max_entropy_value > 0 else 0.0

        # Draw blue dotted line at the ideal count level, labeled with Entropy Balance percentage
        plt.axhline(
            y=ideal_count_reference,
            color='blue',
            linestyle=':',
            linewidth=2,
            alpha=0.9,
                label=f'Entropy Balance: {entropy_balance_pct:.2f}%'
        )
        # Show legend so the line label is visible
        plt.legend(fontsize=fs['label'], loc='upper right')

    plt.tight_layout()
    plt.savefig('distribution_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    with open('distribution_plot.png', 'rb') as f:
        plots['distribution_plot'] = base64.b64encode(f.read()).decode()

    # Insights plot (Bar chart with balance efficiency line)
    plt.figure(figsize=(12, 8))
    class_counts = df['class'].value_counts()
    colors = sns.color_palette('husl', n_colors=len(class_counts))
    
    # Create primary axis for bars
    ax1 = plt.gca()
    
    # Create bar chart
    bars = ax1.bar(range(len(class_counts)), class_counts.values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    plt.title('Class Distribution with Balance Efficiency', pad=20, fontsize=fs['pie_title'], fontweight='bold')
    ax1.set_xlabel('Classes', fontsize=fs['label'])
    ax1.set_ylabel('Number of Samples', fontsize=fs['label'])
    
    # Set x-axis labels
    ax1.set_xticks(range(len(class_counts)))
    ax1.set_xticklabels(class_counts.index, rotation=45, ha='right', fontsize=fs['label'])
    ax1.tick_params(axis='y', labelsize=fs['label'])
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height)}', ha='center', va='bottom', fontsize=fs['label'])
    
    # Calculate and draw balance efficiency line
    total_samples = len(df)
    num_classes = len(class_counts)
    ideal_count = total_samples / num_classes
    # BE = (1 - (1/C) * Σ|n_i - n̄|/n̄) × 100
    sum_abs_deviations = sum(abs(count - ideal_count) for count in class_counts)
    balance_efficiency = int(((1 - (1 / num_classes) * (sum_abs_deviations / ideal_count)) * 100) * 10000) / 10000
    
    # Calculate entropy balance for the insights plot
    if num_classes > 0 and total_samples > 0:
        class_probs = (class_counts / total_samples).values
        # Entropy balance in percentage
        entropy_value = -np.sum(class_probs * np.log2(class_probs)) if np.all(class_probs > 0) else 0.0
        max_entropy_value = np.log2(num_classes) if num_classes > 1 else 1.0
        entropy_balance_pct = int((entropy_value / max_entropy_value * 100.0) * 10000) / 10000 if max_entropy_value > 0 else 0.0
    else:
        entropy_balance_pct = 0.0
    
    # Calculate different levels for each metric
    # Balance efficiency line: shows the ideal balanced count
    balance_efficiency_level = ideal_count
    
    # Entropy balance line: shows the count that would give the current entropy balance
    # This represents how many samples per class would achieve the current entropy balance
    if entropy_balance_pct > 0:
        # Calculate what the count would be if we had the current entropy balance
        # Higher entropy balance means more uniform distribution
        entropy_balance_level = ideal_count * (entropy_balance_pct / 100.0)
    else:
        entropy_balance_level = ideal_count
    
    # Create secondary y-axis for percentages
    ax2 = ax1.twinx()
    ax2.set_ylabel('BE & EB (%)', fontsize=fs['label'], color='purple')
    ax2.tick_params(axis='y', labelsize=fs['label'], colors='purple')
    
    # Set secondary y-axis range (0-100%)
    ax2.set_ylim(0, 100)
    
    # Add horizontal lines on secondary axis to show percentage levels only
    ax2.axhline(y=balance_efficiency, color='red', linestyle='--', linewidth=2, alpha=0.8, 
                label=f'Balance Efficiency: {balance_efficiency:.2f}%')
    ax2.axhline(y=entropy_balance_pct, color='blue', linestyle=':', linewidth=2, alpha=0.9, 
                label=f'Entropy Balance: {entropy_balance_pct:.2f}%')
    
    # Add legend
    ax2.legend(fontsize=fs['label'], loc='upper right')
    
    # Add grid for better readability
    ax1.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plots['insights'] = base64.b64encode(buf.getvalue()).decode()
    plt.close()

    # Sample per class plot
    plt.figure(figsize=(15, 10))
    plt.suptitle('Sample Images per Class', fontsize=fs['sample_title'], y=0.95)
    
    # Get unique classes
    unique_classes = df['class'].unique()
    n_classes = len(unique_classes)
    
    # Calculate grid dimensions
    n_cols = min(5, n_classes)
    n_rows = (n_classes + n_cols - 1) // n_cols
    
    for idx, class_name in enumerate(unique_classes):
        plt.subplot(n_rows, n_cols, idx + 1)
        
        # Get a sample image for this class
        class_sample = df[df['class'] == class_name].iloc[0]
        img_path = class_sample['path']
        
        try:
            img = plt.imread(img_path)
            plt.imshow(img)
            plt.title(f'{class_name}', fontsize=fs['sample_subtitle'])
            plt.axis('off')
        except Exception as e:
            plt.text(0.5, 0.5, f'Error loading image:\n{str(e)}', 
                    ha='center', va='center', fontsize=fs['label'])
            plt.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plots['sample_per_class'] = base64.b64encode(buf.getvalue()).decode()
    plt.close()

    # RGB Histogram
    plt.figure(figsize=(12, 6))
    rgb_means = []
    for _, row in df.iterrows():
        try:
            img = Image.open(row['path']).convert('RGB')
            img_array = np.array(img)
            if len(img_array.shape) == 3:  # RGB image
                rgb_means.append(img_array.mean(axis=(0, 1)))
        except:
            continue
    
    if rgb_means:
        rgb_means = np.array(rgb_means)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Modern color palette
        plt.hist(rgb_means[:, 0], bins=50, alpha=0.6, color=colors[0], label='Red')
        plt.hist(rgb_means[:, 1], bins=50, alpha=0.6, color=colors[1], label='Green')
        plt.hist(rgb_means[:, 2], bins=50, alpha=0.6, color=colors[2], label='Blue')
        plt.title('RGB Channel Distribution', pad=20, fontsize=fs['rgb_title'], fontweight='bold')
        plt.xlabel('Pixel Value', labelpad=10, fontsize=fs['rgb_axis'])
        plt.ylabel('Frequency', labelpad=10, fontsize=fs['rgb_axis'])
        plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=fs['label'])
        plt.xticks(fontsize=fs['rgb_axis'])
        plt.yticks(fontsize=fs['rgb_axis'])
        plt.grid(True, alpha=0.3)
        
        # Calculate RGB insights
        rgb_insights = {
            'channel_stats': {
                'red': {
                    'mean': round(rgb_means[:, 0].mean(), 2),
                    'median': round(np.median(rgb_means[:, 0]), 2),
                    'std': round(rgb_means[:, 0].std(), 2)
                },
                'green': {
                    'mean': round(rgb_means[:, 1].mean(), 2),
                    'median': round(np.median(rgb_means[:, 1]), 2),
                    'std': round(rgb_means[:, 1].std(), 2)
                },
                'blue': {
                    'mean': round(rgb_means[:, 2].mean(), 2),
                    'median': round(np.median(rgb_means[:, 2]), 2),
                    'std': round(rgb_means[:, 2].std(), 2)
                }
            },
            'color_dominance': ['red', 'green', 'blue'][np.argmax(rgb_means.mean(axis=0))],
            'brightness_level': 'High' if rgb_means.mean() > 170 else 'Medium' if rgb_means.mean() > 85 else 'Low',
            'contrast_level': 'High' if rgb_means.std() > 50 else 'Medium' if rgb_means.std() > 25 else 'Low'
        }
    else:
        rgb_insights = None

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plots['rgb_histogram'] = base64.b64encode(buf.getvalue()).decode()
    plt.close()

    return plots, rgb_insights

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a CSV file.'})
    
    try:
        # Read CSV file
        has_headers = request.form.get('has_headers', 'true').lower() == 'true'
        df = pd.read_csv(file, header=0 if has_headers else None)
        
        if len(df.columns) < 2:
            return jsonify({'error': 'CSV file must have at least 2 columns: path and class'})
        
        # If no headers, assign column names
        if not has_headers:
            df.columns = ['path', 'class'] + [f'col{i+3}' for i in range(len(df.columns)-2)]
        else:
            # Rename the first two columns to ensure they are 'path' and 'class'
            df.columns = ['path', 'class'] + list(df.columns[2:])
        
        # Apply multiplier if specified
        multiplier = int(request.form.get('multiplier', 1))
        if multiplier > 1:
            df = pd.concat([df] * multiplier, ignore_index=True)
        
        # Get font size parameters
        font_sizes = {}
        font_size_params = [
            'title', 'label', 'pie_labels', 'pie_title', 
            'sample_title', 'sample_subtitle', 'rgb_title', 'rgb_axis'
        ]
        
        for param in font_size_params:
            value = request.form.get(f'font_size_{param}')
            if value:
                try:
                    font_sizes[param] = int(value)
                except ValueError:
                    pass  # Use default if invalid value
        
        # Analyze dataset
        metrics, image_records = analyze_dataset(df)
        if metrics['valid_images'] == 0:
            return jsonify({'error': 'No valid images found in the dataset'})
        
        # Generate plots with font size parameters
        plots, rgb_insights = generate_plots(df, metrics, font_sizes if font_sizes else None)
        
        return jsonify({
            'metrics': metrics,
            'image_analysis': {'records': image_records},
            'plots': plots,
            'rgb_histogram_insights': rgb_insights
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/regenerate_plots', methods=['POST'])
def regenerate_plots():
    """Regenerate plots with updated font sizes."""
    try:
        # Get font size parameters
        font_sizes = {}
        font_size_params = [
            'title', 'label', 'pie_labels', 'pie_title', 
            'sample_title', 'sample_subtitle', 'rgb_title', 'rgb_axis'
        ]
        
        for param in font_size_params:
            value = request.form.get(f'font_size_{param}')
            if value:
                try:
                    font_sizes[param] = int(value)
                except ValueError:
                    pass  # Use default if invalid value
        
        # Get the uploaded file from the session or request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        # Read CSV file
        has_headers = request.form.get('has_headers', 'true').lower() == 'true'
        df = pd.read_csv(file, header=0 if has_headers else None)
        
        if len(df.columns) < 2:
            return jsonify({'error': 'CSV file must have at least 2 columns: path and class'})
        
        # If no headers, assign column names
        if not has_headers:
            df.columns = ['path', 'class'] + [f'col{i+3}' for i in range(len(df.columns)-2)]
        else:
            # Rename the first two columns to ensure they are 'path' and 'class'
            df.columns = ['path', 'class'] + list(df.columns[2:])
        
        # Apply multiplier if specified
        multiplier = int(request.form.get('multiplier', 1))
        if multiplier > 1:
            df = pd.concat([df] * multiplier, ignore_index=True)
        
        # Generate plots with font size parameters
        plots, rgb_insights = generate_plots(df, "dataset", font_sizes if font_sizes else None)
        
        return jsonify({
            'plots': plots,
            'rgb_histogram_insights': rgb_insights
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/crossval', methods=['POST'])
def cross_validation():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'File must be a CSV'}), 400

        strategy = request.form.get('strategy', 'kfold')
        k = int(request.form.get('k', 5))
        has_headers = request.form.get('has_headers', 'true').lower() == 'true'
        multiplier = float(request.form.get('multiplier', 1.0))

        # Read the CSV file
        df = pd.read_csv(file, header=0 if has_headers else None)
        
        # If no headers, rename columns
        if not has_headers:
            df.columns = ['path', 'class'] + [f'col{i+3}' for i in range(len(df.columns)-2)]
        else:
            # Ensure first two columns are path and class
            df.columns = ['path', 'class'] + list(df.columns[2:])

        # Apply multiplier if specified
        if multiplier > 1:
            df = pd.concat([df] * int(multiplier), ignore_index=True)

        # Get unique classes
        classes = df['class'].unique()
        
        # Create figure and axis
        plt.figure(figsize=(12, 6))
        
        # Prepare data for stacked bar chart
        fold_data = []
        for i in range(k):
            if strategy == 'stratified':
                # For stratified, we'll show class distribution in each fold
                fold_indices = np.arange(len(df)) % k == i
                fold_df = df[fold_indices]
                class_counts = fold_df['class'].value_counts()
                fold_data.append(class_counts)
            else:
                # For regular k-fold, we'll show equal splits
                fold_size = len(df) // k
                start_idx = i * fold_size
                end_idx = start_idx + fold_size if i < k-1 else len(df)
                fold_df = df.iloc[start_idx:end_idx]
                class_counts = fold_df['class'].value_counts()
                fold_data.append(class_counts)

        # Create stacked bar chart
        bottom = np.zeros(k)
        colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
        
        for i, class_name in enumerate(classes):
            values = [fold_data[j].get(class_name, 0) for j in range(k)]
            plt.bar(range(k), values, bottom=bottom, label=class_name, color=colors[i])
            bottom += values

        plt.xlabel('Fold Number')
        plt.ylabel('Number of Samples')
        plt.title(f'Cross-Validation Split ({strategy.capitalize()})')
        plt.xticks(range(k), [f'Fold {i+1}' for i in range(k)])
        plt.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # Save plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return jsonify({
            'plot': plot_base64,
            'k': k,
            'strategy': strategy,
            'n_samples': len(df)
        })

    except Exception as e:
        print(f"Error in cross_validation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/dimred', methods=['POST'])
def dimensionality_reduction():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    try:
        # Read CSV file
        has_headers = request.form.get('has_headers', 'true').lower() == 'true'
        df = pd.read_csv(file, header=0 if has_headers else None)
        
        if len(df.columns) < 2:
            return jsonify({'error': 'CSV file must have at least 2 columns: path and class'})
        
        # If no headers, assign column names
        if not has_headers:
            df.columns = ['path', 'class'] + [f'col{i+3}' for i in range(len(df.columns)-2)]
        else:
            # Rename the first two columns to ensure they are 'path' and 'class'
            df.columns = ['path', 'class'] + list(df.columns[2:])
        
        # Get parameters
        method = request.form.get('method', 'pca')
        multiplier = int(request.form.get('multiplier', 1))
        
        # Apply multiplier
        if multiplier > 1:
            df = pd.concat([df] * multiplier, ignore_index=True)
        
        # Extract features from images
        features = []
        valid_paths = []
        valid_classes = []
        
        for _, row in df.iterrows():
            try:
                img = Image.open(row['path']).convert('RGB')
                img_array = np.array(img)
                if len(img_array.shape) == 3:  # RGB image
                    features.append(img_array.mean(axis=(0, 1)))
                    valid_paths.append(row['path'])
                    valid_classes.append(row['class'])
            except:
                continue
        
        if not features:
            return jsonify({'error': 'No valid images found for dimensionality reduction'})
        
        features = np.array(features)
        
        # Apply dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=2)
            title = 'PCA Visualization'
            n_components = 2
        else:  # tsne
            reducer = TSNE(n_components=3, random_state=42)
            title = 'T-SNE Visualization'
            n_components = 3
        
        reduced_features = reducer.fit_transform(features)
        
        # Prepare data for interactive plot
        unique_classes = np.unique(valid_classes)
        plot_data = []
        
        for i, cls in enumerate(unique_classes):
            mask = np.array(valid_classes) == cls
            if method == 'pca':
                plot_data.append({
                    'x': reduced_features[mask, 0].tolist(),
                    'y': reduced_features[mask, 1].tolist(),
                    'mode': 'markers',
                    'type': 'scatter',
                    'name': cls,
                    'marker': {
                        'size': 6,
                        'opacity': 0.7
                    }
                })
            else:  # tsne 3D
                plot_data.append({
                    'x': reduced_features[mask, 0].tolist(),
                    'y': reduced_features[mask, 1].tolist(),
                    'z': reduced_features[mask, 2].tolist(),
                    'mode': 'markers',
                    'type': 'scatter3d',
                    'name': cls,
                    'marker': {
                        'size': 4,
                        'opacity': 0.7
                    }
                })
        
        layout = {
            'title': title,
            'scene' if method == 'tsne' else 'xaxis': {
                'title': 'Component 1' if method == 'tsne' else 'Component 1'
            },
            'yaxis' if method == 'pca' else 'scene': {
                'title': 'Component 2' if method == 'pca' else 'Component 2'
            }
        }
        
        if method == 'tsne':
            layout['scene']['zaxis'] = {'title': 'Component 3'}
        
        return jsonify({
            'plot_data': plot_data,
            'layout': layout,
            'method': method,
            'n_components': n_components,
            'n_samples': len(features)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7001, debug=True) 
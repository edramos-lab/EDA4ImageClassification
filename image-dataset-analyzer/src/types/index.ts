export interface Metric {
    label: string;
    value: string;
    unit?: string;
    description?: string;
}

export interface Metrics {
    class_distribution: Record<string, number>;
    total_sizes: number;
    avg_width: number;
    avg_height: number;
    size_distribution: Record<string, number>;
    max_samples: number;
    min_samples: number;
    max_class: string;
    min_class: string;
    imbalance_ratio: number;
    balancing_efficiency: number;
    entropy_balance: number;
    dataset_size: number;
    min_width: number;
    min_height: number;
    max_width: number;
    max_height: number;
    valid_images: number;
    invalid_images: number;
    invalid_paths: string[];
    num_classes: number;
    avg_samples_per_class: number;
    std_dev: number;
    cv: number;
    gini: number;
    total_images: number;
    train_count: number;
    test_count: number;
    validation_count: number;
    other_count: number;
}

export interface ImageRecord {
    path: string;
    class: string;
    width: number;
    height: number;
    size_kb: number;
}

export interface Plots {
    distribution: string;
    insights: string;
    sample_per_class: string;
    rgb_histogram: string;
}

export interface RGBHistogramInsights {
    channel_stats: {
        red: ChannelStats;
        green: ChannelStats;
        blue: ChannelStats;
    };
    color_dominance: string;
    brightness_level: string;
    contrast_level: string;
}

export interface ChannelStats {
    mean: number;
    median: number;
    std: number;
}

export interface AnalysisResponse {
    metrics: Metrics;
    image_analysis: {
        records: ImageRecord[];
    };
    plots: Plots;
    rgb_histogram_insights: RGBHistogramInsights;
}

export interface CrossValidationData {
    fold_results: {
        fold: number;
        accuracy: number;
        precision: number;
        recall: number;
        f1_score: number;
    }[];
    overall_metrics: {
        accuracy: number;
        precision: number;
        recall: number;
        f1_score: number;
    };
}

export interface CrossValidationResponse {
    plot: string;
    k: number;
    strategy: 'kfold' | 'stratified';
    n_samples: number;
}

export interface DimensionalityReductionResponse {
    plot_data: Array<{
        x: number[];
        y: number[];
        z?: number[];
        mode: string;
        type: string;
        name: string;
        marker: {
            size: number;
            opacity: number;
        };
    }>;
    layout: {
        title: string;
        scene?: {
            xaxis: { title: string };
            yaxis: { title: string };
            zaxis?: { title: string };
        };
        xaxis?: { title: string };
        yaxis?: { title: string };
    };
    method: string;
    n_components: number;
    n_samples: number;
} 
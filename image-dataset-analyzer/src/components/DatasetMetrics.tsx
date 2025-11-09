import React from 'react';
import { 
    Box, 
    Card, 
    CardContent, 
    Typography
} from '@mui/material';
import { Metrics, ImageRecord, Metric } from '../types';

interface DatasetMetricsProps {
    metrics: Metrics | null;
    imageRecords?: ImageRecord[];
}

interface MetricGroup {
    title: string;
    metrics: Metric[];
}

const DatasetMetrics: React.FC<DatasetMetricsProps> = ({ metrics, imageRecords }) => {
    if (!metrics) {
        return null;
    }

    // Calculate metrics from imageRecords if available
    const calculateFromRecords = (records: ImageRecord[]) => {
        if (!records || records.length === 0) return null;

        const classCounts = records.reduce((acc, record) => {
            acc[record.class] = (acc[record.class] || 0) + 1;
            return acc;
        }, {} as Record<string, number>);

        const counts = Object.values(classCounts);
        const totalSamples = counts.reduce((sum, count) => sum + count, 0);
        const numClasses = counts.length;
        const maxCount = Math.max(...counts);
        const minCount = Math.min(...counts);
        const avgSamplesPerClass = totalSamples / numClasses;
        const imbalanceRatio = maxCount / minCount;

        // Calculate standard deviation
        const stdDev = Math.sqrt(
            counts.reduce((sum, count) => sum + Math.pow(count - avgSamplesPerClass, 2), 0) / numClasses
        );

        // Calculate coefficient of variation
        const cv = (stdDev / avgSamplesPerClass) * 100;

        // Calculate Gini coefficient
        const sortedCounts = [...counts].sort((a, b) => a - b);
        const n = sortedCounts.length;
        const index = Array.from({ length: n }, (_, i) => i + 1);
        const gini = ((2 * sortedCounts.reduce((sum, count, i) => sum + index[i] * count, 0)) / 
                      (n * sortedCounts.reduce((sum, count) => sum + count, 0)) - (n + 1) / n) * 100;

        // Calculate entropy
        const classProbabilities = counts.map(count => count / totalSamples);
        const entropy = -classProbabilities.reduce((sum, p) => sum + p * Math.log2(p), 0);
        const maxEntropy = Math.log2(numClasses);
        const entropyBalance = Math.floor((entropy / maxEntropy) * 100 * 10000) / 10000;

        // Calculate balance efficiency
        // BE = (1 - (1/C) * Σ|n_i - n̄|/n̄) × 100
        const idealCount = totalSamples / numClasses;
        const sumAbsDeviations = counts.reduce((sum, count) => sum + Math.abs(count - idealCount), 0);
        const balanceEfficiency = maxCount === 0 ? 0 : Math.floor(((1 - (1 / numClasses) * (sumAbsDeviations / idealCount)) * 100) * 10000) / 10000;

        return {
            numClasses,
            totalSamples,
            avgSamplesPerClass,
            maxSamplesPerClass: maxCount,
            minSamplesPerClass: minCount,
            imbalanceRatio,
            stdDev,
            cv,
            gini,
            entropyBalance,
            balanceEfficiency
        };
    };

    const calculatedMetrics = calculateFromRecords(imageRecords || []);

    const metricGroups: MetricGroup[] = [
        {
            title: 'Dataset Statistics',
            metrics: [
                {
                    label: 'Number of Classes',
                    value: (metrics?.num_classes || calculatedMetrics?.numClasses || 0).toString(),
                    description: 'Total number of unique classes in the dataset'
                },
                {
                    label: 'Total Images',
                    value: (metrics?.total_images || calculatedMetrics?.totalSamples || 0).toString(),
                    description: 'Total number of valid images in the dataset'
                },
                {
                    label: 'Avg Samples per Class',
                    value: (metrics?.avg_samples_per_class || calculatedMetrics?.avgSamplesPerClass || 0).toFixed(2),
                    description: 'Average number of samples per class'
                }
            ]
        },
        {
            title: 'Distribution Metrics',
            metrics: [
                { 
                    label: 'Standard Deviation', 
                    value: (calculatedMetrics?.stdDev || 0).toFixed(2)
                },
                { 
                    label: 'Coefficient of Variation', 
                    value: (calculatedMetrics?.cv || 0).toFixed(2),
                    unit: '%'
                },
                { 
                    label: 'Gini Coefficient', 
                    value: (calculatedMetrics?.gini || 0).toFixed(2),
                    unit: '%'
                },
            ],
        },
        {
            title: 'Imbalance Metrics',
            metrics: [
                {
                    label: 'Imbalance Ratio',
                    value: (calculatedMetrics?.imbalanceRatio || 0).toFixed(2),
                    description: 'Ratio between the class with most samples and the class with least samples'
                },
                {
                    label: 'Balance Efficiency',
                    value: (calculatedMetrics?.balanceEfficiency || 0).toFixed(2),
                    unit: '%',
                    description: 'Measure of how evenly distributed the samples are across classes'
                },
                {
                    label: 'Entropy Balance',
                    value: (calculatedMetrics?.entropyBalance || 0).toFixed(2),
                    unit: '%',
                    description: 'Measure of class distribution uniformity based on entropy'
                }
            ],
        },
        {
            title: 'Class Distribution',
            metrics: [
                { 
                    label: 'Max Samples', 
                    value: metrics.max_samples?.toString() ?? 'N/A'
                },
                { 
                    label: 'Min Samples', 
                    value: metrics.min_samples?.toString() ?? 'N/A'
                },
                { 
                    label: 'Max Class', 
                    value: metrics.max_class ?? 'N/A'
                },
                { 
                    label: 'Min Class', 
                    value: metrics.min_class ?? 'N/A'
                },
                { 
                    label: 'Imbalance Ratio', 
                    value: (calculatedMetrics?.imbalanceRatio || 0).toFixed(2)
                },
            ],
        },
        {
            title: 'Image Size Metrics',
            metrics: [
                { 
                    label: 'Average Width', 
                    value: metrics.avg_width?.toFixed(0) ?? 'N/A',
                    unit: 'px'
                },
                { 
                    label: 'Average Height', 
                    value: metrics.avg_height?.toFixed(0) ?? 'N/A',
                    unit: 'px'
                },
                { 
                    label: 'Min Width', 
                    value: metrics.min_width?.toFixed(0) ?? 'N/A',
                    unit: 'px'
                },
                { 
                    label: 'Min Height', 
                    value: metrics.min_height?.toFixed(0) ?? 'N/A',
                    unit: 'px'
                },
                { 
                    label: 'Max Width', 
                    value: metrics.max_width?.toFixed(0) ?? 'N/A',
                    unit: 'px'
                },
                { 
                    label: 'Max Height', 
                    value: metrics.max_height?.toFixed(0) ?? 'N/A',
                    unit: 'px'
                },
            ],
        },
        {
            title: 'Validation Metrics',
            metrics: [
                { 
                    label: 'Valid Images', 
                    value: metrics.valid_images?.toString() ?? 'N/A',
                    unit: 'images'
                },
                { 
                    label: 'Invalid Images', 
                    value: metrics.invalid_images?.toString() ?? 'N/A',
                    unit: 'images'
                },
                { 
                    label: 'Train Images', 
                    value: metrics.train_count?.toString() ?? 'N/A',
                    unit: 'images',
                    description: 'Number of images in training set'
                },
                { 
                    label: 'Test Images', 
                    value: metrics.test_count?.toString() ?? 'N/A',
                    unit: 'images',
                    description: 'Number of images in test set'
                },
                { 
                    label: 'Validation Images', 
                    value: metrics.validation_count?.toString() ?? 'N/A',
                    unit: 'images',
                    description: 'Number of images in validation set'
                },
                { 
                    label: 'Other Images', 
                    value: metrics.other_count?.toString() ?? 'N/A',
                    unit: 'images',
                    description: 'Images not in train/test/validation folders'
                },
            ],
        },
    ];

    return (
        <Box sx={{ mt: 4 }}>
            <Typography variant="h5" gutterBottom>
                Dataset Metrics
            </Typography>
            <Box sx={{ 
                display: 'grid',
                gridTemplateColumns: {
                    xs: '1fr',
                    sm: 'repeat(2, 1fr)',
                    md: 'repeat(4, 1fr)'
                },
                gap: 2
            }}>
                {metricGroups.map((group) => (
                    <Card key={group.title}>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                {group.title}
                            </Typography>
                            {group.metrics.map((metric) => (
                                <Box key={metric.label} sx={{ mb: 1 }}>
                                    <Typography variant="body2" color="text.secondary">
                                        {metric.label}
                                    </Typography>
                                    <Typography variant="body1">
                                        {metric.value}
                                        {metric.unit}
                                    </Typography>
                                    {metric.description && (
                                        <Typography variant="caption" color="text.secondary" sx={{ fontStyle: 'italic' }}>
                                            {metric.description}
                                        </Typography>
                                    )}
                                </Box>
                            ))}
                        </CardContent>
                    </Card>
                ))}
            </Box>
            
            {metrics.invalid_paths && metrics.invalid_paths.length > 0 && (
                <Card sx={{ mt: 2 }}>
                    <CardContent>
                        <Typography variant="h6" gutterBottom>
                            Invalid Image Paths
                        </Typography>
                        <Typography variant="body2" component="pre" sx={{ 
                            whiteSpace: 'pre-wrap',
                            wordBreak: 'break-all',
                            maxHeight: '200px',
                            overflow: 'auto'
                        }}>
                            {metrics.invalid_paths.join('\n')}
                        </Typography>
                    </CardContent>
                </Card>
            )}
        </Box>
    );
};

export default DatasetMetrics; 
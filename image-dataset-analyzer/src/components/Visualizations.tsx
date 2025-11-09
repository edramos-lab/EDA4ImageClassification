import React, { useState, useEffect, useRef } from 'react';
import {
    Box,
    Typography,
    Card,
    CardContent,
    CardActions,
    Button,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    TextField,
    SelectChangeEvent,
    Paper,
    Divider,
    Grid,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    TableContainer,
    Table,
    TableHead,
    TableBody,
    TableRow,
    TableCell,
    FormHelperText,
    Slider
} from '@mui/material';
import Plot from 'react-plotly.js';
import { CrossValidationResponse, DimensionalityReductionResponse, RGBHistogramInsights, Plots } from '../types';

interface VisualizationsProps {
    plots: Plots;
    rgbHistogramInsights: RGBHistogramInsights | null;
    onCrossValidation: (file: File, strategy: 'kfold' | 'stratified', k: number, hasHeaders: boolean, multiplier: number) => Promise<void>;
    onDimensionalityReduction: (file: File, method: 'pca' | 'tsne', hasHeaders: boolean, multiplier: number) => Promise<void>;
    crossValidationData: CrossValidationResponse | null;
    dimensionalityReductionData: DimensionalityReductionResponse | null;
    datasetFile: File | null;
}

const Visualizations: React.FC<VisualizationsProps> = ({
    plots,
    rgbHistogramInsights,
    onCrossValidation,
    onDimensionalityReduction,
    crossValidationData,
    dimensionalityReductionData,
    datasetFile
}) => {
    const [cvDialogOpen, setCvDialogOpen] = useState(false);
    const [drDialogOpen, setDrDialogOpen] = useState(false);
    const [cvStrategy, setCvStrategy] = useState<'kfold' | 'stratified'>('kfold');
    const [cvK, setCvK] = useState<number>(5);
    const [cvHasHeaders, setCvHasHeaders] = useState<boolean>(true);
    const [cvMultiplier, setCvMultiplier] = useState<number>(1);
    const [dimRedMethod, setDimRedMethod] = useState<'pca' | 'tsne'>('pca');
    const [dimRedHasHeaders, setDimRedHasHeaders] = useState<boolean>(true);
    const [dimRedMultiplier, setDimRedMultiplier] = useState<number>(1);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleCvStrategyChange = (event: SelectChangeEvent) => {
        setCvStrategy(event.target.value as 'kfold' | 'stratified');
    };

    const handleDimRedMethodChange = (event: SelectChangeEvent) => {
        setDimRedMethod(event.target.value as 'pca' | 'tsne');
    };

    const handleCvSubmit = async () => {
        if (!datasetFile) {
            setError('No dataset file available');
            return;
        }
        try {
            setIsLoading(true);
            setError(null);
            console.log('Starting cross-validation with file:', datasetFile.name);
            await onCrossValidation(datasetFile, cvStrategy, cvK, cvHasHeaders, cvMultiplier);
            console.log('Cross-validation completed successfully');
            setCvDialogOpen(false);
        } catch (err) {
            console.error('Cross-validation error:', err);
            setError(err instanceof Error ? err.message : 'An error occurred during cross-validation');
        } finally {
            setIsLoading(false);
        }
    };

    const handleDrSubmit = async () => {
        if (!datasetFile) {
            setError('No dataset file available');
            return;
        }
        try {
            setIsLoading(true);
            setError(null);
            console.log('Starting dimensionality reduction with file:', datasetFile.name);
            await onDimensionalityReduction(datasetFile, dimRedMethod, dimRedHasHeaders, dimRedMultiplier);
            console.log('Dimensionality reduction completed successfully');
            setDrDialogOpen(false);
        } catch (err) {
            console.error('Dimensionality reduction error:', err);
            setError(err instanceof Error ? err.message : 'An error occurred during dimensionality reduction');
        } finally {
            setIsLoading(false);
        }
    };

    // Add useEffect to monitor state changes
    useEffect(() => {
        console.log('Cross-validation data updated:', crossValidationData);
    }, [crossValidationData]);

    useEffect(() => {
        console.log('Dimensionality reduction data updated:', dimensionalityReductionData);
    }, [dimensionalityReductionData]);

    return (
        <Box sx={{ mt: 4 }}>
            {error && (
                <Typography color="error" sx={{ mb: 2 }}>
                    {error}
                </Typography>
            )}
            <Typography variant="h5" gutterBottom>
                Dataset Visualizations
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
                {plots.distribution && (
                    <Box sx={{ flex: '1 1 400px', minWidth: 0 }}>
                        <Paper 
                            elevation={2} 
                            sx={{ 
                                p: 3, 
                                height: '100%',
                                borderRadius: 2,
                                background: 'linear-gradient(145deg, #ffffff 0%, #f5f5f5 100%)',
                                transition: 'transform 0.2s ease-in-out',
                                '&:hover': {
                                    transform: 'translateY(-4px)',
                                    boxShadow: '0 8px 16px rgba(0,0,0,0.1)'
                                }
                            }}
                        >
                            <Typography 
                                variant="h6" 
                                gutterBottom 
                                sx={{ 
                                    fontWeight: 600,
                                    color: '#1a1a1a',
                                    borderBottom: '2px solid #e0e0e0',
                                    pb: 1
                                }}
                            >
                                Class Distribution
                            </Typography>
                            <img 
                                src={`data:image/png;base64,${plots.distribution}`} 
                                alt="Class Distribution" 
                                style={{ 
                                    width: '100%', 
                                    height: 'auto',
                                    borderRadius: '8px',
                                    fontSize: '1.5rem'
                                }} 
                            />
                        </Paper>
                    </Box>
                )}

                {plots.insights && (
                    <Box sx={{ flex: '1 1 400px', minWidth: 0 }}>
                        <Paper 
                            elevation={2} 
                            sx={{ 
                                p: 3, 
                                height: '100%',
                                borderRadius: 2,
                                background: 'linear-gradient(145deg, #ffffff 0%, #f5f5f5 100%)',
                                transition: 'transform 0.2s ease-in-out',
                                '&:hover': {
                                    transform: 'translateY(-4px)',
                                    boxShadow: '0 8px 16px rgba(0,0,0,0.1)'
                                }
                            }}
                        >
                            <Typography 
                                variant="h6" 
                                gutterBottom 
                                sx={{ 
                                    fontWeight: 600,
                                    color: '#1a1a1a',
                                    borderBottom: '2px solid #e0e0e0',
                                    pb: 1,
                                    fontSize: '2rem'
                                }}
                            >
                                Class Distribution with Balance Efficiency
                            </Typography>
                            <img 
                                src={`data:image/png;base64,${plots.insights}`} 
                                alt="Dimension Distribution" 
                                style={{ 
                                    width: '100%', 
                                    height: 'auto',
                                    borderRadius: '8px',
                                    fontSize: '10rem'
                                }} 
                            />
                        </Paper>
                    </Box>
                )}

                {plots.sample_per_class && (
                    <Box sx={{ flex: '1 1 400px', minWidth: 0 }}>
                        <Paper 
                            elevation={2} 
                            sx={{ 
                                p: 3, 
                                height: '100%',
                                borderRadius: 2,
                                background: 'linear-gradient(145deg, #ffffff 0%, #f5f5f5 100%)',
                                transition: 'transform 0.2s ease-in-out',
                                '&:hover': {
                                    transform: 'translateY(-4px)',
                                    boxShadow: '0 8px 16px rgba(0,0,0,0.1)'
                                }
                            }}
                        >
                            <Typography 
                                variant="h6" 
                                gutterBottom 
                                sx={{ 
                                    fontWeight: 600,
                                    color: '#1a1a1a',
                                    borderBottom: '2px solid #e0e0e0',
                                    pb: 1
                                }}
                            >
                                Sample Images per Class
                            </Typography>
                            <img 
                                src={`data:image/png;base64,${plots.sample_per_class}`} 
                                alt="Samples per Class" 
                                style={{ 
                                    width: '100%', 
                                    height: 'auto',
                                    borderRadius: '8px',
                                    fontSize: '10rem'
                                }} 
                            />
                        </Paper>
                    </Box>
                )}

                {plots.rgb_histogram && (
                    <Box sx={{ flex: '1 1 400px', minWidth: 0 }}>
                        <Paper 
                            elevation={2} 
                            sx={{ 
                                p: 3, 
                                height: '100%',
                                borderRadius: 2,
                                background: 'linear-gradient(145deg, #ffffff 0%, #f5f5f5 100%)',
                                transition: 'transform 0.2s ease-in-out',
                                '&:hover': {
                                    transform: 'translateY(-4px)',
                                    boxShadow: '0 8px 16px rgba(0,0,0,0.1)'
                                }
                            }}
                        >
                            <Typography 
                                variant="h6" 
                                gutterBottom 
                                sx={{ 
                                    fontWeight: 600,
                                    color: '#1a1a1a',
                                    borderBottom: '2px solid #e0e0e0',
                                    pb: 1
                                }}
                            >
                                RGB Channel Distribution
                            </Typography>
                            <img 
                                src={`data:image/png;base64,${plots.rgb_histogram}`} 
                                alt="RGB Histogram" 
                                style={{ 
                                    width: '100%', 
                                    height: 'auto',
                                    borderRadius: '8px'
                                }} 
                            />
                            {rgbHistogramInsights && (
                                <RGBHistogram insights={rgbHistogramInsights} />
                            )}
                        </Paper>
                    </Box>
                )}
            </Box>

            <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
                <Button 
                    variant="contained" 
                    color="primary" 
                    onClick={() => setCvDialogOpen(true)}
                >
                    Perform k fold splits
                </Button>
                <Button 
                    variant="contained" 
                    color="secondary" 
                    onClick={() => setDrDialogOpen(true)}
                >
                    Perform Dimensionality Reduction
                </Button>
            </Box>

            <Dialog open={cvDialogOpen} onClose={() => !isLoading && setCvDialogOpen(false)}>
                <DialogTitle>Cross Validation Settings</DialogTitle>
                <DialogContent>
                    {error && (
                        <Typography color="error" sx={{ mb: 2 }}>
                            {error}
                        </Typography>
                    )}
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                        Configure how your dataset will be split for cross-validation.
                    </Typography>
                    <FormControl fullWidth sx={{ mt: 2 }}>
                        <InputLabel>Strategy</InputLabel>
                        <Select
                            value={cvStrategy}
                            label="Strategy"
                            onChange={handleCvStrategyChange}
                            disabled={isLoading}
                        >
                            <MenuItem value="kfold">K-Fold</MenuItem>
                            <MenuItem value="stratified">Stratified K-Fold</MenuItem>
                        </Select>
                        <FormHelperText>
                            {cvStrategy === 'kfold' 
                                ? 'Standard k-fold cross-validation splits the data into k equal parts'
                                : 'Stratified k-fold preserves the percentage of samples for each class'}
                        </FormHelperText>
                    </FormControl>
                    <TextField
                        fullWidth
                        label="Number of Folds (k)"
                        type="number"
                        value={cvK}
                        onChange={(e) => setCvK(Number(e.target.value))}
                        disabled={isLoading}
                        sx={{ mt: 2 }}
                    />
                    <TextField
                        fullWidth
                        label="Dataset Size Multiplier"
                        type="number"
                        value={cvMultiplier}
                        onChange={(e) => setCvMultiplier(Number(e.target.value))}
                        disabled={isLoading}
                        sx={{ mt: 2 }}
                    />
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setCvDialogOpen(false)} disabled={isLoading}>
                        Cancel
                    </Button>
                    <Button 
                        onClick={handleCvSubmit} 
                        variant="contained"
                        disabled={isLoading}
                    >
                        {isLoading ? 'Running...' : 'Run'}
                    </Button>
                </DialogActions>
            </Dialog>

            <Dialog open={drDialogOpen} onClose={() => setDrDialogOpen(false)}>
                <DialogTitle>Dimensionality Reduction Settings</DialogTitle>
                <DialogContent>
                    <FormControl fullWidth sx={{ mt: 2 }}>
                        <InputLabel>Method</InputLabel>
                        <Select
                            value={dimRedMethod}
                            label="Method"
                            onChange={handleDimRedMethodChange}
                        >
                            <MenuItem value="pca">PCA</MenuItem>
                            <MenuItem value="tsne">t-SNE</MenuItem>
                        </Select>
                    </FormControl>
                    <TextField
                        fullWidth
                        label="Dataset Size Multiplier"
                        type="number"
                        value={dimRedMultiplier}
                        onChange={(e) => setDimRedMultiplier(Number(e.target.value))}
                        sx={{ mt: 2 }}
                    />
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setDrDialogOpen(false)}>Cancel</Button>
                    <Button 
                        onClick={handleDrSubmit} 
                        variant="contained"
                        disabled={isLoading}
                    >
                        {isLoading ? 'Running...' : 'Run'}
                    </Button>
                </DialogActions>
            </Dialog>

            {crossValidationData && (
                <Paper elevation={2} sx={{ p: 2 }}>
                    <CrossValidationResults data={crossValidationData} />
                </Paper>
            )}

            {dimensionalityReductionData && (
                <Paper elevation={3} sx={{ p: 2, mt: 3 }}>
                    <Typography variant="h6" gutterBottom>
                        Dimensionality Reduction Results
                    </Typography>
                    {dimensionalityReductionData.plot_data && (
                        <Box sx={{ mt: 2 }}>
                            <DimensionalityReduction 
                                plot_data={dimensionalityReductionData.plot_data} 
                                layout={dimensionalityReductionData.layout} 
                                method={dimRedMethod} 
                                n_components={dimensionalityReductionData.n_components} 
                                n_samples={dimensionalityReductionData.n_samples} 
                            />
                        </Box>
                    )}
                </Paper>
            )}
        </Box>
    );
};

const DimensionalityReduction: React.FC<{
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
}> = ({ plot_data, layout, method, n_components, n_samples }) => {
    const plotRef = useRef<Plot>(null);

    return (
        <Paper 
            elevation={2} 
            sx={{ 
                p: 2, 
                display: 'flex', 
                flexDirection: 'column',
                background: 'linear-gradient(145deg, #ffffff 0%, #f5f5f5 100%)',
                borderRadius: 2,
                transition: 'all 0.3s ease',
                '&:hover': {
                    transform: 'translateY(-2px)',
                    boxShadow: '0 4px 20px rgba(0,0,0,0.1)'
                }
            }}
        >
            <Typography 
                variant="h6" 
                gutterBottom 
                sx={{ 
                    fontWeight: 600,
                    color: '#1a237e',
                    borderBottom: '2px solid #e0e0e0',
                    pb: 1,
                    mb: 2
                }}
            >
                {method === 'pca' ? 'PCA' : 'T-SNE'} Visualization
            </Typography>
            
            <Box sx={{ width: '100%', height: 500 }}>
                <Plot
                    ref={plotRef}
                    data={plot_data as any}
                    layout={{
                        ...layout,
                        width: undefined,
                        height: 480,
                        margin: { l: 50, r: 50, t: 50, b: 50 },
                        showlegend: true,
                        legend: {
                            x: 1.02,
                            y: 1,
                            bgcolor: 'rgba(255, 255, 255, 0.8)',
                            bordercolor: '#ccc',
                            borderwidth: 1
                        }
                    } as any}
                    config={{
                        displayModeBar: true,
                        displaylogo: false,
                        modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
                        toImageButtonOptions: {
                            format: 'png',
                            filename: `${method}_visualization`,
                            height: 600,
                            width: 800,
                            scale: 2
                        }
                    }}
                    style={{ width: '100%', height: '100%' }}
                    useResizeHandler={true}
                />
            </Box>
            
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Number of samples: {n_samples} | Components: {n_components}
            </Typography>
        </Paper>
    );
};

const RGBHistogram: React.FC<{ insights: RGBHistogramInsights | null }> = ({ insights }) => {
    if (!insights || !insights.channel_stats) return null;

    const { red, green, blue } = insights.channel_stats;
    if (!red || !green || !blue) return null;

    // Calculate channel dominance percentage
    const totalMean = red.mean + green.mean + blue.mean;
    const redPercentage = (red.mean / totalMean * 100).toFixed(1);
    const greenPercentage = (green.mean / totalMean * 100).toFixed(1);
    const bluePercentage = (blue.mean / totalMean * 100).toFixed(1);

    // Determine color temperature
    const getColorTemperature = () => {
        const redBlueRatio = red.mean / blue.mean;
        if (redBlueRatio > 1.2) return 'Warm';
        if (redBlueRatio < 0.8) return 'Cool';
        return 'Neutral';
    };

    // Determine saturation level
    const getSaturationLevel = () => {
        const maxChannel = Math.max(red.mean, green.mean, blue.mean);
        const minChannel = Math.min(red.mean, green.mean, blue.mean);
        const saturation = (maxChannel - minChannel) / maxChannel;
        if (saturation > 0.5) return 'High';
        if (saturation > 0.2) return 'Medium';
        return 'Low';
    };

    return (
        <Box sx={{ mt: 4 }}>
            <Typography variant="h6" gutterBottom>
                RGB Channel Statistics
            </Typography>
            <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                    <Box sx={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                        <Box sx={{ flex: 1, minWidth: '200px' }}>
                            <Typography variant="subtitle1" color="error">Red Channel</Typography>
                            <Typography>Mean: {red.mean?.toFixed(2) || 'N/A'}</Typography>
                            <Typography>Median: {red.median?.toFixed(2) || 'N/A'}</Typography>
                            <Typography>Std Dev: {red.std?.toFixed(2) || 'N/A'}</Typography>
                            <Typography>Dominance: {redPercentage}%</Typography>
                        </Box>
                        <Box sx={{ flex: 1, minWidth: '200px' }}>
                            <Typography variant="subtitle1" color="success.main">Green Channel</Typography>
                            <Typography>Mean: {green.mean?.toFixed(2) || 'N/A'}</Typography>
                            <Typography>Median: {green.median?.toFixed(2) || 'N/A'}</Typography>
                            <Typography>Std Dev: {green.std?.toFixed(2) || 'N/A'}</Typography>
                            <Typography>Dominance: {greenPercentage}%</Typography>
                        </Box>
                        <Box sx={{ flex: 1, minWidth: '200px' }}>
                            <Typography variant="subtitle1" color="info.main">Blue Channel</Typography>
                            <Typography>Mean: {blue.mean?.toFixed(2) || 'N/A'}</Typography>
                            <Typography>Median: {blue.median?.toFixed(2) || 'N/A'}</Typography>
                            <Typography>Std Dev: {blue.std?.toFixed(2) || 'N/A'}</Typography>
                            <Typography>Dominance: {bluePercentage}%</Typography>
                        </Box>
                    </Box>

                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                        <Box>
                            <Typography variant="subtitle1">Color Analysis</Typography>
                            <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
                                <Box>
                                    <Typography variant="subtitle2">Dominant Color</Typography>
                                    <Typography color={
                                        insights.color_dominance === 'red' ? 'error' :
                                        insights.color_dominance === 'green' ? 'success.main' :
                                        'info.main'
                                    }>
                                        {insights.color_dominance ? insights.color_dominance.charAt(0).toUpperCase() + insights.color_dominance.slice(1) : 'N/A'}
                                    </Typography>
                                </Box>
                                <Box>
                                    <Typography variant="subtitle2">Color Temperature</Typography>
                                    <Typography>{getColorTemperature()}</Typography>
                                </Box>
                                <Box>
                                    <Typography variant="subtitle2">Saturation Level</Typography>
                                    <Typography>{getSaturationLevel()}</Typography>
                                </Box>
                            </Box>
                        </Box>

                        <Box>
                            <Typography variant="subtitle1">Image Quality</Typography>
                            <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
                                <Box>
                                    <Typography variant="subtitle2">Brightness</Typography>
                                    <Typography>{insights.brightness_level || 'N/A'}</Typography>
                                </Box>
                                <Box>
                                    <Typography variant="subtitle2">Contrast</Typography>
                                    <Typography>{insights.contrast_level || 'N/A'}</Typography>
                                </Box>
                            </Box>
                        </Box>
                    </Box>
                </Box>
            </Paper>
        </Box>
    );
};

const CrossValidationResults: React.FC<{ data: CrossValidationResponse }> = ({ data }) => {
    return (
        <Box sx={{ mt: 2 }}>
            <Paper elevation={2} sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                    Cross-Validation Split Visualization
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                    This visualization shows how your dataset is split into {data.k} folds for cross-validation.
                    {data.strategy === 'stratified' 
                        ? ' The stratified approach ensures that each fold maintains the same class distribution as the original dataset.'
                        : ' Each fold contains an equal number of samples from the dataset.'}
                </Typography>
                {data.plot && (
                    <Box sx={{ 
                        mt: 2,
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        gap: 2
                    }}>
                        <Paper 
                            elevation={1} 
                            sx={{ 
                                p: 2,
                                width: '100%',
                                maxWidth: '800px',
                                backgroundColor: 'background.default'
                            }}
                        >
                            <img 
                                src={`data:image/png;base64,${data.plot}`} 
                                alt="Cross-Validation Split" 
                                style={{ 
                                    width: '100%', 
                                    height: 'auto',
                                    borderRadius: '4px'
                                }} 
                            />
                        </Paper>
                        <Box sx={{ 
                            display: 'flex',
                            flexDirection: 'column',
                            gap: 1,
                            width: '100%',
                            maxWidth: '800px'
                        }}>
                            <Typography variant="subtitle2" color="primary">
                                Key Information:
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                                • Number of Folds: {data.k}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                                • Strategy: {data.strategy === 'stratified' ? 'Stratified K-Fold' : 'Standard K-Fold'}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                                • Total Samples: {data.n_samples}
                            </Typography>
                            {data.strategy === 'stratified' && (
                                <Typography variant="body2" color="text.secondary">
                                    • Class Distribution: Preserved across all folds
                                </Typography>
                            )}
                        </Box>
                    </Box>
                )}
            </Paper>
        </Box>
    );
};

export default Visualizations; 
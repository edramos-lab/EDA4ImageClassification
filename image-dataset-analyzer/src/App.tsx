import React, { useState } from 'react';
import { 
  Container, 
  Box, 
  Typography,
  ThemeProvider,
  createTheme,
  CssBaseline,
  CircularProgress,
  IconButton,
  Paper
} from '@mui/material';
import GitHubIcon from '@mui/icons-material/GitHub';
import YouTubeIcon from '@mui/icons-material/YouTube';
import FileUpload from './components/FileUpload';
import DatasetMetrics from './components/DatasetMetrics';
import Visualizations from './components/Visualizations';
import FontSizeControls from './components/FontSizeControls';
import { uploadFile, regeneratePlots, FontSizes } from './services/api';
import { CrossValidationResponse, DimensionalityReductionResponse, Metrics, ImageRecord, Plots, RGBHistogramInsights, AnalysisResponse } from './types';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

const App: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [analysisData, setAnalysisData] = useState<AnalysisResponse | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [crossValidationData, setCrossValidationData] = useState<CrossValidationResponse | null>(null);
  const [dimensionalityReductionData, setDimensionalityReductionData] = useState<DimensionalityReductionResponse | null>(null);
  const [currentFile, setCurrentFile] = useState<File | null>(null);
  const [fontSizes, setFontSizes] = useState<FontSizes>({
    title: 14,
    label: 12,
    pie_labels: 18,
    pie_title: 18,
    sample_title: 26,
    sample_subtitle: 26,
    rgb_title: 14,
    rgb_axis: 12
  });

  const calculateMetrics = (data: ImageRecord[]): Metrics => {
    const classDistribution: Record<string, number> = {};
    const sizeDistribution: Record<string, number> = {};
    let totalSizes = 0;
    let totalWidth = 0;
    let totalHeight = 0;
    let validImages = 0;
    let invalidImages = 0;
    const invalidPaths: string[] = [];

    // Calculate class distribution and size metrics
    data.forEach(record => {
      // Class distribution
      classDistribution[record.class] = (classDistribution[record.class] || 0) + 1;

      // Size metrics
      if (record.width && record.height) {
        const size = record.width * record.height;
        const sizeKey = `${record.width}x${record.height}`;
        sizeDistribution[sizeKey] = (sizeDistribution[sizeKey] || 0) + 1;
        totalSizes += size;
        totalWidth += record.width;
        totalHeight += record.height;
        validImages++;
      } else {
        invalidImages++;
        invalidPaths.push(record.path);
      }
    });

    // Calculate class distribution percentages
    const totalImages = data.length;
    Object.keys(classDistribution).forEach(key => {
        classDistribution[key] = (classDistribution[key] / totalImages) * 100;
    });

    // Calculate size distribution percentages
    Object.keys(sizeDistribution).forEach(key => {
      sizeDistribution[key] = (sizeDistribution[key] / totalImages) * 100;
    });

    // Calculate min/max samples per class
    const classCounts = Object.values(classDistribution).map(count => (count * totalImages) / 100);
    const maxSamples = Math.max(...classCounts);
    const minSamples = Math.min(...classCounts);
    const maxClass = Object.keys(classDistribution).find(key => 
      classDistribution[key] === Math.max(...Object.values(classDistribution))
    ) || '';
    const minClass = Object.keys(classDistribution).find(key => 
      classDistribution[key] === Math.min(...Object.values(classDistribution))
    ) || '';

    // Calculate imbalance ratio
    const ir = maxSamples / minSamples;

    // Calculate balancing efficiency
    const numClasses = Object.keys(classDistribution).length;
    const idealCount = totalImages / numClasses;
    // BE = (1 - (1/C) * Σ|n_i - n̄|/n̄) × 100
    const sumAbsDeviations = classCounts.reduce((sum, count) => sum + Math.abs(count - idealCount), 0);
    const be = Math.floor((1 - (1 / numClasses) * (sumAbsDeviations / idealCount)) * 100 * 10000) / 10000;

    // Calculate entropy balance
    const classProbs = classCounts.map(count => count / totalImages);
    const entropy = -classProbs.reduce((sum, prob) => sum + prob * Math.log2(prob), 0);
    const maxEntropy = Math.log2(numClasses);
    const eb = Math.floor((entropy / maxEntropy) * 100 * 10000) / 10000;

    // Calculate min/max dimensions
    const validRecords = data.filter(record => record.width && record.height);
    const minWidth = Math.min(...validRecords.map(r => r.width!));
    const minHeight = Math.min(...validRecords.map(r => r.height!));
    const maxWidth = Math.max(...validRecords.map(r => r.width!));
    const maxHeight = Math.max(...validRecords.map(r => r.height!));

    // Calculate average dimensions
    const avgWidth = totalWidth / validImages;
    const avgHeight = totalHeight / validImages;

    // Calculate additional metrics
    const avgSamplesPerClass = totalImages / numClasses;
    const stdDev = Math.sqrt(
      classCounts.reduce((sum, count) => sum + Math.pow(count - avgSamplesPerClass, 2), 0) / numClasses
    );
    const cv = (stdDev / avgSamplesPerClass) * 100;

    // Calculate Gini coefficient
    const sortedCounts = [...classCounts].sort((a, b) => a - b);
    const n = sortedCounts.length;
    const index = Array.from({ length: n }, (_, i) => i + 1);
    const gini = ((2 * sortedCounts.reduce((sum, count, i) => sum + index[i] * count, 0)) / 
                  (n * sortedCounts.reduce((sum, count) => sum + count, 0)) - (n + 1) / n) * 100;

    return {
      class_distribution: classDistribution,
      total_sizes: totalSizes,
      avg_width: avgWidth,
      avg_height: avgHeight,
      size_distribution: sizeDistribution,
      max_samples: maxSamples,
      min_samples: minSamples,
      max_class: maxClass,
      min_class: minClass,
      imbalance_ratio: ir,
      balancing_efficiency: be,
      entropy_balance: eb,
      dataset_size: totalImages,
      min_width: minWidth,
      min_height: minHeight,
      max_width: maxWidth,
      max_height: maxHeight,
      valid_images: validImages,
      invalid_images: invalidImages,
      invalid_paths: invalidPaths,
      num_classes: numClasses,
      avg_samples_per_class: avgSamplesPerClass,
      std_dev: stdDev,
      cv: cv,
      gini: gini,
      total_images: totalImages,
      train_count: 0,
      test_count: 0,
      validation_count: 0,
      other_count: 0
    };
  };

  const handleFileUpload = async (file: File, hasHeaders: boolean): Promise<void> => {
    try {
        setLoading(true);
        setError(null);
        setCurrentFile(file);

        console.log('Starting file upload with:', {
            fileName: file.name,
            fileSize: file.size,
            fileType: file.type,
            hasHeaders
        });

        // Use the uploadFile API function with font sizes
        const data = await uploadFile(file, hasHeaders, 1, fontSizes);

        console.log('Parsed response data:', data);

        if (!data.metrics || !data.image_analysis) {
            console.error('Invalid response format:', data);
            throw new Error('Invalid response format from server');
        }

        setAnalysisData(data);
        setMetrics(data.metrics);
        console.log('File upload completed successfully');
    } catch (error) {
        console.error('Error uploading file:', error);
        const errorMessage = error instanceof Error ? error.message : 'An error occurred during file upload';
        
        // Handle specific error messages
        if (errorMessage.includes("No valid images found")) {
            setError('No valid images found in the dataset. Please check that:\n' +
                '1. The image paths in your CSV file are correct\n' +
                '2. The images exist at the specified paths\n' +
                '3. The images are in a supported format (JPEG, PNG, etc.)');
        } else {
            setError(errorMessage);
        }
    } finally {
        setLoading(false);
    }
  };

  const handleCrossValidation = async (file: File, strategy: 'kfold' | 'stratified', k: number, hasHeaders: boolean, multiplier: number) => {
    try {
        console.log('File details:', {
            name: file.name,
            size: file.size,
            type: file.type,
            hasHeaders,
            multiplier
        });

        const formData = new FormData();
        formData.append('file', file);
        formData.append('strategy', strategy);
        formData.append('k', k.toString());
        formData.append('has_headers', hasHeaders.toString());
        formData.append('multiplier', multiplier.toString());

        console.log('Sending cross-validation request with:', {
            strategy,
            k,
            hasHeaders,
            multiplier
        });

        const response = await fetch('http://localhost:7001/crossval', {
            method: 'POST',
            body: formData
        });

        console.log('Response status:', response.status);
        const responseText = await response.text();
        console.log('Response text:', responseText);

        const data = JSON.parse(responseText);
        console.log('Parsed response data:', data);

        if (!data.plot) {
            throw new Error('No plot data received from server');
        }

        setCrossValidationData(data);
    } catch (error) {
        console.error('Error performing cross-validation:', error);
        throw error;
    }
  };

  const handleDimensionalityReduction = async (
    file: File,
    method: 'pca' | 'tsne',
    hasHeaders: boolean,
    multiplier: number
  ): Promise<void> => {
    try {
      setLoading(true);
      setError(null);

      const formData = new FormData();
      formData.append('file', file);
      formData.append('method', method);
      formData.append('has_headers', hasHeaders.toString());
      formData.append('multiplier', multiplier.toString());

      console.log('Sending dimensionality reduction request with:', {
        method,
        hasHeaders,
        multiplier
      });

      const response = await fetch('http://localhost:7001/dimred', {
        method: 'POST',
        body: formData,
      });

      console.log('Response status:', response.status);
      const responseText = await response.text();
      console.log('Response text:', responseText);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}, body: ${responseText}`);
      }

      const data = JSON.parse(responseText);
      console.log('Parsed response data:', data);
      setDimensionalityReductionData(data);
    } catch (error) {
      console.error('Error performing dimensionality reduction:', error);
      throw error;
    } finally {
      setLoading(false);
    }
  };

  const handleFontSizeChange = async (newFontSizes: FontSizes) => {
    setFontSizes(newFontSizes);
    
    // Regenerate plots with new font sizes if we have a current file
    if (currentFile && analysisData) {
      try {
        setLoading(true);
        const result = await regeneratePlots(currentFile, true, 1, newFontSizes);
        setAnalysisData({
          ...analysisData,
          plots: result.plots,
          rgb_histogram_insights: result.rgb_histogram_insights
        });
      } catch (error) {
        console.error('Error regenerating plots:', error);
        setError('Failed to update plot font sizes');
      } finally {
        setLoading(false);
      }
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ 
        minHeight: '100vh', 
        display: 'flex', 
        flexDirection: 'column',
        background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)'
      }}>
        <Container maxWidth="lg" sx={{ py: 4, flex: 1 }}>
          <Typography variant="h4" component="h1" gutterBottom>
            Image Dataset Analyzer
          </Typography>
          <FileUpload 
            onFileUpload={handleFileUpload} 
            loading={loading}
            error={error}
          />
          <FontSizeControls 
            currentFontSizes={fontSizes} 
            onFontSizeChange={handleFontSizeChange}
            disabled={loading}
          />
          {analysisData && metrics && (
            <>
              <DatasetMetrics metrics={metrics} imageRecords={analysisData.image_analysis.records} />
              <Visualizations
                plots={analysisData.plots}
                rgbHistogramInsights={analysisData.rgb_histogram_insights}
                onCrossValidation={handleCrossValidation}
                onDimensionalityReduction={handleDimensionalityReduction}
                crossValidationData={crossValidationData}
                dimensionalityReductionData={dimensionalityReductionData}
                datasetFile={currentFile}
              />
            </>
          )}
        </Container>
        
        {/* Footer */}
        <Paper 
          elevation={3} 
          sx={{ 
            mt: 'auto',
            background: 'linear-gradient(145deg, #1a237e 0%, #283593 100%)',
            color: 'white',
            py: 3
          }}
        >
          <Container maxWidth="lg">
            <Box sx={{ 
              display: 'flex', 
              flexDirection: { xs: 'column', md: 'row' },
              justifyContent: 'space-between', 
              alignItems: 'center',
              gap: 2
            }}>
              <Typography variant="h6" sx={{ fontWeight: 600 }}>
                Connect with the Developer
              </Typography>
              
              <Box sx={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: 3,
                flexWrap: 'wrap',
                justifyContent: 'center'
              }}>
                <Box sx={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: 1,
                  '&:hover': { transform: 'translateY(-2px)' },
                  transition: 'transform 0.2s ease'
                }}>
                  <IconButton
                    href="https://github.com/edramos-lab"
                    target="_blank"
                    rel="noopener noreferrer"
                    sx={{ 
                      color: 'white',
                      '&:hover': { 
                        backgroundColor: 'rgba(255, 255, 255, 0.1)',
                        transform: 'scale(1.1)'
                      },
                      transition: 'all 0.2s ease'
                    }}
                  >
                    <GitHubIcon sx={{ fontSize: 28 }} />
                  </IconButton>
                  <Typography variant="body1" sx={{ fontWeight: 500 }}>
                    @edramos-lab
                  </Typography>
                </Box>
                
                <Box sx={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: 1,
                  '&:hover': { transform: 'translateY(-2px)' },
                  transition: 'transform 0.2s ease'
                }}>
                  <IconButton
                    href="https://youtube.com/@Sigma-AI-mx"
                    target="_blank"
                    rel="noopener noreferrer"
                    sx={{ 
                      color: 'white',
                      '&:hover': { 
                        backgroundColor: 'rgba(255, 255, 255, 0.1)',
                        transform: 'scale(1.1)'
                      },
                      transition: 'all 0.2s ease'
                    }}
                  >
                    <YouTubeIcon sx={{ fontSize: 28 }} />
                  </IconButton>
                  <Typography variant="body1" sx={{ fontWeight: 500 }}>
                    @Sigma-AI-mx
                  </Typography>
                </Box>
              </Box>
            </Box>
            
            <Typography 
              variant="body2" 
              sx={{ 
                textAlign: 'center', 
                mt: 2, 
                opacity: 0.8,
                fontStyle: 'italic'
              }}
            >
              Empowering AI research with comprehensive dataset analysis tools
            </Typography>
          </Container>
        </Paper>
      </Box>
    </ThemeProvider>
  );
};

export default App;

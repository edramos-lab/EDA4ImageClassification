import React, { useState } from 'react';
import {
    Box,
    Paper,
    Typography,
    Slider,
    Button,
    Accordion,
    AccordionSummary,
    AccordionDetails
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { FontSizes } from '../services/api';

interface FontSizeControlsProps {
    onFontSizeChange: (fontSizes: FontSizes) => void;
    currentFontSizes: FontSizes;
    disabled?: boolean;
}

const FontSizeControls: React.FC<FontSizeControlsProps> = ({
    onFontSizeChange,
    currentFontSizes,
    disabled = false
}) => {
    const [localFontSizes, setLocalFontSizes] = useState<FontSizes>(currentFontSizes);

    const handleSliderChange = (key: keyof FontSizes) => (event: Event, newValue: number | number[]) => {
        const newFontSizes = { ...localFontSizes, [key]: newValue as number };
        setLocalFontSizes(newFontSizes);
    };

    const handleInputChange = (key: keyof FontSizes) => (event: React.ChangeEvent<HTMLInputElement>) => {
        const value = parseInt(event.target.value);
        if (!isNaN(value) && value > 0) {
            const newFontSizes = { ...localFontSizes, [key]: value };
            setLocalFontSizes(newFontSizes);
        }
    };

    const handleApply = () => {
        onFontSizeChange(localFontSizes);
    };

    const handleReset = () => {
        const defaultFontSizes: FontSizes = {
            title: 14,
            label: 12,
            pie_labels: 18,
            pie_title: 18,
            sample_title: 26,
            sample_subtitle: 26,
            rgb_title: 14,
            rgb_axis: 12
        };
        setLocalFontSizes(defaultFontSizes);
        onFontSizeChange(defaultFontSizes);
    };

    const fontSizeOptions = [
        { key: 'title', label: 'Plot Titles', min: 8, max: 32, step: 1 },
        { key: 'label', label: 'Axis Labels & General Text', min: 8, max: 24, step: 1 },
        { key: 'pie_labels', label: 'Pie Chart Labels', min: 8, max: 32, step: 1 },
        { key: 'pie_title', label: 'Pie Chart Title', min: 8, max: 32, step: 1 },
        { key: 'sample_title', label: 'Sample Images Title', min: 8, max: 40, step: 1 },
        { key: 'sample_subtitle', label: 'Sample Image Subtitles', min: 8, max: 40, step: 1 },
        { key: 'rgb_title', label: 'RGB Histogram Title', min: 8, max: 32, step: 1 },
        { key: 'rgb_axis', label: 'RGB Histogram Axis', min: 8, max: 24, step: 1 }
    ];

    return (
        <Paper 
            elevation={2} 
            sx={{ 
                p: 3, 
                mb: 3,
                background: 'linear-gradient(145deg, #f8f9fa 0%, #e9ecef 100%)',
                borderRadius: 2
            }}
        >
            <Typography 
                variant="h6" 
                gutterBottom 
                sx={{ 
                    fontWeight: 600,
                    color: '#1a1a1a',
                    borderBottom: '2px solid #dee2e6',
                    pb: 1,
                    mb: 2
                }}
            >
                📝 Font Size Controls
            </Typography>

            <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="subtitle1" fontWeight={500}>
                        Adjust Text Sizes
                    </Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
                        {fontSizeOptions.map((option) => (
                            <Box key={option.key} sx={{ flex: '1 1 300px', minWidth: 0 }}>
                                <Box sx={{ mb: 2 }}>
                                    <Typography variant="body2" gutterBottom fontWeight={500}>
                                        {option.label}
                                    </Typography>
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                                        <Slider
                                            value={localFontSizes[option.key as keyof FontSizes] || 12}
                                            onChange={handleSliderChange(option.key as keyof FontSizes)}
                                            min={option.min}
                                            max={option.max}
                                            step={option.step}
                                            disabled={disabled}
                                            sx={{
                                                flex: 1,
                                                '& .MuiSlider-thumb': {
                                                    backgroundColor: '#1976d2',
                                                },
                                                '& .MuiSlider-track': {
                                                    backgroundColor: '#1976d2',
                                                },
                                                '& .MuiSlider-rail': {
                                                    backgroundColor: '#bdbdbd',
                                                }
                                            }}
                                        />
                                        <Typography 
                                            variant="body2" 
                                            sx={{ 
                                                minWidth: '40px',
                                                textAlign: 'center',
                                                fontWeight: 600,
                                                color: '#1976d2'
                                            }}
                                        >
                                            {localFontSizes[option.key as keyof FontSizes] || 12}
                                        </Typography>
                                    </Box>
                                </Box>
                            </Box>
                        ))}
                    </Box>

                    <Box sx={{ display: 'flex', gap: 2, mt: 3, justifyContent: 'center' }}>
                        <Button
                            variant="contained"
                            onClick={handleApply}
                            disabled={disabled}
                            sx={{
                                background: 'linear-gradient(45deg, #1976d2 30%, #42a5f5 90%)',
                                color: 'white',
                                fontWeight: 600,
                                '&:hover': {
                                    background: 'linear-gradient(45deg, #1565c0 30%, #1976d2 90%)',
                                }
                            }}
                        >
                            Apply Changes
                        </Button>
                        <Button
                            variant="outlined"
                            onClick={handleReset}
                            disabled={disabled}
                            sx={{
                                borderColor: '#1976d2',
                                color: '#1976d2',
                                fontWeight: 600,
                                '&:hover': {
                                    borderColor: '#1565c0',
                                    backgroundColor: 'rgba(25, 118, 210, 0.04)',
                                }
                            }}
                        >
                            Reset to Defaults
                        </Button>
                    </Box>
                </AccordionDetails>
            </Accordion>
        </Paper>
    );
};

export default FontSizeControls; 
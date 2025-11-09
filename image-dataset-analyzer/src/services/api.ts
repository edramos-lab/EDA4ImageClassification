import axios from 'axios';
import { AnalysisResponse, CrossValidationResponse, DimensionalityReductionResponse } from '../types';

const API_BASE_URL = 'http://localhost:7001';

export interface FontSizes {
    title?: number;
    label?: number;
    pie_labels?: number;
    pie_title?: number;
    sample_title?: number;
    sample_subtitle?: number;
    rgb_title?: number;
    rgb_axis?: number;
}

export const uploadFile = async (
    file: File,
    hasHeaders: boolean,
    multiplier: number,
    fontSizes?: FontSizes
): Promise<AnalysisResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('has_headers', String(hasHeaders));
    formData.append('multiplier', String(multiplier));

    // Add font size parameters if provided
    if (fontSizes) {
        Object.entries(fontSizes).forEach(([key, value]) => {
            if (value !== undefined) {
                formData.append(`font_size_${key}`, String(value));
            }
        });
    }

    const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });

    return response.data;
};

export const performCrossValidation = async (
    file: File,
    strategy: 'kfold' | 'stratified',
    k: number,
    hasHeaders: boolean,
    multiplier: number
): Promise<CrossValidationResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('strategy', strategy);
    formData.append('k', String(k));
    formData.append('has_headers', String(hasHeaders));
    formData.append('multiplier', String(multiplier));

    const response = await axios.post(`${API_BASE_URL}/crossval`, formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });

    return response.data;
};

export const performDimensionalityReduction = async (
    file: File,
    method: 'pca' | 'tsne',
    hasHeaders: boolean,
    multiplier: number
): Promise<DimensionalityReductionResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('method', method);
    formData.append('has_headers', String(hasHeaders));
    formData.append('multiplier', String(multiplier));

    const response = await axios.post(`${API_BASE_URL}/dimred`, formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });

    return response.data;
};

export const regeneratePlots = async (
    file: File,
    hasHeaders: boolean,
    multiplier: number,
    fontSizes: FontSizes
): Promise<{ plots: any; rgb_histogram_insights: any }> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('has_headers', String(hasHeaders));
    formData.append('multiplier', String(multiplier));

    // Add font size parameters
    Object.entries(fontSizes).forEach(([key, value]) => {
        if (value !== undefined) {
            formData.append(`font_size_${key}`, String(value));
        }
    });

    const response = await axios.post(`${API_BASE_URL}/regenerate_plots`, formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });

    return response.data;
}; 
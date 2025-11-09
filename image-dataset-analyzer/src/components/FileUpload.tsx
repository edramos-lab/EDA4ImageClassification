import React, { useState } from 'react';
import { Box, Button, Typography, Paper, FormControlLabel, Checkbox, CircularProgress } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

interface FileUploadProps {
  onFileUpload: (file: File, hasHeaders: boolean) => void;
  loading?: boolean;
  error?: string | null;
}

const FileUpload: React.FC<FileUploadProps> = ({ onFileUpload, loading = false, error = null }) => {
  const [file, setFile] = useState<File | null>(null);
  const [hasHeaders, setHasHeaders] = useState<boolean>(true);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      // Read the file content
      const reader = new FileReader();
      reader.onload = (e) => {
        const content = e.target?.result as string;
        if (content) {
          // Process the CSV content to strip whitespace
          const lines = content.split('\n');
          const processedLines = lines.map(line => {
            return line.split(',').map(value => value.trim()).join(',');
          });
          
          // Create a new file with processed content
          const processedContent = processedLines.join('\n');
          const processedFile = new File([processedContent], selectedFile.name, { type: selectedFile.type });
          setFile(processedFile);
        }
      };
      reader.readAsText(selectedFile);
    }
  };

  const handleUpload = async () => {
    if (file) {
      onFileUpload(file, hasHeaders);
    }
  };

  return (
    <Paper 
      elevation={3} 
      sx={{ 
        p: 3, 
        display: 'flex', 
        flexDirection: 'column', 
        alignItems: 'center',
        gap: 2
      }}
    >
      <Typography variant="h6" gutterBottom>
        Upload Dataset
      </Typography>
      <Typography variant="body2" color="textSecondary" paragraph>
        Upload a CSV file containing image paths and class labels. The file should have at least two columns:
        <br />
        <br />
        <strong>Required CSV Structure:</strong>
        <br />
        - <strong>path</strong>: Full file path to the image (e.g., /home/user/datasets/images/dog1.jpg)
        <br />
        - <strong>class</strong>: Class label for the image (e.g., dog, cat, bird)
        <br />
        <br />
        <strong>Example CSV Content:</strong>
        <br />
        path,class
        <br />
        /home/user/datasets/images/dog1.jpg,dog
        <br />
        /home/user/datasets/images/cat1.jpg,cat
        <br />
        /home/user/datasets/images/bird1.jpg,bird
      </Typography>
      {error && (
        <Typography 
          color="error" 
          sx={{ 
            mb: 2,
            whiteSpace: 'pre-line',
            textAlign: 'center',
            maxWidth: '600px'
          }}
        >
          {error}
        </Typography>
      )}
      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
        <Button
          variant="contained"
          component="label"
          startIcon={<CloudUploadIcon />}
          disabled={loading}
        >
          Choose File
          <input
            type="file"
            hidden
            accept=".csv"
            onChange={handleFileChange}
            disabled={loading}
          />
        </Button>
        {file && (
          <Typography variant="body2">
            Selected file: {file.name}
          </Typography>
        )}
        <FormControlLabel
          control={
            <Checkbox
              checked={hasHeaders}
              onChange={(e) => setHasHeaders(e.target.checked)}
              disabled={loading}
            />
          }
          label="CSV file has headers"
        />
        <Button
          variant="contained"
          color="primary"
          onClick={handleUpload}
          disabled={!file || loading}
          startIcon={loading ? <CircularProgress size={20} color="inherit" /> : null}
        >
          {loading ? 'Uploading...' : 'Upload & Analyze'}
        </Button>
      </Box>
    </Paper>
  );
};

export default FileUpload; 
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Select from 'react-select';
import styles from "../styles/searchpage.module.css";
import uploadIcon from "../images/icons/medetect.png";
import Button from '@mui/material/Button';
import IconButton from '@mui/material/IconButton';
import DeleteIcon from '@mui/icons-material/Delete';

const medicineOptions = [
  { value: 'Ascozin', label: 'Ascozin' },
  { value: 'Bioflu', label: 'Bioflu' },
  { value: 'Biogesic', label: 'Biogesic' },
  { value: 'Bonamine', label: 'Bonamine' },
  { value: 'Buscopan', label: 'Buscopan' },
  { value: 'DayZinc', label: 'DayZinc' },
  { value: 'Decolgen', label: 'Decolgen' },
  { value: 'Flanax', label: 'Flanax' },
  { value: 'Imodium', label: 'Imodium' },
  { value: 'Lactezin', label: 'Lactezin' },
  { value: 'Lagundi', label: 'Lagundi' },
  { value: 'Midol', label: 'Midol' },
  { value: 'Myra E', label: 'Myra E' },
  { value: 'Neurogen E', label: 'Neurogen E' },
  { value: 'Omeprazole', label: 'Omeprazole' },
  { value: 'Rinityn', label: 'Rinityn' },
  { value: 'Rogin E', label: 'Rogin E' },
  { value: 'Sinecod', label: 'Sinecod' },
  { value: 'Tempra', label: 'Tempra' },
  { value: 'Tuseran Forte', label: 'Tuseran Forte' }
];

const SearchPage = () => {
  const navigate = useNavigate();
  const [input, setInput] = useState('');

  const searchPill = async (filter) => {
    const url = process.env.REACT_APP_MONGODB_URL || 'https://ap-southeast-1.aws.data.mongodb-api.com/app/data-ahmhw/endpoint/data/v1/action/find';
    const apiKey = process.env.REACT_APP_MONGODB_API_KEY || '';
    const requestBody = {
      collection: 'pills_info',
      database: 'medscan_db',
      dataSource: 'MedScan',
      filter,
    };

    console.log('DB Step 1: MongoDB Request - Filter:', filter);
    console.log('DB Step 2: MongoDB Request - Body:', JSON.stringify(requestBody));

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/ejson',
          'Accept': 'application/json',
          'api-key': apiKey,
        },
        body: JSON.stringify(requestBody),
      });

      console.log('DB Step 3: MongoDB Response - Status:', response.status);
      const responseText = await response.text();
      console.log('DB Step 4: MongoDB Response - Raw Text:', responseText);

      if (!response.ok) {
        throw new Error(`MongoDB fetch failed with status ${response.status}: ${responseText}`);
      }

      const data = JSON.parse(responseText);
      console.log('DB Step 5: MongoDB Response - Parsed Data:', data);
      return data;
    } catch (error) {
      console.error('DB Step 6: MongoDB Fetch Error:', error);
      return null;
    }
  };

  const handleChange = (selectedOption) => {
    setInput(selectedOption.value);
    handleSearch(selectedOption.value);
  };

  const handleSearch = async (query) => {
    const pillData = await searchPill({ name: { $regex: query, $options: 'i' } });
    if (pillData && pillData.documents && pillData.documents.length > 0) {
      const matchedPill = pillData.documents[0];
      const pillDetails = {
        name: matchedPill.name,
        quick_summary: matchedPill.quick_summary || 'Not available',
        chemical_composition: matchedPill.chemical_composition || 'Not available',
        uses: matchedPill.uses || 'Not available',
        side_effects: matchedPill.side_effects || 'Not available',
        image_url: matchedPill.image_url || null
      };
      console.log('Manual Search - Navigating with:', pillDetails);
      navigate('/summary-page', { state: { data: pillDetails } });
    } else {
      console.log('Manual Search - No documents returned');
      alert('No data found in database - check CORS settings in MongoDB Atlas');
    }
  };

  const handleSearchImage = async () => {
    const imgView = document.getElementById('img-view');
    const backgroundImage = imgView.style.backgroundImage;

    if (!backgroundImage || backgroundImage === 'none') {
      alert('Please upload an image before searching.');
      return;
    }

    const fileInput = document.getElementById('input-file');
    if (!fileInput.files || fileInput.files.length === 0) {
      alert('No file selected.');
      return;
    }

    const formData = new FormData();
    formData.append('image', fileInput.files[0]);
    console.log('ML Step A: Uploading file:', fileInput.files[0].name);

    try {
      console.log('ML Step B: Sending request to Flask /predict_pill...');
      const predictResponse = await fetch('http://localhost:5000/predict_pill', {
        method: 'POST',
        body: formData
      });

      if (!predictResponse.ok) {
        const errorText = await predictResponse.text();
        console.error('ML Step C: Prediction error response:', errorText);
        throw new Error(`Prediction failed with status ${predictResponse.status}: ${errorText}`);
      }

      const predictData = await predictResponse.json();
      console.log('ML Step D: Prediction result:', predictData);

      if (predictData.error) {
        console.error('ML Step E: Prediction error from server:', predictData.error);
        alert(`Error from server: ${predictData.error}`);
        return;
      }

      if (!predictData.is_pill) {
        console.log('ML Step F: Image is not a pill:', predictData.message);
        alert(predictData.message);
        return;
      }

      const pillName = predictData.predicted_class;
      const confidence = predictData.confidence;
      const binaryConfidence = predictData.binary_confidence;
      console.log('ML Step G: Predicted pill name:', pillName, 'Confidence:', confidence, 'Binary Confidence:', binaryConfidence);

      console.log('DB Step H: Fetching pill details from MongoDB...');
      const pillData = await searchPill({ name: { $regex: pillName, $options: 'i' } });

      let pillDetails = {
        name: pillName,
        quick_summary: 'Not available',
        chemical_composition: 'Not available',
        uses: 'Not available',
        side_effects: 'Not available',
        image_url: null
      };

      if (pillData && pillData.documents && pillData.documents.length > 0) {
        const matchedPill = pillData.documents[0];
        pillDetails = {
          name: matchedPill.name,
          quick_summary: matchedPill.quick_summary || 'Not available',
          chemical_composition: matchedPill.chemical_composition || 'Not available',
          uses: matchedPill.uses || 'Not available',
          side_effects: matchedPill.side_effects || 'Not available',
          image_url: matchedPill.image_url || null
        };
        console.log('DB Step I: Found pill in DB:', pillDetails);
      } else {
        console.log('DB Step J: No pill found in MongoDB for:', pillName);
        alert('Pill not found in database. Showing predicted name only.');
      }

      console.log('Final Step K: Navigating to summary page with data:', pillDetails);
      navigate('/summary-page', {
        state: {
          data: pillDetails
        }
      });
    } catch (error) {
      console.error('Final Step L: Error in handleSearchImage:', error);
      alert(`Failed to predict or fetch pill data: ${error.message}`);
    }
  };

  const uploadImage = (event) => {
    const file = event.target.files[0];
    if (!file) {
      console.error("No file selected.");
      return;
    }
    console.log('Uploaded file:', file.name);
    try {
      const imgLink = URL.createObjectURL(file);
      document.getElementById('img-view').style.backgroundImage = `url(${imgLink})`;
      document.getElementById('img-view').textContent = '';
      document.getElementById('img-view').style.border = 'none';
    } catch (error) {
      console.error("Failed to create object URL:", error);
    }
  };

  const resetImage = () => {
    document.getElementById('input-file').value = '';
    document.getElementById('img-view').style.backgroundImage = 'none';
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const handleDrop = (event) => {
    event.preventDefault();
    document.getElementById('input-file').files = event.dataTransfer.files;
    uploadImage({ target: { files: event.dataTransfer.files } });
  };

  return (
    <div className={styles.searchpage}>
      <div className={styles.searchContainer}>
        <Select
          className={styles.searchContainerWrapper}
          placeholder="Search By Pill Keyword"
          value={medicineOptions.find(option => option.value === input)}
          onChange={handleChange}
          options={medicineOptions}
        />
      </div>

      <div className={styles.upload}>
        <label htmlFor="input-file" className={styles.dropArea} onDrop={handleDrop} onDragOver={handleDragOver}>
          <input type="file" accept="image/*" id="input-file" onChange={uploadImage} hidden />
          <div id="img-view" className={styles.imgView}>
            <img src={uploadIcon} alt="upload icon" className={styles.icon} />
            <p className={styles.uploadtext}>Upload Your Image Here</p>
          </div>
        </label>
        <div className={styles.bin}>
          <IconButton aria-label="delete" size="large" onClick={resetImage}>
            <DeleteIcon fontSize="inherit" />
          </IconButton>
        </div>
      </div>

      <div className={styles.searchButtonSet}>
        <Button variant="contained" size="large" onClick={handleSearchImage}>Image Search</Button>
      </div>
    </div>
  );
};

export default SearchPage;
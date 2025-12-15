import React, { useState } from 'react';
import { useLocation } from 'react-router-dom';
import styles from '../styles/summarypage.module.css';

function SummaryPage() {
  const { state } = useLocation();
  const pillData = state?.data || {};

  const {
    name = 'Unknown Pill',
    quick_summary = 'Not available',
    chemical_composition = 'Not available',
    uses = 'Not available',
    side_effects = 'Not available',
    image_url = null,
  } = pillData;

  const [selectedLanguage, setSelectedLanguage] = useState('si');
  const [translations, setTranslations] = useState({
    quickSummary: '',
    composition: '',
    pillUses: '',
    sideEffects: '',
  });

  const fetchTranslatedDescription = async (language, texts) => {
    try {
      const response = await fetch('http://localhost:5000/translate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          texts: {
            quickSummary: texts.quickSummary,
            composition: texts.composition,
            pillUses: texts.pillUses,
            sideEffects: texts.sideEffects,
          },
          language: language,
        }),
      });

      if (!response.ok) {
        throw new Error(`Translation failed with status ${response.status}`);
      }

      const data = await response.json();
      console.log('Translation response:', data);
      return data.translations || {
        quickSummary: '',
        composition: '',
        pillUses: '',
        sideEffects: '',
      };
    } catch (error) {
      console.error('Error fetching translated description:', error);
      return {
        quickSummary: '',
        composition: '',
        pillUses: '',
        sideEffects: '',
      };
    }
  };

  const handleTranslateClick = async () => {
    const translatedDescriptions = await fetchTranslatedDescription(selectedLanguage, {
      quickSummary: quick_summary,
      composition: chemical_composition,
      pillUses: uses,
      sideEffects: side_effects,
    });
    setTranslations(translatedDescriptions);
  };

  const handleLanguageChange = (e) => {
    setSelectedLanguage(e.target.value);
    setTranslations({ quickSummary: '', composition: '', pillUses: '', sideEffects: '' });
  };

  let pillImage;
  try {
    pillImage = require(`../images/pills/${name}.png`);
  } catch (error) {
    console.warn(`Image for ${name} not found, using fallback.`);
    pillImage = image_url || 'https://via.placeholder.com/150?text=No+Image';
  }

  return (
    <div className={styles['background-image']}>
      <div className={styles['summary-container']}>
        <h1 className={styles.heading}>Here's what we found:</h1>

        <div className={styles.pillImageContainer}>
          <img src={pillImage} alt={name} className={styles.pillImage} />
        </div>

        <div className={styles.medicineTitle}>Medicine Name:</div>
        <div className={styles.medicineName}>"{name}"</div>

        <div className={styles['description-container']}>
          <div className={styles.detailsOverview}>
            <h1 className={styles.detailsHeading}>Quick Summary:</h1>
            <div className={styles.detailsParaContainer}>
              <p className={styles.detailsPara}>{quick_summary}</p>
              {translations.quickSummary && (
                <p className={styles.detailsPara}>{translations.quickSummary}</p>
              )}
            </div>
          </div>

          <div className={styles.detailsOverview}>
            <h1 className={styles.detailsHeading}>Chemical Composition:</h1>
            <div className={styles.detailsParaContainer}>
              <p className={styles.detailsPara}>{chemical_composition}</p>
              {translations.composition && (
                <p className={styles.detailsPara}>{translations.composition}</p>
              )}
            </div>
          </div>

          <div className={styles.detailsOverview}>
            <h1 className={styles.detailsHeading}>Pill Benefits:</h1>
            <div className={styles.detailsParaContainer}>
              <p className={styles.detailsPara}>{uses}</p>
              {translations.pillUses && (
                <p className={styles.detailsPara}>{translations.pillUses}</p>
              )}
            </div>
          </div>

          <div className={styles.detailsOverview}>
            <h1 className={styles.detailsHeading}>Side Effects:</h1>
            <div className={styles.detailsParaContainer}>
              <p className={styles.detailsPara}>{side_effects}</p>
              {translations.sideEffects && (
                <p className={styles.detailsPara}>{translations.sideEffects}</p>
              )}
            </div>
          </div>

          <div className={styles['dropdown-container']}>
            <select
              className={styles['buttonDropdown']}
              value={selectedLanguage}
              onChange={handleLanguageChange}
            >
              <option value="si">Sinhala</option>
              <option value="ta">Tamil</option>
            </select>
            <button
              className={styles['buttonDropdown']}
              onClick={handleTranslateClick}
            >
              Translate
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default SummaryPage;
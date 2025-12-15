import React, { useState, useEffect, useRef } from 'react';
import styles from "../styles/homepage.module.css";
import { useNavigate } from 'react-router-dom';
import stethoscope from "../images/icons/stethoscope.png";
import pills from "../images/icons/pills.png";
import searchIcon from "../images/icons/search.png";
import discordIcon from "../images/icons/discord.png";
import tablet1 from "../images/icons/tablet1.png";
import tablet2 from "../images/icons/tablet2.png";
import tablet3 from "../images/icons/tablet3.png";
import tablet4 from "../images/icons/tablet4.png";
import heart from "../images/icons/heart.png";
import plus from "../images/icons/plus.png";
import circle from "../images/icons/circle.png";

const HomePage = () => {
  const navigate = useNavigate();
  const [isVisible, setIsVisible] = useState(false);
  const [isNavBoxSetVisible, setIsNavBoxSetVisible] = useState(false);
  const hiddenElementRef = useRef(null);
  const navBoxSetRef = useRef(null);

  useEffect(() => {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        setIsVisible(entry.isIntersecting);
      });
    });

    const hiddenElement = hiddenElementRef.current;
    if (hiddenElement) {
      observer.observe(hiddenElement);
    }

    const navBoxSetObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        setIsNavBoxSetVisible(entry.isIntersecting);
      });
    });

    const navBoxSetElement = navBoxSetRef.current;
    if (navBoxSetElement) {
      navBoxSetObserver.observe(navBoxSetElement);
    }

    return () => {
      if (hiddenElement) {
        observer.unobserve(hiddenElement);
      }
      if (navBoxSetElement) {
        navBoxSetObserver.unobserve(navBoxSetElement);
      }
    };
  }, []);

  const handleSearch = () => {
    navigate('/search-page');
  };

  const handleCommunity = () => {
    window.open('https://discord.gg/qxtB8QPW3E', '_blank', 'noopener noreferrer');
  };


  return (
    <div className={styles.homepage}>
      <div className={styles.topset}>
        <div ref={hiddenElementRef} className={`${styles.hidden} ${isVisible ? styles.show : ''} ${styles.topsetleft}`}>
          <h4 className={styles.welcome}> Welcome to </h4>
          <h1 className={styles.medetect}> Medetect </h1>
          <p className={styles.startinfo}> We provide pill information to your fingertips </p>
          <button onClick={handleSearch} className={styles.searchPageButton}>
            <div className={styles.button}>Get Information</div>
          </button>
        </div>
        
        <div className={styles.topsetright}>
            <img className={styles.pillsImg} alt="logo" src={pills} />
        </div>
        <div className={styles.topsetright}>
            <img className={styles.stethoscopeImg} alt="logo" src={stethoscope} />
        </div>
            <div className={styles.tabletset}>
                <img className={styles.tablet1} alt="tablet1" src={tablet1} />
                <img className={styles.tablet2} alt="tablet2" src={tablet2} />
                <img className={styles.tablet3} alt="tablet3" src={tablet3} />
                <img className={styles.tablet4} alt="tablet4" src={tablet4} />
              </div>
      </div>

      <div ref={navBoxSetRef} className={`${styles.navBoxSet} ${isNavBoxSetVisible ? styles.show : ''}`}>
        <div className={styles.navBox} onClick={handleSearch}>
          <img src={heart} alt="" className={styles.smallIcon1}/>
          <img src={plus} alt="" className={styles.smallIcon3}/>
          <img src={searchIcon} alt="Search" className={styles.mainIcon}/>
          <img src={circle} alt="" className={styles.smallIcon4}/>
          <img src={plus} alt="" className={styles.smallIcon2}/>
          <h1 style={{ margin: '15px' }}>Search</h1>
          <p style={{ maxWidth: '200px', margin: 'auto' }}>Unlock the secrets of your prescription - just a click and a glance away.</p>
        </div>
        <div className={styles.navBox} onClick={handleCommunity}>
          <img src={heart} alt="" className={styles.smallIcon1}/>
          <img src={circle} alt="" className={styles.smallIcon3}/>
          <img src={discordIcon} alt="" className={styles.mainIcon}/>
          <img src={plus} alt="" className={styles.smallIcon4}/>
          <img src={heart} alt="" className={styles.smallIcon2}/>
          <h1 style={{ margin: '15px' }}>Community</h1>
          <p style={{ maxWidth: '240px', margin: 'auto' }}>Join the conversation, exchange stories, and share wisdom in our health-conscious hive!</p>
        </div>
      </div>
    </div>
  );
};

export default HomePage;

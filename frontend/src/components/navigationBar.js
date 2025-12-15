import { useCallback } from "react";
import { useNavigate } from "react-router-dom";
import styles from "./navigationbar.module.css";
import medetectLogo from "../images/icons/medetect.png";


const NavigationBar = () => {
  const navigate = useNavigate();

  const onHomeTextClick = useCallback(() => {
    navigate("/");
  }, [navigate]);

  const onSearchTextClick = useCallback(() => {
    navigate("/search-page");
  }, [navigate]);

  return (
    <div className={styles.navigationBar}>
        <div className={styles.logo}>
          <img className={styles.medetectIcon} alt="logo" src={medetectLogo} />
          <b className={styles.medetectTxt}>Medetect</b>
        </div>
          <div className={styles.pages}>
            <div className={styles.pages} onClick={onHomeTextClick}> Home </div>
            <div className={styles.pages} onClick={onSearchTextClick}> Search </div>
            <a className={styles.pages} href="https://discord.gg/qxtB8QPW3E" target="_blank" rel="noopener noreferrer"> Community </a>
          </div>
    </div>
  );
};

export default NavigationBar;

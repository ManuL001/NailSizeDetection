# Nail Size Detection Tool  
**Classifying Nail Sizes via Exoskeleton Data**

---

## Overview  
This repository contains the code and data processing pipeline for classifying nail sizes based on motion data captured using an **OptiTrack Motion-Capturing system**. The project simulates inputs from an exoskeleton to enhance task recognition, focusing on detecting nail sizes through hammering motions. A **Random Forest Classifier** is used to distinguish between **20mm**, **40mm**, and **80mm** nails by analyzing positional data from key anatomical points such as the **hand**, **lower arm**, **upper arm**, and **shoulder**.

---

## Key Features  
- **Automated Detection of Hammering Events**  
  Identifies the start of hammer swings and impact points using motion trajectory analysis.
  
- **Biomechanical Feature Extraction**  
  Calculates elbow angles and hammer speed to improve classification accuracy.
  
- **Machine Learning Classification**  
  Implements a Random Forest model with hyperparameter tuning for optimal performance.
  
- **Data Preprocessing Pipeline**  
  Cleans and formats motion data using **Pandas** and **NumPy** for efficient model training.

---

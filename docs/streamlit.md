# Streamlit Guide

This project uses **YOLO object detection** together with a **Basler industrial camera** to verify the completeness of assembly kits.  
It compares detected parts against a reference (master kit) and marks them as:

- ✅ **Correct parts** (green)
- ❌ **Incorrect / Extra parts** (red)
- ⚠️ **Missing parts** (purple)

## Features

- Real-time object detection with YOLOv8
- Integration with Basler camera via `pypylon`
- Interactive Streamlit interface
- GPU/CPU device selection
- Visualization of missing, correct, and incorrect parts
- Structured output regions: **Inventory**, **Master Kit**, and **Assembly area**

## Code Workflow

1. **Initialize** Basler camera and YOLO model  
2. **Capture images** from camera  
3. **Detect objects** using YOLO  
4. **Sort items** into:
    - Inventory objects  
    - Master kit objects  
    - Assembly objects  
5. **Compare counts**:
    - Mark **missing** parts  
    - Mark **extra** parts  
    - Mark **correct** parts  
6. Display results in **Streamlit dashboard**

## Flowchart

The process flow is summarized below:

![Workflow Diagram](images/ablaufdiagramm.jpg)

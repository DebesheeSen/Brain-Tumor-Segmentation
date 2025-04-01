# **Brain Tumor Segmentation & Extraction using UNet**

## **Project Overview**  
This project focuses on **brain tumor segmentation** using a **UNet-based deep learning model**. The model is trained to detect tumor regions in MRI scans and generate segmentation masks. The final output includes:  
1. **Predicted tumor mask**  
2. **Ground truth mask** (for comparison)  
3. **Extracted tumor region from the original image** (with a white background)  

## **Technologies Used**  
- **Deep Learning Framework**: PyTorch  
- **Model Architecture**: UNet, Attention Unet  
- **Image Processing**: PIL, NumPy, OpenCV, Matplotlib  
- **Dataset**: Brain MRI Tumor Dataset (from Kaggle)  

## **Project Workflow**  
1. **Load Trained UNet Model**  
   - The pre-trained model is loaded from a checkpoint file.  
   - The model is set to **evaluation mode** to perform inference.  

2. **Preprocessing the Input Image**  
   - Convert MRI scans to grayscale.  
   - Resize images to **512×512**.  
   - Normalize and convert them into tensors.  

3. **Model Prediction**  
   - Perform forward pass through UNet.  
   - Apply **sigmoid activation** and thresholding (`> 0.5`) to generate a binary segmentation mask.  

4. **Post-processing & Output Generation**  
   - Convert the predicted mask to a **black & white mask** (Tumor = **Black**, Background = **White**).  
   - **Extract tumor** from the original MRI scan while setting the background to white.  

5. **Performance Metrics**  
   - **IoU (Intersection over Union)**: Measures overlap between predicted and ground truth masks.  
   - **Dice Score**: Measures similarity between predicted and actual segmentation.  

## **Expected Outputs**  
- **Original MRI Image**  
- **Ground Truth Tumor Mask**  
- **Predicted Tumor Mask (Inverted)**  
- **Tumor Cropped Image (White background, Tumor preserved)**  

## **Improvements**  
- Improved segmentation accuracy using **Attention UNet**.

## **Future Enhancements** 
- Train the model with **larger & diverse MRI datasets**.  
- Extend the project to classify tumor **types (e.g., glioma, meningioma, pituitary tumor)**.
- Combine multiple(two or more) unet improvements in one model.

This project provides an **end-to-end solution** for brain tumor segmentation, with practical applications in **medical image analysis** and **computer-aided diagnosis (CAD)**.

Project Title: Kidney stone detection on CT images
Team Members Name: 
1.	Bibek Lamsal           (11711213)
2.	Praharsha Mutyala (11712768) 
3.	Vignesh Pasula        (11603369) 
4.	Hemanth Maddila  (11727971)
Goals and Objectives:
The primary goal of this feature engineering project on kidney stone detection using CT scan images is to enhance the accuracy and reliability of detection while minimizing false positives. Through a comprehensive approach, the project aims to identify and engineer relevant features in CT scans, considering factors such as noise reduction, normalization, and resizing during data preprocessing. The focus is on implementing effective feature extraction techniques to capture patterns, textures, and shapes indicative of kidney stones. Dimensionality reduction and feature selection will be explored to streamline the model's complexity and enhance interpretability. Integration of the engineered features into machine learning models, particularly convolutional neural networks (CNNs) or traditional classifiers, is a key objective. The project will prioritize thorough validation, optimization, and performance evaluation, using metrics like accuracy and sensitivity. Visualization and interpretability of the selected features will be emphasized to provide insights into the decision-making process. The entire feature engineering process will be meticulously documented, culminating in a comprehensive report summarizing methodologies, challenges, findings, and recommendations.
Motivation:
Kidney stones pose a significant health concern globally, necessitating accurate and efficient diagnostic tools for timely intervention. Traditional methods of kidney stone detection in CT scans often face challenges such as false positives and limited interpretability. This project is motivated by the urgent need to enhance the accuracy of detection while minimizing erroneous diagnoses, ultimately improving patient outcomes. By delving into feature engineering, we aim to unlock the potential of advanced image analysis techniques, leveraging insights from medical imaging and machine learning. The project's significance lies in its potential to contribute to a more robust and interpretable kidney stone detection system, providing healthcare professionals with a reliable tool for early and precise identification. Addressing these challenges is not only pivotal for patient well-being but also aligns with the broader goal of advancing medical imaging technologies for more effective clinical practices.
Significance:
The significance of the feature engineering project on kidney stone detection using CT scan images is multi-faceted and holds profound implications for both medical diagnostics and patient care. Firstly, improved accuracy in identifying kidney stones is crucial for timely and precise medical interventions. The reduction of false positives ensures that patients receive appropriate treatment without unnecessary interventions, minimizing stress and potential health risks. Additionally, a more accurate detection system contributes to enhanced resource allocation within healthcare facilities, optimizing workflow and reducing the burden on medical professionals. The project's exploration of advanced image analysis techniques and feature engineering not only addresses current limitations in kidney stone detection but also sets the stage for the development of more sophisticated diagnostic tools. This endeavor aligns with broader efforts in the field of medical imaging, where technological advancements play a pivotal role in advancing healthcare practices. Ultimately, the project's significance lies in its potential to significantly improve diagnostic accuracy, leading to better patient outcomes and contributing to the ongoing evolution of medical imaging technologies.

Objectives:
1. Feature Identification: Conduct a comprehensive literature review and collaborate with domain experts to identify key features in CT scan images indicative of kidney stones.
2. Data Preprocessing: Implement preprocessing steps, including noise reduction, normalization, and resizing, to ensure the data is prepared optimally for feature extraction.
3. Feature Extraction Techniques: Explore and implement advanced feature extraction techniques, such as texture analysis, shape analysis, and intensity-based features, to capture relevant patterns in CT scans.
4. Dimensionality Reduction: Investigate dimensionality reduction methods like Principal Component Analysis (PCA) or t-Distributed Stochastic Neighbor Embedding (t-SNE) to reduce the complexity of the feature space.
5. Feature Selection: Evaluate the importance of each extracted feature and employ feature selection methods, such as Recursive Feature Elimination (RFE) or feature importance scores, to identify the most relevant features.
6. Integration with Machine Learning Models: Integrate the engineered features into machine learning models, such as convolutional neural networks (CNNs) or ensemble classifiers, to build a robust kidney stone detection system.
7. Validation and Optimization: Validate the performance of the feature-engineered model on a separate dataset to ensure generalization and optimize parameters for improved accuracy.
8. Interpretability and Visualization: Develop visualizations and interpretation tools for the selected features to enhance the understanding of the model's decision-making process.
9. Documentation and Reporting: Document the entire feature engineering process, including methodologies, challenges, and findings, and compile a comprehensive report outlining the outcomes and recommendations.
10. Comparison with Baseline Models: Compare the performance of the feature-engineered model with baseline models to quantify the improvement achieved through the feature engineering process.




Features :
Enhancing kidney stone detection from CT scan images involves extracting diverse features to capture patterns indicative of stone presence. Intensity-based features, like mean intensity and distribution metrics, offer insights into brightness and contrast. Texture-based metrics, including Haralick analysis and GLCM, discern finer details. Shape-based features focus on geometrical attributes, aiding discrimination between stones and normal structures. Frequency-based features from Fourier and Wavelet Transforms explore pixel intensity variations. Gradient-based features aid in edge detection. Statistical metrics provide a comprehensive profile. LBP captures local patterns, and spatial features explore relationships and geometric properties. Zernike moments and fractal dimension compactly represent shape and complexity. Integrating these features into a machine learning framework promises heightened accuracy in kidney stone detection, advancing medical imaging diagnostics.
Related work:
1. Automated Kidney Stone Detection in Non-Contrast Computed Tomography Images: This paper presents an automated approach for kidney stone detection in non-contrast CT images. The authors utilize image processing techniques to extract relevant features and employ machine learning algorithms for classification.
2. Automatic Detection of Kidney Stone in Computed Tomography Urography: Focused on CT urography, this work introduces an automatic system for kidney stone detection. The study incorporates advanced image processing methods to identify and classify kidney stones, contributing to improved diagnostic capabilities.
3. Automated Detection of Kidney Stones in CT Images using Shape and Intensity Features: The paper explores the use of both shape and intensity features for automated kidney stone detection in CT images. The combination of these features enhances the accuracy of detection, providing a comprehensive analysis of stone characteristics.
4. A Review on Computer Aided Diagnosis of Kidney Stones in Computed Tomography Images: Offering a comprehensive review, this paper discusses various computer-aided diagnosis approaches for kidney stones in CT images. It provides insights into the evolution of techniques and methodologies in the field.
5. Computer-Aided Detection of Kidney Stones in Non-Contrast CT Images: Focusing on non-contrast CT images, this work proposes a computer-aided detection system for kidney stones. The authors leverage image analysis and pattern recognition techniques to achieve accurate and efficient stone detection.

 
Design of features:
The feature design for the kidney stone detection project centers on extracting discriminative characteristics from CT scan images, distinguishing kidneys with stones from those without. The chosen features encompass diverse aspects of image content and structure. Intensity-based features, such as mean intensity and standard deviation, provide insights into overall brightness and contrast variations. Texture-based features, including Haralick textural features and Local Binary Pattern, capture intricate patterns within the images.
Shape-based descriptors, such as contour features, area, and perimeter, quantify structural characteristics of kidney regions. Frequency-based features, derived from Fourier and Wavelet transforms, analyze frequency components and multi-scale information. Gradient-based features, employing Sobel and Laplacian filters, enhance edges and highlight structural boundaries.
Statistical metrics like skewness, kurtosis, and entropy describe the distribution and randomness of pixel values. Spatial features, including Zernike moments and fractal dimension, emphasize spatial distribution and structural complexity. This comprehensive feature set aims to create a nuanced representation of kidney structures, facilitating the development of a robust machine learning model for accurate kidney stone detection. The integration of these features ensures sensitivity to variations in intensity, texture, and shape, laying the foundation for a reliable and effective kidney stone detection system from CT scan images.
Analysis:

Exploratory Data Analysis:
In this current project, we are using 625 images with kidney stones and 828 images with normal kidneys. Labels 0 and 1 are  given for with and without stones. The code generates pairwise image comparisons by randomly selecting and displaying two grayscale images from the training set. Each subplot exhibits images with corresponding labels, facilitating a qualitative assessment of visual features associated with kidney stones and normal kidneys. This approach aids in understanding the distinguishability of visual cues, contributing valuable insights for feature engineering and model interpretability. The diversity in visual characteristics observed in these comparisons informs the image classification pipeline, enhancing the exploration of patterns indicative of kidney stones.
                              ![image](https://github.com/killerbeelsl/unt_feature_engineering_projects/assets/44836378/fec9ecde-3b84-43b9-a329-6cf5e71abf56)
              
			Fig 1.1 pairwise image comparisons
The below image represents the distribution of the dataset into two classes like 0 and 1 which represents the kidney images with and without stones. The code utilizes a countplot from the Seaborn library to visualize the distribution of classes in the dataset. The plot displays the frequency of each class, distinguishing between kidney stones (1) and normal kidneys (0). A balanced distribution is preferable for effective machine learning model training, ensuring that the algorithm learns from an adequate number of instances for each class. An uneven distribution may lead to biased predictions, emphasizing the significance of balanced datasets. This analysis helps evaluate the dataset's composition, providing insights into potential class imbalances that might influence model performance.                               
                             ![image](https://github.com/killerbeelsl/unt_feature_engineering_projects/assets/44836378/b1d296ef-a190-4b1a-bba2-1e6ce83477d2)
                             
				         Fig. 1.2. class distribution
The code creates a histogram of image sizes within the dataset, specifically focusing on the height dimension. The plot illustrates the distribution of image heights, revealing insights into the variability of image dimensions. A narrow histogram suggests a consistent image size, while a broader distribution indicates diverse sizes. This analysis aids in understanding the dataset's structural characteristics, assisting in preprocessing steps such as resizing for uniformity. It is crucial for model training and ensures compatibility with algorithms that require consistent input dimensions.
 ![image](https://github.com/killerbeelsl/unt_feature_engineering_projects/assets/44836378/369fd788-1c2f-49ed-b629-bf3f1b77fcda)

Fig 1.3. Image size distribution




The provided code conducts exploratory data analysis (EDA) on a dataset containing kidney stone and normal images. It begins by loading and preprocessing grayscale images, normalizing pixel values, and resizing them to a standardized 128x128 size. The distribution of classes (kidney stone or normal) is visualized using a count plot, revealing the balance or imbalance in the dataset. The intensity distribution of six sample images is depicted through histograms, offering insights into pixel intensity variations within each class. 
 ![image](https://github.com/killerbeelsl/unt_feature_engineering_projects/assets/44836378/3196c3fb-e324-491f-9a85-87dfd787a469)

Fig 1.4. Intensity distribution of images
The code conducts texture analysis using Grey Level Co-occurrence Matrix (GLCM) features, extracting contrast, energy, and entropy from kidney stone and normal images. The `calculate_glcm_features` function facilitates this, offering insights into spatial relationships of pixel intensities. The visual representation of original images alongside GLCM features provides a tangible exploration of textural nuances. This analysis lays a foundational understanding for subsequent stages, aiding in the identification of distinctive patterns within the dataset. The emphasis on GLCM features enhances informed decision-making in subsequent tasks such as model development or classification.
                ![image](https://github.com/killerbeelsl/unt_feature_engineering_projects/assets/44836378/541bb37c-727f-4792-aa9a-e8a20620f46d)
     
Fig. 1.5. GLCM features 
The code performs image filtering on a kidney stone dataset using Gaussian filtering and blurring techniques. The expected outcomes are analyzed based on the original grayscale images and the results of applying these filters. The first row displays the raw stone images, and subsequent rows show the effects of Gaussian filtering and blurring. For stone images, the filters are expected to enhance contours and reduce noise, facilitating feature extraction. Normal images may experience subtle noise reduction. The visualizations offer a comparative assessment, crucial for understanding the impact of these filters on dataset characteristics and guiding subsequent analysis or machine learning applications.
      ![image](https://github.com/killerbeelsl/unt_feature_engineering_projects/assets/44836378/1633a05e-6791-4468-a382-391940f89bd5)

    Fig 1.6. Filtering on images
Pixel Intensity Distribution:
The initial visualization displays the pixel intensity distribution of a sample image from the dataset. The blue histogram with a Kernel Density Estimate (KDE) illustrates the frequency of pixel intensities. This aids in understanding the overall brightness variation in the image. The depicted distribution serves as a representative example for further analysis.
 ![image](https://github.com/killerbeelsl/unt_feature_engineering_projects/assets/44836378/b4e136ac-88ec-497e-b71d-57e974038577)

		         Fig 1.7. Pixel intensity distribution 


Histograms of Original and Processed Images:
The code proceeds to generate histograms for both original and processed (Gaussian filtered or blurred) images. The left subplot exhibits the histogram of the original image, while the right subplot presents the histogram of the processed image. The blue histograms represent original images, and the red histograms represent processed images. This side-by-side comparison facilitates the observation of changes in pixel intensity distribution due to the applied filters.
Analysis of Gaussian Filtering:
For each of the three selected images, Gaussian filtering is applied. The resulting histograms illustrate the impact on pixel intensity distribution. Gaussian filtering tends to smooth the image, reducing high-frequency noise. The histograms reveal potential shifts in intensity values, reflecting the filtering effect. The comparative visualizations assist in evaluating the filtering's influence on pixel intensity.
 ![image](https://github.com/killerbeelsl/unt_feature_engineering_projects/assets/44836378/437db018-294d-4312-9266-314061b1b303)

Fig 1.8. Gaussian filtering
Analysis of Blurring:
Similarly, blurring is applied to the selected images, and histograms are generated for comparison. Blurring, a simpler form of image smoothing, aims to reduce fine details. The histograms showcase alterations in intensity distribution post-blurring. Comparing original and blurred histograms aids in assessing the blurring's impact on pixel intensity characteristics.
 ![image](https://github.com/killerbeelsl/unt_feature_engineering_projects/assets/44836378/0021cf3f-3459-4331-9b80-0513bffc3c7d)

Fig. 1.9. Blurring filtering 

Implementation:
Resnet50 model:
Resnet technique was used to train the model. Google drive was mounted into the workspace to allow access of dataset stored in Google drive. Preprocessing was done by ImageDataGenerator and pixel values are normalized from 0 to 1 with rescaling. Loaded the training images and testing images from the kidney stones dataset. A pretrained ResNet50 model was loaded from keras, and the images are inputted into 224,224,3 and is freezed to further avoid any changes during the training process. A new model is created with a global average pooling layer and for binary classification a dense output layer with a sigmoid activation function . Compiled and trained the new model with batch size of 73 with 10 epochs and saved it. To verify the performance of the trained model the data of test images is loaded and ImageDataGenerator for test data is used to preprocess the data. The trained model is then evaluated with test data. The accuracy and loss of the model are printed.
Analysis of the Kidney Stones Detection Model:
1. Data Preprocessing: The code employs the `ImageDataGenerator` from TensorFlow to preprocess and augment training images. The generator is configured to rescale pixel values to the range [0, 1]. The `flow_from_directory` function is used to load training images from the specified directory, with a target size of (224, 224) pixels.
2. Transfer Learning with ResNet50: The pre trained ResNet50 model which is  obtained from the `keras.applications` module, is loaded. This architecture is known for its deep structure and excellent performance on image-related tasks. All layers of the ResNet50 base model are frozen to retain pre-trained weights during subsequent training.
3. Model Architecture: A new model is constructed by adding a Global Average Pooling layer and a Dense layer with a sigmoid activation for binary classification on top of the ResNet50 base. The resulting model is compiled with the Adam optimizer and binary crossentropy loss, suitable for binary classification tasks.
4. Model Training: The model is trained using the `fit` method with the training data generated by the ImageDataGenerator. Training is performed for 10 epochs. The training history is stored in the `history` variable for potential analysis of accuracy and loss trends.
5. Model Saving: After training, the model is saved in the Hierarchical Data Format (HDF5) with the file name "Kidney_Stones_ResNet_model.h5" for future use or deployment.
 
              



Expected output:
The training process should exhibit decreasing training loss and increasing accuracy over the epochs, reflecting the model's learning. The saved model file can be employed for predictions on new kidney stone images. Monitoring the training history (accuracy and loss) can aid in assessing the model's convergence and potential overfitting. Evaluating the model on a separate validation set is recommended for a comprehensive performance analysis.
  

Analysis:
 
Upon loading the test dataset and configuring a data generator, the model undergoes evaluation using the `evaluate` method, which yields the test loss and accuracy. The resulting `test_accuracy` becomes a pivotal metric, representing the model's ability to generalize to previously unseen data. Interpreting the test accuracy is crucial, as a higher value suggests robust generalization, indicating the model's proficiency in correctly classifying images outside its training set. This metric serves as a benchmark for the model's overall performance on new instances. Here, we achieved an accuracy of 0.57 and test loss 0.68 and it depicts it is a good model.
To delve into the specifics of individual predictions, the model utilizes the `predict` method, generating probability scores for each instance. These probabilities are subsequently converted into binary predictions using a threshold of 0.5. Sample predictions are then presented, providing a direct comparison between the model's predictions and the actual classes of the test images. The analysis of sample predictions offers a nuanced view of the model's performance on a case-by-case basis, uncovering potential strengths and weaknesses. Examining instances where the model deviates from actual classes can provide insights into areas for improvement or highlight specific challenges in the classification task.
In practical applications, accuracy and individual predictions play a pivotal role in determining the reliability of the model for real-world scenarios. Furthermore, more advanced evaluation techniques, such as confusion matrices or precision-recall curves, could be employed for a deeper understanding of the model's performance across different classes. Overall, the provided code and its subsequent analysis contribute to a comprehensive assessment of the CNN model's efficacy in handling new and previously unseen data. 
CNN model:
1. Model Architecture: The model consists of convolutional layers with increasing filter sizes, followed by max-pooling layers for feature extraction. Additional convolutional layers increase the model's capacity to capture hierarchical features. Dense layers at the end are responsible for classification based on learned features.
2. Activation Functions: ReLU activation is used throughout the convolutional layers, introducing non-linearity to the model, and aiding in feature learning. The final dense layer uses a sigmoid activation, suitable for binary classification, producing a probability output.
3. Pooling Layers: Max pooling reduces spatial dimensions, retaining dominant features, and promoting translation invariance.
5. Adjustability: The model's architecture can be modified based on the specific requirements of the image classification task. Hyperparameters like filter sizes, the number of filters, and units in dense layers can be adjusted for optimal performance.
6. Training and Evaluation: The model needs to be trained on labeled data using an appropriate loss function, optimizer, and evaluation metrics. During training, the model learns to minimize the chosen loss, and evaluation on a separate dataset assesses its generalization ability. Model performance can be improved by tuning hyperparameters, increasing dataset size, and utilizing techniques like data augmentation.
                 
                

Expected Output: The model is designed for binary classification tasks, possibly distinguishing between kidney stones (Class 1) and normal cases (Class 0). Training on appropriate image datasets, the model should learn to extract features indicative of the presence or absence of kidney stones.
Here we achieved, test accuracy of 0.765 and test loss of 0.512 which depicts the model is good.

                   






Analysis:
 
The provided code is designed to assess the performance of a pre-trained Convolutional Neural Network (CNN) model in the context of kidney stone detection. The test data is loaded using the `flow_from_directory` method, which not only organizes the data but also resizes images to a predefined target size for consistency. The `ImageDataGenerator` is applied for data normalization, ensuring uniform pixel values across the dataset. The subsequent step involves evaluating the pre-trained CNN model on the loaded test data using the `evaluate` method. This process yields key metrics such as test accuracy and test loss. The printed results provide insights into how well the model generalizes to unseen data.
In terms of expected outcomes, a high-test accuracy suggests that the model performs well in correctly classifying kidney stone images. A low-test loss indicates that the predictions align closely with the actual class labels. These metrics are essential for gauging the model's effectiveness in real-world scenarios.
Monitoring these metrics over time is crucial for maintaining the model's reliability and detecting any potential issues, such as overfitting or underfitting, which might affect its performance on new, unseen data. Overall, this evaluation step is fundamental for ensuring the robustness and applicability of the model in practical settings. 
Project management:
Implementation status report:
In this phase 1, we have achieved building 2 models and got good accuracy of the models but not as expected. We have used resnet50 and CNN model to build good accurate models, but the dataset is comprising of all images and the run time is too long. Here we used limited resources and inputs to get the model running and generate accuracy.
Work completed:
Name	Description 	Responsibility 	Contribution 
Bibek Lamsal           	Worked on CNN model	Improving the model performance with better accuracy	25%
Praharsha Mutyala	Worked on resnet50 model	Accuracy of the model to be improved with full dataset	25%
Vignesh Pasula        	Worked on EDA and analysis	Exploratory Data Analysis tasks and the analysis of each filter used in this project	25%
Hemanth Maddila   	Worked on report and analysis	Report analysis	25%

Work to be completed :
Name	Description 	Responsibility 	Issues/concerns 
Bibek Lamsal           	Re-work on CNN model and try to build svm model	Improving the model performance with better accuracy and working on svm model building	Images dataset is huge size and using as many as possible
Praharsha Mutyala	Re-work on resnet50 model and try building randomforest model for better results	Accuracy of the model to be improved with full dataset and rework on it to achieve better accuracy	Improving the accuracy with more images
Vignesh Pasula        	Try with various more filters and implement them	Exploratory Data Analysis tasks and the analysis of each filter used in this project and explore more filters to analysis in details	Research various papers for more preprocessing methods
Hemanth Maddila   	Work on further analysis and final draft	Report analysis and making final draft	In detailed analysis for the further models


References :
 

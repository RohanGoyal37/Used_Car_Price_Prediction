# 🚗 Used Car Price Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**An intelligent web application that predicts used car prices with 85-95% accuracy using machine learning**

[Demo](#-demo) • [Features](#-key-features) • [Installation](#-installation) • [Usage](#-usage) • [Model Performance](#-model-performance)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Demo](#-demo)
- [Technologies Used](#-technologies-used)
- [Dataset Information](#-dataset-information)
- [Model Architecture](#-model-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Overview

This project is a comprehensive **machine learning solution** for predicting the selling price of used cars in the Indian market. Built with **Streamlit** and powered by **XGBoost**, the application provides an intuitive interface for both data exploration and real-time price predictions.

The system analyzes over **8,000 used car records** and leverages 12+ features including manufacturing year, kilometers driven, fuel type, engine specifications, and more to deliver accurate price estimates. Whether you're a buyer, seller, or automotive enthusiast, this tool helps make informed decisions backed by data science.

### 🎓 Project Context

Developed during my **Data Analyst Internship at ShapeAI**, this project demonstrates end-to-end machine learning workflow from data preprocessing to model deployment.

---

## ✨ Key Features

### 📊 **Comprehensive Data Exploration**
- Interactive dataset viewer with detailed statistics
- Summary metrics including mean, median, and standard deviation
- Missing value analysis and data shape information
- Column-wise data type insights

### 📈 **Rich Visualizations**
- **Scatter Plots**: Explore relationships between features and price
- **Histograms**: Understand distribution of continuous variables
- **Box Plots**: Identify outliers and data spread
- **Correlation Heatmaps**: Discover feature correlations
- Customizable plot parameters for deep analysis

### 🔮 **Intelligent Price Prediction**
- User-friendly input interface for car specifications
- Real-time predictions using trained XGBoost model
- Model accuracy metrics displayed (R², MAE, MSLE, RMSE)
- Smart warnings for brands with limited training data
- Input validation and error handling

### 🧠 **Robust ML Pipeline**
- Advanced preprocessing with outlier removal (1st-99th percentile)
- Intelligent missing value imputation (median/mode strategy)
- Label encoding for categorical variables
- 5-fold cross-validation for reliable performance estimates
- Model persistence using Joblib

---

## 🎬 Demo

### Home Page
![Home Page](car.jpg)

### Prediction Interface
Enter car details such as brand, year, kilometers driven, fuel type, and specifications to get instant price predictions with confidence metrics.

### Data Visualization
Explore interactive charts that reveal insights about pricing patterns, feature correlations, and market trends.

---

## 🛠 Technologies Used

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.12+ |
| **Web Framework** | Streamlit |
| **Machine Learning** | XGBoost, Scikit-learn |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |

---

## 📊 Dataset Information

**Source**: `Car details v3.csv` (~8,000 records)

### Features

| Feature | Description | Type |
|---------|-------------|------|
| `name` | Car model name (used to extract brand) | Text |
| `year` | Manufacturing year | Numeric |
| `selling_price` | **Target variable** - Price in INR | Numeric |
| `km_driven` | Kilometers driven | Numeric |
| `fuel` | Fuel type (Petrol/Diesel/CNG/LPG/Electric) | Categorical |
| `seller_type` | Individual/Dealer/Trustmark Dealer | Categorical |
| `transmission` | Manual/Automatic | Categorical |
| `owner` | Ownership history | Categorical |
| `mileage` | Fuel efficiency (kmpl or km/kg) | Numeric |
| `engine` | Engine capacity (CC) | Numeric |
| `max_power` | Maximum power (bhp) | Numeric |
| `seats` | Number of seats | Numeric |

### Preprocessing Pipeline

The data undergoes a comprehensive preprocessing pipeline implemented in `prepro.py`:

1. **Brand Extraction**: Extract brand name from car model string
2. **Unit Conversion**: Remove units like CC, bhp, kmpl and convert to numeric values
3. **Missing Values**: Impute using median for numeric features and mode for categorical features
4. **Outlier Removal**: Filter selling price to 1st-99th percentile range for robust predictions
5. **Label Encoding**: Convert categorical features to numeric codes for model compatibility
6. **Feature Selection**: Drop irrelevant columns such as name and torque

---

## 🧠 Model Architecture

### Algorithm: **XGBoost Regressor**

The project uses XGBoost (Extreme Gradient Boosting), a powerful ensemble learning method that builds multiple decision trees sequentially. Each tree corrects the errors of the previous ones, resulting in highly accurate predictions.

### Model Configuration

- **Maximum Depth**: 4 levels to prevent overfitting
- **Number of Estimators**: 1000 trees for robust learning
- **Learning Rate**: 0.1 for stable convergence
- **Booster**: gbtree (tree-based model)
- **Objective**: Regression with squared error loss

### Training Process

1. **Data Preparation**: Preprocessed data loaded from the preprocessing pipeline
2. **Model Training**: XGBoost trained on the full feature set using `train_and_save_model.py`
3. **Cross-Validation**: 5-fold cross-validation for reliable performance estimation via `crossval_check.py`
4. **Model Persistence**: Trained model serialized and saved as `xgb_model.pkl` using Joblib

### Why XGBoost?

- **High Accuracy**: Gradient boosting delivers superior predictive performance
- **Handles Non-linearity**: Captures complex interactions between features
- **Robust to Outliers**: Less sensitive to extreme values compared to linear models
- **Feature Importance**: Provides interpretability through feature ranking
- **Efficient**: Optimized for speed and memory usage

---

## 📥 Installation

### Prerequisites

- Python 3.12 or higher
- pip package manager
- Basic familiarity with command line

### Quick Start

**Step 1**: Clone the repository to your local machine

**Step 2**: Navigate to the project directory

**Step 3**: Install all required dependencies using pip and the requirements file

**Step 4**: Verify that both the dataset file `Car details v3.csv` and trained model `xgb_model.pkl` are present in the root directory

**Step 5**: If the model file is missing, train it by running the training script `train_and_save_model.py`

---

## 🚀 Usage

### Launch the Application

Start the Streamlit application by running `main.py` with the streamlit command. The app will automatically open in your default web browser at localhost port 8501.

### Navigation Guide

#### 🏠 **Home**
Get an overview of the application, its purpose, and key capabilities. Perfect starting point for new users.

#### 📊 **Data Summary**
- View the complete dataset with all 8,000+ records
- Explore comprehensive summary statistics for each feature
- Check for missing values and data quality issues
- Understand the structure and dimensions of the data

#### 📈 **Visualize Data**
- Create custom scatter plots to analyze price relationships (e.g., Price vs. Mileage, Price vs. Year)
- Generate histograms to understand feature distributions
- Plot box plots to identify outliers and data spread
- View correlation heatmaps to discover relationships between features
- Customize visualization parameters for deeper insights

#### 🔮 **Predict**
1. Select your car's brand from the dropdown menu
2. Enter detailed specifications including year, kilometers driven, fuel type, transmission, seller type, ownership history, mileage, engine capacity, maximum power, and number of seats
3. Click the **"Predict"** button to generate price estimate
4. View the predicted price along with model accuracy metrics
5. Note any warnings for brands with limited training data

#### 👤 **About Me**
Learn about the project contributor and development context

### Model Evaluation

To evaluate model performance with detailed cross-validation metrics, run the `crossval_check.py` script from your terminal. This will display mean R² scores across all folds.

---

## 📊 Model Performance

### Key Metrics

The model's performance is evaluated using multiple industry-standard metrics:

| Metric | Description | Typical Value |
|--------|-------------|---------------|
| **R² Score** | Coefficient of determination - measures how well predictions match actual values | 0.85 - 0.95 |
| **MAE** | Mean Absolute Error - average absolute difference between predictions and actual prices | ₹50,000 - ₹80,000 |
| **RMSE** | Root Mean Squared Error - penalizes larger errors more heavily | ₹80,000 - ₹1,20,000 |
| **MSLE** | Mean Squared Log Error - handles large price variations effectively | 0.02 - 0.05 |

### Cross-Validation Results

- **5-Fold Cross-Validation Mean R²**: Approximately 0.87 - 0.93
- **Consistency**: Low variance across folds indicates robust model performance
- **Reliability**: Model generalizes well to unseen data

The high R² score indicates that the model explains 85-95% of the variance in used car prices, making it highly reliable for real-world predictions.

---

## 📁 Project Structure

```
used-car-price-prediction/
│
├── 📄 main.py                    # Application entry point and navigation
├── 🏠 home.py                    # Home page component
├── 📊 data.py                    # Data summary page
├── 📈 Visualization.py           # Interactive visualization page
├── 🔮 prediction.py              # Price prediction interface
├── 👤 about_me.py                # Contributor information
│
├── 🔧 prepro.py                  # Data preprocessing pipeline
├── 🧠 Model.py                   # Model loading and prediction logic
├── 🎓 train_and_save_model.py   # Model training script
├── ✅ crossval_check.py          # Cross-validation evaluation
│
├── 📊 Car details v3.csv         # Training dataset (~8K records)
├── 🤖 xgb_model.pkl              # Trained XGBoost model
├── 🖼️ car.jpg                    # Home page image
│
├── 📋 requirements.txt           # Python dependencies
├── 📖 README.md                  # Project documentation
└── 📜 LICENSE                    # MIT License
```

---

## 🔮 Future Enhancements

### Planned Features

- [ ] **Real-time Data Integration**: Fetch live market prices for comparison and validation
- [ ] **Deep Learning Models**: Experiment with neural networks for potentially improved accuracy
- [ ] **Advanced Feature Engineering**: Add derived features like car age, depreciation rate, and brand popularity scores
- [ ] **Model Explainability**: Integrate SHAP values for transparent prediction interpretation
- [ ] **Mobile Optimization**: Fully responsive design optimized for mobile devices
- [ ] **User Authentication**: Save prediction history, compare multiple cars, and bookmark favorite searches
- [ ] **REST API Development**: Programmatic access for developers and third-party integrations
- [ ] **Multi-language Support**: Hindi and regional language support for wider accessibility
- [ ] **Price Trend Analysis**: Historical price tracking and future price forecasting
- [ ] **Comparison Tool**: Side-by-side comparison of multiple car models
- [ ] **Market Insights**: Regional price variations and demand trends

---

## 🤝 Contributing

Contributions are welcome and greatly appreciated! Whether you're fixing bugs, improving documentation, or adding new features, your input helps make this project better.

### How to Contribute

1. **Fork** the repository to your GitHub account
2. **Create** a feature branch for your changes
3. **Commit** your changes with clear, descriptive messages
4. **Push** your changes to your forked repository
5. **Open** a Pull Request with a detailed description of your changes

### Areas for Contribution

- **Bug Fixes**: Identify and fix issues in the codebase
- **New Features**: Add visualization options, prediction enhancements, or new pages
- **Model Improvements**: Experiment with different algorithms or hyperparameters
- **Documentation**: Improve README, add code comments, or create tutorials
- **UI/UX Enhancements**: Improve design, layout, and user experience
- **Performance Optimization**: Speed improvements and code refactoring
- **Testing**: Add unit tests and integration tests

### Code of Conduct

Please be respectful and constructive in all interactions. We're building a welcoming community for everyone interested in data science and machine learning.

---

## 📄 License

This project is licensed under the **MIT License**, which means you're free to use, modify, and distribute this software for any purpose, including commercial use. See the [LICENSE](LICENSE) file for complete details.

---

## 👨‍💻 Contact

**Rohan Goyal**

- **Role**: Data Analyst Intern @ ShapeAI
- **GitHub**: [@RohanGoyal37](https://github.com/RohanGoyal37)
- **LinkedIn**: [Rohan Goyal](https://www.linkedin.com/in/rohan-agarwal37)
- **Email**: Available on GitHub profile

Feel free to reach out for:
- Questions about the project
- Collaboration opportunities
- Bug reports or feature requests
- General feedback and suggestions

---

## 🙏 Acknowledgments

- **ShapeAI** for providing the internship opportunity and valuable guidance throughout the project
- **XGBoost Development Team** for creating such a powerful and efficient machine learning library
- **Streamlit** for the intuitive web framework that makes data apps accessible
- **Open Source Community** for countless tools, libraries, and resources that made this project possible
- **Dataset Contributors** for compiling comprehensive used car data

---

## 📈 Project Stats

- **Lines of Code**: 1,500+
- **Data Points**: 8,000+ used car records
- **Features Analyzed**: 12+ variables
- **Model Accuracy**: 85-95% R² score
- **Technologies Used**: 7+ Python libraries

---

<div align="center">

**If you find this project helpful, please consider giving it a ⭐ on GitHub!**

**Stars help others discover the project and motivate continued development**

Made with ❤️ and ☕ by Rohan Goyal

</div>

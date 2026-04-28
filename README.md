# biaslens-ai
AI tool to detect and reduce bias in machine learning models.

# BiasLens AI

BiasLens AI is a prototype system designed to detect, analyze, and reduce bias in machine learning decision-making processes. The goal of this project is to demonstrate how unfair patterns in data can influence AI models, and how simple corrective techniques can improve fairness.


# Problem Statement

Machine learning models often learn from historical data that may contain hidden biases. These biases can lead to unfair outcomes in critical applications such as loan approvals, hiring systems, and insurance decisions.

For example, if past data shows higher approval rates for a particular group, the model may learn to favor that group, even if it is not justified.


# Objective

The objective of this project is to:

- Detect bias in model predictions  
- Quantify the level of unfairness  
- Provide a mechanism to reduce bias  
- Present results in an interpretable and user-friendly manner  


# How the System Works

The system follows a simple pipeline:

1. Load dataset  
2. Train a machine learning model  
3. Generate predictions  
4. Measure bias between groups  
5. Apply a bias mitigation technique  
6. Compare results before and after correction  


# Bias Detection Method

Bias is detected by comparing approval rates between different groups (in this case, based on gender).

Step 1: Model Training

A Logistic Regression model is trained using the dataset features:
- Gender  
- Income  
- Credit Score  

Step 2: Prediction

The model generates predictions (Approved / Not Approved).

Step 3: Group-wise Analysis

The system calculates:

- Male approval rate  
- Female approval rate  

Step 4: Bias Gap Calculation

Bias is quantified using the difference in approval rates:

Bias Gap = | Male Approval Rate - Female Approval Rate |

A larger gap indicates higher bias.


# Why Bias Occurs

Bias arises because:

- The dataset contains imbalanced outcomes  
- Certain groups have historically lower approvals  
- The model learns and replicates these patterns  

Even if bias is not explicitly programmed, it emerges from data.


# Bias Mitigation Technique

To reduce bias, the system applies a simple but effective method:

### Removal of Sensitive Attribute

The "Gender" feature is removed during retraining.

Before:
Model uses → Gender + Income + Credit Score  

After:
Model uses → Income + Credit Score only  

# Effect

- The model can no longer directly use gender in decision-making  
- Decisions become more dependent on relevant financial features  
- Bias gap reduces significantly  


# Important Insight

Bias is reduced but not eliminated completely.

This happens because:

- Other features (like income) may still indirectly reflect bias  
- Data distribution itself is not fully balanced  

This highlights a real-world challenge: removing bias requires more than just removing sensitive attributes.


# Fairness Score

To improve interpretability, the system converts bias gap into a fairness score:

Fairness Score = 100 - Bias Gap

A higher score indicates a more fair model.


# AI-Based Explanation (Design Decision)

The system was designed to include AI-generated explanations using the Gemini API to describe why bias occurs.

However, during implementation, API limitations (quota and availability issues) prevented reliable real-time usage.

# Engineering Decision

To ensure system stability during demonstration:

- A fallback explanation mechanism was implemented  
- The system provides consistent, human-readable explanations based on model behavior  

This approach ensures:
- No runtime failures  
- Clear interpretability  
- Reliable demo performance  


# Features

- Dataset visualization  
- Model prediction display  
- Bias detection using group comparison  
- Bias gap calculation  
- Fairness score visualization  
- Bias mitigation via feature removal  
- Explanation system for interpretability  


# Tech Stack

- Python  
- Streamlit (Frontend + UI)  
- Scikit-learn (Machine Learning)  
- Pandas (Data Processing)  


# Limitations

- Uses a small synthetic dataset  
- Applies a basic fairness technique  
- Does not include advanced fairness metrics  
- Explanation system uses fallback instead of live AI API  


# Future Scope

- Integration with real-world datasets  
- Advanced fairness algorithms (reweighting, adversarial debiasing)  
- Full AI-powered explanation system  
- API-based deployment for enterprise use  
- Continuous bias monitoring over time  


# Conclusion

BiasLens AI demonstrates how bias can be detected and reduced in machine learning systems using simple, interpretable methods. It emphasizes the importance of fairness in AI and highlights the challenges involved in achieving unbiased decision-making.


# Repository Structure

- app.py → Main application  
- data.csv → Dataset used for demonstration  
- README.md → Project documentation  

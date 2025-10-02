# Bias-Detection-in-Machine-Learning-Models
Dataset:
        - German Credit Dataset (UCI) – 1,000 loan applicants.
        - Features: Age, Sex, Job, Savings, Loan Amount, etc.
        - Target: Risk (good/bad).
        - Sensitive attributes: Sex (gender) and Age group.

Methodology:
            - Preprocessed categorical variables with LabelEncoder.
            - Scaled numerical features with StandardScaler.
            - Trained Logistic Regression and Decision Tree models.
            - Evaluated accuracy and fairness metrics:
                                                      - Demographic Parity Difference
                                                      - Equal Opportunity Difference
                                                      - Group-wise selection rates

Insights:
- Accuracy was ~70–75% depending on the model.
- Detected disparities across gender and age groups, showing bias in approval rates.
- Highlighted trade-offs between accuracy and fairness when applying bias mitigation strategies.

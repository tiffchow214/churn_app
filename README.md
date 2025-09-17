Link to Kaggle dataset used to train the logistic regression model: https://www.kaggle.com/datasets/samuelsemaya/e-commerce-customer-churn

Predict which customers are likely to churn, explain why, and choose an actionable decision threshold (Profit/ROI or Custom). Built from a Google Colab prototype and deployed as an interactive app.

Live app: https://huggingface.co/spaces/tiffcz214/ecommerce_churn_app

Demo video:
https://www.loom.com/share/bd6ca6ec9ee34eb1a966b8b1168ee4c3?sid=38898a4f-55ad-45bc-8888-9e268d075a49 

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
](https://www.loom.com/share/bd6ca6ec9ee34eb1a966b8b1168ee4c3?sid=38898a4f-55ad-45bc-8888-9e268d075a49

🚀 What this app does

Single prediction: enter one customer’s attributes → get a churn probability, a flag (act / don’t act), and reasons.

Batch scoring (CSV): score thousands at once → get a scored.csv with probability, decision, and short “why” for every customer.

Model drivers: see the big patterns that increase/decrease churn across the population.

🧠 Model & features

Model: Logistic Regression (simple, fast, interpretable)

Key inputs (features):

tenure_months

delivery_distance_km

num_devices

satisfaction_score (1–5)

num_addresses

days_since_last_order

cashback_amount

complaint (0/1)

preferred_category (e.g., Laptop & Accessory, Mobile Phone, Others)

marital_status (Single/Married/Divorced)

Outputs:

probability (0–1)

decision (flag / don’t flag based on chosen threshold)

why (top factors pushing the score up/down)

🎯 Decision threshold (Recall / Profit / Custom)

The model produces a probability; you choose a cutoff to take action.

Recall – “catch as many churners as possible.” It’s mainly a training/evaluation metric, so it’s not a toggle in the app.

Profit (ROI) – a business-driven default (≈ 0.39 in my runs) that balances the value of saving a customer vs. outreach cost.

Custom – drag the slider to match your current budget or risk appetite.

Rule of thumb: below the threshold → Do not flag; above → Flag (likely churn).

🖱️ How to use the app Single prediction

Pick Decision threshold (start with Profit/ROI).

Set inputs (tenure, distance, satisfaction, etc.).

Click Predict → see probability, decision, and “why”.

Example personas

Low risk: long tenure, high satisfaction, healthy cashback → usually Do not flag.

Likely churn: short tenure, recent complaint, long gap since last order, low cashback → usually Flag with suggested actions.

Batch (CSV)

Go to Batch (CSV) tab.

Upload a file with the columns listed above.

Choose your threshold policy and click Score file.

Download the scored.csv (probability, decision, why) for CRM/playbooks.

📊 Model drivers (population-level “why”)

Risk ↑: “Others” category, mid-tenure (12–24 months), complaints, missing fields, long delivery distance.

Risk ↓: longer tenure, higher cashback, more devices per address, tech categories (e.g., Mobile Phone, Laptop & Accessory).)

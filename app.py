
import gradio as gr
import pandas as pd

def load_drivers_df():
    if DRIVERS_JSON_PATH.exists():
        try:
            df = pd.read_json(DRIVERS_JSON_PATH)
            if "coef" in df.columns:
                df = df.sort_values("coef", ascending=False)
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=["feature","coef","odds_ratio"])

drivers_df = load_drivers_df()

cat_opts = meta.get("ui_options", {})
marital_choices = cat_opts.get("MaritalStatus", ["Married","Single","Divorced"])
cat_choices = cat_opts.get("PreferedOrderCat", ["Laptop & Accessory","Mobile Phone","Fashion","Grocery","Others"])

with gr.Blocks(title="E-commerce Churn Predictor") as demo:
    gr.Markdown("## E-commerce Churn Predictor\nGet churn probability, a decision flag, and **why** (risk flags).")

    with gr.Tab("Single prediction"):
        with gr.Row():
            with gr.Column():
                Tenure = gr.Slider(0, 60, value=6, step=1, label="Tenure (months)")
                WarehouseToHome = gr.Slider(0, 50, value=10, step=1, label="Warehouse → Home (km)")
                NumberOfDeviceRegistered = gr.Slider(1, 6, value=3, step=1, label="# Devices registered")
                SatisfactionScore = gr.Slider(1, 5, value=3, step=1, label="Satisfaction score (1–5)")
                NumberOfAddress = gr.Slider(1, 10, value=2, step=1, label="# Addresses on file")
                DaySinceLastOrder = gr.Slider(0, 90, value=7, step=1, label="Days since last order")
                CashbackAmount = gr.Slider(0, 350, value=150, step=1, label="Cashback amount")
                Complain = gr.Radio(choices=["No","Yes"], value="No", label="Complained?")
                PreferedOrderCat = gr.Dropdown(cat_choices, value=cat_choices[0], label="Preferred order category")
                MaritalStatus = gr.Dropdown(marital_choices, value=marital_choices[0], label="Marital status")

                gr.Markdown("**Decision threshold**")
                threshold_mode   = gr.Radio(["Recall (F2)", "Profit (ROI)", "Custom"], value="Recall (F2)")
                profit_threshold = gr.Slider(0.05, 0.95, step=0.01, value=thr_profit_default, label="Profit threshold")
                custom_threshold = gr.Slider(0.05, 0.95, step=0.01, value=thr_f2, label="Custom threshold")

                btn_single = gr.Button("Predict")

            with gr.Column():
                prob = gr.Textbox(label="Churn probability", interactive=False)
                dec  = gr.Textbox(label="Decision", interactive=False)
                expl = gr.Textbox(label="Why this decision (risk flags)", lines=6, interactive=False)

        btn_single.click(
            predict_fn,
            inputs=[Tenure, WarehouseToHome, NumberOfDeviceRegistered, SatisfactionScore,
                    MaritalStatus, NumberOfAddress, Complain, DaySinceLastOrder,
                    CashbackAmount, PreferedOrderCat, threshold_mode, custom_threshold, profit_threshold],
            outputs=[prob, dec, expl]
        )

    with gr.Tab("Batch (CSV)"):
        gr.Markdown(
            "Upload a CSV with these exact columns:\n\n"
            f"`{', '.join(REQUIRED_COLS)}`\n\n"
            "Tip: `Complain` can be **Yes/No** or **1/0**."
        )
        file_in = gr.File(file_types=[".csv"], label="Upload CSV")
        thr_mode_b = gr.Radio(["Recall (F2)", "Profit (ROI)", "Custom"], value="Recall (F2)", label="Decision threshold")
        profit_thr_b = gr.Slider(0.05, 0.95, step=0.01, value=thr_profit_default, label="Profit threshold")
        custom_thr_b = gr.Slider(0.05, 0.95, step=0.01, value=thr_f2, label="Custom threshold")
        btn_batch = gr.Button("Score file")

        preview_out = gr.Dataframe(label="Preview (first 25 rows)", interactive=False)
        file_out = gr.File(label="Download scored CSV")

        btn_batch.click(
            batch_predict_fn,
            inputs=[file_in, thr_mode_b, custom_thr_b, profit_thr_b],
            outputs=[preview_out, file_out]
        )

    with gr.Tab("Model drivers"):
        gr.Markdown("Top global drivers from a transparent logistic refit (odds ratios shown).")
        gr.Dataframe(value=drivers_df, interactive=False, wrap=True, label="Global drivers")

# Choose a specific port to avoid conflicts in Colab
demo.launch(server_name="0.0.0.0", server_port=7866, share=True, debug=True)

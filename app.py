# app.py
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as mtick
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="BudgetVision", layout="wide")
st.title("BudgetVision: Automated Budget Variance & Forecasting Tool")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    # Read the CSV
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of your data")
    st.write(df.head())

    # Build quarter_date once (expects 'Fiscal Year' and 'Quarter' columns like 'Q1')
    if ('Fiscal Year' in df.columns) and ('Quarter' in df.columns):
        try:
            df['quarter_date'] = pd.PeriodIndex(
                year=df['Fiscal Year'].astype(int),
                quarter=df['Quarter'].astype(str).str.extract(r'(\d)')[0].astype(int),
                freq='Q'
            ).to_timestamp()
        except Exception as e:
            st.error(f"Failed to build quarter_date from Fiscal Year & Quarter: {e}")
            st.stop()
    else:
        st.error("CSV must contain 'Fiscal Year' and 'Quarter' columns for the quarterly analysis.")
        st.stop()




    # Aggregate by quarter (budget)
    fq_budget = df.groupby('quarter_date')['Budget'].sum().reset_index()
    fq_budget_series = fq_budget.set_index('quarter_date')['Budget']

    # Prepare fq_summary used in variance sections
    fq_summary = fq_budget.copy()
    fq_summary['Period'] = fq_summary['quarter_date']
    fq_summary['Fiscal Year'] = fq_summary['quarter_date'].dt.year
    fq_summary['Quarter'] = 'Q' + fq_summary['quarter_date'].dt.quarter.astype(str)

    # Big Dashboard Header
    st.markdown(
        """
        <div style='background:#0b69ff; padding:16px; border-radius:8px; margin-top:20px; margin-bottom:10px;'>
            <h1 style='margin:0; font-size:32px; color:white; text-align:center; letter-spacing:2px;'>
                &nbsp; — &nbsp; ANALYSIS &nbsp; — &nbsp;
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )


    # Create top-level tabs right after preview
    tabs = st.tabs(["Overview", "Seasonality", "Forecasting", "Variance Analysis", "Anomalies"])

    ####################################
    # Overview tab
    ####################################
    with tabs[0]:
        st.subheader("Overview")
        st.write("Quick summary statistics for Budget by quarter:")
        st.dataframe(fq_budget.describe())

        st.subheader("Quarterly Budgets Over Time")
        fig_over, ax_over = plt.subplots(figsize=(14, 5))
        sns.lineplot(data=fq_summary, x='Period', y='Budget', marker='o', linewidth=2, color='royalblue', ax=ax_over)
        ax_over.set_title("Quarterly Budgets Over Time")
        ax_over.set_xlabel("Fiscal Quarter")
        ax_over.set_ylabel("Total Budget")
        # Show fewer x-axis labels (only Q1 of each year)
        q1_labels = fq_summary[fq_summary['Quarter'] == 'Q1']
        try:
            ax_over.set_xticks(q1_labels['Period'])
            ax_over.set_xticklabels(q1_labels['Fiscal Year'], rotation=45)
        except Exception:
            pass
        ax_over.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        plt.tight_layout()
        st.pyplot(fig_over)


    ####################################
    # Seasonality tab
    ####################################
    with tabs[1]:
        st.subheader("Quarterly Seasonality Analysis")

        # Moving Average Smoothing (4-quarter centered)
        fq_budget['budget_ma'] = fq_budget['Budget'].rolling(window=4, center=True).mean()

        # Seasonal factor (watch edges where MA is NaN)
        fq_budget['sf'] = fq_budget['Budget'] / fq_budget['budget_ma']

        st.write("Quarterly Seasonal Factor Summary:")
        st.write(fq_budget['sf'].describe())

        # Trend plot
        fig1, ax1 = plt.subplots(figsize=(14, 6))
        ax1.plot(fq_budget['quarter_date'], fq_budget['Budget'], marker='o', label='Quarterly Budget')
        ax1.plot(fq_budget['quarter_date'], fq_budget['budget_ma'], color='red', linewidth=2, label='4Q Moving Average')
        ax1.set_xlabel("Quarter")
        ax1.set_ylabel("Budget")
        ax1.set_title("Quarterly Budget with Trend (Seasonality Check)")
        ax1.legend()
        st.pyplot(fig1)

        # Boxplot of seasonal factors
        fq_budget['Quarter_Num'] = fq_budget['quarter_date'].dt.quarter
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        box_data = [fq_budget[fq_budget['Quarter_Num']==q]['sf'].dropna() for q in [1,2,3,4]]
        ax2.boxplot(box_data, labels=['Q1','Q2','Q3','Q4'])
        ax2.set_title("Seasonal Factors by Quarter")
        ax2.set_xlabel("Quarter")
        ax2.set_ylabel("Seasonal Factor")
        st.pyplot(fig2)

    ####################################
    # Forecasting tab
    ####################################

    with tabs[2]:
        st.markdown("<h2 style='font-size:26px; margin-bottom:6px'>Forecasting</h2>", unsafe_allow_html=True)

        # ensure series is ready
        fq_budget_series = fq_budget.set_index('quarter_date')['Budget'] if 'quarter_date' in fq_budget.columns else fq_budget
        series = fq_budget_series.dropna().astype(float)
        if series.shape[0] < 6:
            st.info("Not enough data for robust forecasting (need at least ~6 quarters). Some models may fail or be skipped.")
        
        def safe_mape(y_true, y_pred):
            y_true = pd.to_numeric(y_true, errors='coerce')
            y_pred = pd.to_numeric(y_pred, errors='coerce')
            denom = y_true.replace(0, np.nan)
            return np.mean(np.abs((y_true - y_pred) / denom)) * 100

        # Helper to compute pseudo (last 4 quarters) metrics
        def pseudo_metrics(full_series, fit_fn, forecast_steps=4):
            if full_series.shape[0] <= forecast_steps + 1:
                return (np.nan, np.nan)  # not enough data
            train = full_series[:-forecast_steps]
            try:
                model_train = fit_fn(train)
                pf = model_train.forecast(forecast_steps)
                pf.index = full_series[-forecast_steps:].index
                rmse_pf = np.sqrt(mean_squared_error(full_series[-forecast_steps:], pf))
                mape_pf = safe_mape(full_series[-forecast_steps:], pf)
                return (rmse_pf, mape_pf)
            except Exception:
                return (np.nan, np.nan)

        # ---------- Exponential Smoothing ----------
        st.subheader("Exponential Smoothing (Additive Seasonality)")
        try:
            exp_model = ExponentialSmoothing(series, seasonal='add', seasonal_periods=4)
            exp_fit = exp_model.fit()
            exp_forecast_next4 = exp_fit.forecast(4)

            exp_rmse_in = np.sqrt(mean_squared_error(series, exp_fit.fittedvalues))
            exp_mape_in = safe_mape(series, exp_fit.fittedvalues)

            exp_rmse_pseudo, exp_mape_pseudo = pseudo_metrics(series, 
                                                              lambda tr: ExponentialSmoothing(tr, seasonal='add', seasonal_periods=4).fit())

            st.write(f"**In-sample RMSE:** {exp_rmse_in:.2f}   |   **In-sample MAPE (%):** {exp_mape_in:.2f}")
            st.write(f"**Pseudo RMSE (last 4 q):** {exp_rmse_pseudo if not np.isnan(exp_rmse_pseudo) else 'N/A'}   |   **Pseudo MAPE (%):** {exp_mape_pseudo if not np.isnan(exp_mape_pseudo) else 'N/A'}")

            fig4, ax4 = plt.subplots(figsize=(14,6))
            ax4.plot(series.index, series, marker='o', label='Actual')
            ax4.plot(series.index, exp_fit.fittedvalues, color='red', label='Fitted (ExpSmoothing)')
            ax4.plot(exp_forecast_next4.index, exp_forecast_next4, marker='o', linestyle='--', color='green', label='Forecast Next 4 Q')
            ax4.set_title('Exponential Smoothing Forecast')
            ax4.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
            ax4.legend()
            plt.tight_layout()
            st.pyplot(fig4)
        except Exception as e:
            st.warning(f"Exponential Smoothing failed: {e}")

        st.markdown("---")

        # ---------- Holt's Linear ----------
        st.subheader("Holt's Linear Method")
        try:
            holt_model = Holt(series, exponential=False, damped_trend=False).fit()
            holt_fitted = holt_model.fittedvalues
            holt_forecast_next4 = holt_model.forecast(4)

            holt_rmse_in = np.sqrt(mean_squared_error(series, holt_fitted))
            holt_mape_in = safe_mape(series, holt_fitted)

            holt_rmse_pseudo, holt_mape_pseudo = pseudo_metrics(series,
                                                                lambda tr: Holt(tr, exponential=False, damped_trend=False).fit())

            st.write(f"**In-sample RMSE:** {holt_rmse_in:.2f}   |   **In-sample MAPE (%):** {holt_mape_in:.2f}")
            st.write(f"**Pseudo RMSE (last 4 q):** {holt_rmse_pseudo if not np.isnan(holt_rmse_pseudo) else 'N/A'}   |   **Pseudo MAPE (%):** {holt_mape_pseudo if not np.isnan(holt_mape_pseudo) else 'N/A'}")

            fig5, ax5 = plt.subplots(figsize=(14,6))
            ax5.plot(series.index, series, marker='o', label='Actual')
            ax5.plot(series.index, holt_fitted, color='red', label="Fitted (Holt)")
            ax5.plot(holt_forecast_next4.index, holt_forecast_next4, marker='o', linestyle='--', color='green', label='Forecast Next 4 Q')
            ax5.set_title("Holt's Method Forecast")
            ax5.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
            ax5.legend()
            plt.tight_layout()
            st.pyplot(fig5)
        except Exception as e:
            st.warning(f"Holt's method failed: {e}")

        st.markdown("---")

        # ---------- Holt-Winters ----------
        st.subheader("Holt-Winters (Additive Trend & Seasonality)")
        try:
            hw_model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=4).fit()
            hw_fitted = hw_model.fittedvalues
            hw_forecast_next4 = hw_model.forecast(4)

            hw_rmse_in = np.sqrt(mean_squared_error(series, hw_fitted))
            hw_mape_in = safe_mape(series, hw_fitted)

            hw_rmse_pseudo, hw_mape_pseudo = pseudo_metrics(series,
                                                            lambda tr: ExponentialSmoothing(tr, trend='add', seasonal='add', seasonal_periods=4).fit())

            st.write(f"**In-sample RMSE:** {hw_rmse_in:.2f}   |   **In-sample MAPE (%):** {hw_mape_in:.2f}")
            st.write(f"**Pseudo RMSE (last 4 q):** {hw_rmse_pseudo if not np.isnan(hw_rmse_pseudo) else 'N/A'}   |   **Pseudo MAPE (%):** {hw_mape_pseudo if not np.isnan(hw_mape_pseudo) else 'N/A'}")

            fig6, ax6 = plt.subplots(figsize=(14,6))
            ax6.plot(series.index, series, marker='o', label='Actual')
            ax6.plot(series.index, hw_fitted, color='red', label="Fitted (Winters)")
            ax6.plot(hw_forecast_next4.index, hw_forecast_next4, marker='o', linestyle='--', color='green', label='Forecast Next 4 Q')
            ax6.set_title("Holt-Winters Forecast")
            ax6.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
            ax6.legend()
            plt.tight_layout()
            st.pyplot(fig6)
        except Exception as e:
            st.warning(f"Holt-Winters failed: {e}")

        st.markdown("---")

        # ---------- SARIMA (guarded) ----------
        st.subheader("SARIMA (seasonal ARIMA) — guarded")
        try:
            if series.shape[0] >= 12:
                sarima_model = SARIMAX(series,
                                       order=(1,1,1),
                                       seasonal_order=(1,1,1,4),
                                       enforce_stationarity=False,
                                       enforce_invertibility=False).fit(disp=False)
                sarima_fitted = sarima_model.fittedvalues
                sarima_forecast_next4 = sarima_model.get_forecast(steps=4).predicted_mean

                sarima_rmse_in = np.sqrt(mean_squared_error(series, sarima_fitted))
                sarima_mape_in = safe_mape(series, sarima_fitted)

                sarima_rmse_pseudo, sarima_mape_pseudo = pseudo_metrics(series,
                                                                        lambda tr: SARIMAX(tr, order=(1,1,1), seasonal_order=(1,1,1,4),
                                                                                           enforce_stationarity=False, enforce_invertibility=False).fit(disp=False))

                st.write(f"**In-sample RMSE:** {sarima_rmse_in:.2f}   |   **In-sample MAPE (%):** {sarima_mape_in:.2f}")
                st.write(f"**Pseudo RMSE (last 4 q):** {sarima_rmse_pseudo if not np.isnan(sarima_rmse_pseudo) else 'N/A'}   |   **Pseudo MAPE (%):** {sarima_mape_pseudo if not np.isnan(sarima_mape_pseudo) else 'N/A'}")

                fig_sarima, ax_sarima = plt.subplots(figsize=(12,6))
                ax_sarima.plot(series.index, series, marker='o', label='Actual')
                ax_sarima.plot(series.index, sarima_fitted, color='red', label='Fitted (SARIMA)')
                ax_sarima.plot(sarima_forecast_next4.index, sarima_forecast_next4, marker='o', linestyle='--', color='green', label='Forecast Next 4 Q')
                ax_sarima.set_title('SARIMA Forecast')
                ax_sarima.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
                ax_sarima.legend()
                plt.tight_layout()
                st.pyplot(fig_sarima)
            else:
                st.info("Not enough data for SARIMA (need ~12+ quarters).")
        except Exception as e:
            st.warning(f"SARIMA fitting failed: {e}")

        st.markdown("---")

        # ---------- TSDM (Trend-Seasonal Decomposition) ----------
        st.subheader("TSDM: Trend-Seasonal Decomposition Forecast")
        try:
            decomp = seasonal_decompose(series, model='multiplicative', period=4)
            trend = decomp.trend
            seasonal = decomp.seasonal
            trend_filled = trend.fillna(method='bfill').fillna(method='ffill')
            tsdm_fitted = trend_filled * seasonal

            tsdm_rmse_in = np.nan
            tsdm_mape_in = np.nan
            if tsdm_fitted.dropna().shape[0] > 0:
                tsdm_rmse_in = np.sqrt(mean_squared_error(series.loc[tsdm_fitted.dropna().index], tsdm_fitted.dropna()))
                tsdm_mape_in = safe_mape(series.loc[tsdm_fitted.dropna().index], tsdm_fitted.dropna())

            # forecast by extrapolating last trend slope and multiplying seasonal pattern
            trend_vals = trend_filled[-2:]
            slope = trend_vals.iloc[1] - trend_vals.iloc[0]
            tsdm_forecast_next4 = np.array([trend_vals.iloc[1] + slope*(i+1) for i in range(4)])
            seasonal_pattern = seasonal[-4:].values
            tsdm_forecast_next4 = tsdm_forecast_next4 * seasonal_pattern
            forecast_index = pd.date_range(start=series.index[-1] + pd.offsets.QuarterEnd(1), periods=4, freq='Q')
            tsdm_forecast_next4 = pd.Series(tsdm_forecast_next4, index=forecast_index)

            # pseudo (last 4)
            tsdm_rmse_pseudo, tsdm_mape_pseudo = pseudo_metrics(series,
                                                                lambda tr: (
                                                                    seasonal_decompose(tr, model='multiplicative', period=4),
                                                                    None  # placeholder: we compute pseudo manually below
                                                                )[0]  # won't be used directly; we'll compute pseudo using the same logic
                                                               )
            # compute pseudo manually safely if enough data
            if series.shape[0] > 4:
                train = series[:-4]
                decomp_train = seasonal_decompose(train, model='multiplicative', period=4)
                trend_train = decomp_train.trend.fillna(method='bfill').fillna(method='ffill')
                seasonal_train = decomp_train.seasonal
                slope_train = trend_train[-2:].iloc[1] - trend_train[-2:].iloc[0]
                trend_pf = [trend_train[-1] + slope_train*(i+1) for i in range(4)]
                seasonal_pf = seasonal_train[-4:].values
                tsdm_pf = np.array(trend_pf) * seasonal_pf
                tsdm_pf = pd.Series(tsdm_pf, index=series[-4:].index)
                tsdm_rmse_pseudo = np.sqrt(mean_squared_error(series[-4:], tsdm_pf))
                tsdm_mape_pseudo = safe_mape(series[-4:], tsdm_pf)

            st.write(f"**In-sample RMSE:** {tsdm_rmse_in if not np.isnan(tsdm_rmse_in) else 'N/A'}   |   **In-sample MAPE (%):** {tsdm_mape_in if not np.isnan(tsdm_mape_in) else 'N/A'}")
            st.write(f"**Pseudo RMSE (last 4 q):** {tsdm_rmse_pseudo if not np.isnan(tsdm_rmse_pseudo) else 'N/A'}   |   **Pseudo MAPE (%):** {tsdm_mape_pseudo if not np.isnan(tsdm_mape_pseudo) else 'N/A'}")

            fig_tsdm, ax_tsdm = plt.subplots(figsize=(12,6))
            ax_tsdm.plot(series.index, series, marker='o', label='Actual')
            ax_tsdm.plot(series.index, tsdm_fitted, color='red', label='Fitted (TSDM)')
            ax_tsdm.plot(tsdm_forecast_next4.index, tsdm_forecast_next4, marker='o', linestyle='--', color='green', label='Forecast Next 4 Q')
            ax_tsdm.set_title("TSDM Forecast")
            ax_tsdm.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
            ax_tsdm.legend()
            plt.tight_layout()
            st.pyplot(fig_tsdm)
        except Exception as e:
            st.warning(f"TSDM failed: {e}")

        # Optionally: Model comparison table (compact)
        try:
            comp = pd.DataFrame({
                'Model': ['ExpSmoothing','Holt','Winters','SARIMA','TSDM'],
                'In-sample RMSE': [
                    exp_rmse_in if 'exp_rmse_in' in locals() else np.nan,
                    holt_rmse_in if 'holt_rmse_in' in locals() else np.nan,
                    hw_rmse_in if 'hw_rmse_in' in locals() else np.nan,
                    sarima_rmse_in if 'sarima_rmse_in' in locals() else np.nan,
                    tsdm_rmse_in if 'tsdm_rmse_in' in locals() else np.nan
                ],
                'Pseudo RMSE (last4)': [
                    exp_rmse_pseudo if 'exp_rmse_pseudo' in locals() else np.nan,
                    holt_rmse_pseudo if 'holt_rmse_pseudo' in locals() else np.nan,
                    hw_rmse_pseudo if 'hw_rmse_pseudo' in locals() else np.nan,
                    sarima_rmse_pseudo if 'sarima_rmse_pseudo' in locals() else np.nan,
                    tsdm_rmse_pseudo if 'tsdm_rmse_pseudo' in locals() else np.nan
                ],
                'In-sample MAPE (%)': [
                    exp_mape_in if 'exp_mape_in' in locals() else np.nan,
                    holt_mape_in if 'holt_mape_in' in locals() else np.nan,
                    hw_mape_in if 'hw_mape_in' in locals() else np.nan,
                    sarima_mape_in if 'sarima_mape_in' in locals() else np.nan,
                    tsdm_mape_in if 'tsdm_mape_in' in locals() else np.nan
                ],
                'Pseudo MAPE (%)': [
                    exp_mape_pseudo if 'exp_mape_pseudo' in locals() else np.nan,
                    holt_mape_pseudo if 'holt_mape_pseudo' in locals() else np.nan,
                    hw_mape_pseudo if 'hw_mape_pseudo' in locals() else np.nan,
                    sarima_mape_pseudo if 'sarima_mape_pseudo' in locals() else np.nan,
                    tsdm_mape_pseudo if 'tsdm_mape_pseudo' in locals() else np.nan
                ]
            })
            # format numbers for readability
            comp_display = comp.copy()
            for c in ['In-sample RMSE','Pseudo RMSE (last4)']:
                comp_display[c] = comp_display[c].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
            for c in ['In-sample MAPE (%)','Pseudo MAPE (%)']:
                comp_display[c] = comp_display[c].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
            st.markdown("### Model comparison summary")
            st.dataframe(comp_display, use_container_width=True)
        except Exception:
            pass



    ####################################
    # Variance Analysis tab
    ####################################
    with tabs[3]:
        st.markdown("<h2 style='font-size:26px; margin-bottom:6px'>Variance Analysis</h2>", unsafe_allow_html=True)

        # Build quarterly aggregated table with Budget and Actual (if available)
        if 'Actual' in df.columns:
            quarter_agg = df.groupby('quarter_date').agg(Budget=('Budget', 'sum'), Actual=('Actual', 'sum')).reset_index()
        else:
            # Create Actual column of NaN to keep downstream code simple
            quarter_agg = df.groupby('quarter_date').agg(Budget=('Budget', 'sum')).reset_index()
            quarter_agg['Actual'] = np.nan

        # Prepare series / summary DF
        quarter_agg = quarter_agg.sort_values('quarter_date').reset_index(drop=True)
        quarter_agg['Period'] = quarter_agg['quarter_date']  # Timestamp
        quarter_agg['Quarter_Label'] = quarter_agg['Period'].dt.to_period('Q').astype(str)

        # Variance columns (Actual - Budget) and percent relative to Budget
        quarter_agg['Budget_vs_Actual_Var'] = quarter_agg['Actual'] - quarter_agg['Budget']
        quarter_agg['Budget_vs_Actual_Pct'] = (quarter_agg['Budget_vs_Actual_Var'] / quarter_agg['Budget'].replace(0, np.nan)) * 100

        # -------------------------------------------------------
        # SECTION A — Annual Budget vs Actual (YoY Combined)
        # -------------------------------------------------------
        st.markdown("### Section A — Budget vs Actual (Yearly Totals)")
        st.write("Annual totals for Budget and Actual, including YoY variance and % change.")

        # Aggregate by Fiscal Year
        annual = df.copy()
        annual['Fiscal_Year'] = annual['quarter_date'].dt.year

        annual_agg = (
            annual.groupby('Fiscal_Year')
                  .agg(Budget=('Budget', 'sum'),
                       Actual=('Actual', 'sum') if 'Actual' in df.columns else ('Budget', lambda x: np.nan))
                  .reset_index()
        )

        # Variance columns
        annual_agg['Variance'] = annual_agg['Actual'] - annual_agg['Budget']
        annual_agg['Pct_Variance'] = (annual_agg['Variance'] / annual_agg['Budget'].replace(0, np.nan)) * 100

        # Create YoY comparisons
        annual_agg['Prev_Year_Budget'] = annual_agg['Budget'].shift(1)
        annual_agg['Prev_Year_Actual'] = annual_agg['Actual'].shift(1)

        annual_agg['YoY_Budget_Change'] = annual_agg['Budget'] - annual_agg['Prev_Year_Budget']
        annual_agg['YoY_Actual_Change'] = annual_agg['Actual'] - annual_agg['Prev_Year_Actual']

        annual_agg['YoY_Budget_%'] = (annual_agg['YoY_Budget_Change'] /
                                      annual_agg['Prev_Year_Budget'].replace(0, np.nan)) * 100
        annual_agg['YoY_Actual_%'] = (annual_agg['YoY_Actual_Change'] /
                                      annual_agg['Prev_Year_Actual'].replace(0, np.nan)) * 100

        # ----- Formatting -----
        annual_disp = annual_agg.copy()

        money_cols = ['Budget','Actual','Variance','Prev_Year_Budget','Prev_Year_Actual',
                      'YoY_Budget_Change','YoY_Actual_Change']
        pct_cols = ['Pct_Variance','YoY_Budget_%','YoY_Actual_%']

        for col in money_cols:
            annual_disp[col] = pd.to_numeric(annual_disp[col], errors='coerce').apply(
                lambda x: f"${x:,.0f}" if pd.notnull(x) else ""
            )

        for col in pct_cols:
            annual_disp[col] = pd.to_numeric(annual_disp[col], errors='coerce').apply(
                lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
            )

        st.dataframe(annual_disp, use_container_width=True)

        # ----- Plot annual Budget vs Actual -----
        fig_ann, ax_ann = plt.subplots(figsize=(12,5))
        x = annual_agg['Fiscal_Year']

        ax_ann.bar(x - 0.15, annual_agg['Budget'], width=0.3, label='Budget', color='royalblue')
        ax_ann.bar(x + 0.15, annual_agg['Actual'], width=0.3, label='Actual', color='seagreen')
        ax_ann.set_xlabel("Fiscal Year")
        ax_ann.set_ylabel("Amount")
        ax_ann.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        ax_ann.legend()

        ax_ann.set_title("Annual Budget vs Actual Comparison")
        plt.tight_layout()
        st.pyplot(fig_ann)

        # ----- Plot YoY percentage change -----
        fig_yoy, ax_yoy = plt.subplots(figsize=(12,5))
        ax_yoy.plot(x, annual_agg['YoY_Budget_%'], marker='o', color='royalblue', label='YoY Budget %')
        ax_yoy.plot(x, annual_agg['YoY_Actual_%'], marker='o', color='seagreen', label='YoY Actual %')
        ax_yoy.set_xlabel("Fiscal Year")
        ax_yoy.set_ylabel("YoY % Change")
        ax_yoy.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax_yoy.legend()
        ax_yoy.set_title("Year-over-Year % Change (Budget & Actual)")
        plt.tight_layout()
        st.pyplot(fig_yoy)


        # ----------------------
        # Section B: Year-over-Year (YoY) variance (same quarter last year)
        # ----------------------
        st.markdown("### Section B — Year-over-Year (YoY) Variance (Same Quarter Last Year)")
        fq_summary_local = quarter_agg.copy()
        fq_summary_local['Fiscal_Year'] = fq_summary_local['Period'].dt.year
        fq_summary_local['Quarter_Num'] = fq_summary_local['Period'].dt.quarter

        # pivot tables (Budget & Actual)
        pivot_budget = fq_summary_local.pivot(index='Quarter_Num', columns='Fiscal_Year', values='Budget')
        pivot_actual = fq_summary_local.pivot(index='Quarter_Num', columns='Fiscal_Year', values='Actual')

        latest_year = fq_summary_local['Fiscal_Year'].max()
        prev_year = latest_year - 1

        st.write(f"Comparing quarters: **{prev_year} → {latest_year}**")

        if (prev_year in pivot_budget.columns) and (latest_year in pivot_budget.columns):
            # compute YoY for Budget
            yoy_df = pd.DataFrame(index=pivot_budget.index)
            yoy_df[f'Budget_{prev_year}'] = pivot_budget[prev_year].astype(float)
            yoy_df[f'Budget_{latest_year}'] = pivot_budget[latest_year].astype(float)
            yoy_df['Budget_YoY_Var'] = yoy_df[f'Budget_{latest_year}'] - yoy_df[f'Budget_{prev_year}']
            yoy_df['Budget_YoY_%'] = (yoy_df['Budget_YoY_Var'] / yoy_df[f'Budget_{prev_year}'].replace(0, np.nan)) * 100

            # compute YoY for Actual if available
            actuals_present = (prev_year in pivot_actual.columns) and (latest_year in pivot_actual.columns)
            if actuals_present:
                yoy_df[f'Actual_{prev_year}'] = pivot_actual[prev_year].astype(float)
                yoy_df[f'Actual_{latest_year}'] = pivot_actual[latest_year].astype(float)
                yoy_df['Actual_YoY_Var'] = yoy_df[f'Actual_{latest_year}'] - yoy_df[f'Actual_{prev_year}']
                yoy_df['Actual_YoY_%'] = (yoy_df['Actual_YoY_Var'] / yoy_df[f'Actual_{prev_year}'].replace(0, np.nan)) * 100
            else:
                yoy_df['Actual_YoY_Var'] = np.nan
                yoy_df['Actual_YoY_%'] = np.nan

            # Build a clean display DataFrame with explicit formatting
            disp_rows = []
            for q in yoy_df.index:
                row = {'Quarter': f"Q{int(q)}"}
                # Budget columns
                b_prev = yoy_df.at[q, f'Budget_{prev_year}']
                b_curr = yoy_df.at[q, f'Budget_{latest_year}']
                b_var = yoy_df.at[q, 'Budget_YoY_Var']
                b_pct = yoy_df.at[q, 'Budget_YoY_%']

                row[f'Budget_{prev_year}'] = f"${b_prev:,.0f}" if pd.notnull(b_prev) else ""
                row[f'Budget_{latest_year}'] = f"${b_curr:,.0f}" if pd.notnull(b_curr) else ""
                row['Budget_YoY_Var'] = f"${b_var:,.0f}" if pd.notnull(b_var) else ""
                row['Budget_YoY_%'] = f"{b_pct:.2f}%" if pd.notnull(b_pct) else ""

                # Actual columns (only if present)
                if actuals_present:
                    a_prev = yoy_df.at[q, f'Actual_{prev_year}']
                    a_curr = yoy_df.at[q, f'Actual_{latest_year}']
                    a_var = yoy_df.at[q, 'Actual_YoY_Var']
                    a_pct = yoy_df.at[q, 'Actual_YoY_%']

                    row[f'Actual_{prev_year}'] = f"${a_prev:,.0f}" if pd.notnull(a_prev) else ""
                    row[f'Actual_{latest_year}'] = f"${a_curr:,.0f}" if pd.notnull(a_curr) else ""
                    row['Actual_YoY_Var'] = f"${a_var:,.0f}" if pd.notnull(a_var) else ""
                    row['Actual_YoY_%'] = f"{a_pct:.2f}%" if pd.notnull(a_pct) else ""
                else:
                    row[f'Actual_{prev_year}'] = ""
                    row[f'Actual_{latest_year}'] = ""
                    row['Actual_YoY_Var'] = ""
                    row['Actual_YoY_%'] = ""

                disp_rows.append(row)

            disp_yoy = pd.DataFrame(disp_rows)

            # Display table
            st.dataframe(disp_yoy, use_container_width=True)

            # Plot YoY: bar for Budget_YoY_Var, line for Budget_YoY_%
            fig_b, ax_b = plt.subplots(figsize=(10,5))
            pos_b = np.arange(len(disp_yoy))
            budget_yoy_var_plot = yoy_df['Budget_YoY_Var'].fillna(0).values
            budget_yoy_pct_plot = yoy_df['Budget_YoY_%'].fillna(0).values

            ax_b.bar(pos_b, budget_yoy_var_plot, color='slateblue', label=f'Budget {latest_year} - {prev_year}')
            ax_b.set_xticks(pos_b)
            ax_b.set_xticklabels(disp_yoy['Quarter'], rotation=0)
            ax_b.set_ylabel('Budget Difference')
            ax_b.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))

            ax_b2 = ax_b.twinx()
            ax_b2.plot(pos_b, budget_yoy_pct_plot, color='orange', marker='o', label='Budget YoY %')
            ax_b2.set_ylabel('YoY %')
            ax_b2.yaxis.set_major_formatter(mtick.PercentFormatter())

            # If actual YoY % exists, plot it too
            if actuals_present:
                actual_yoy_pct_plot = yoy_df['Actual_YoY_%'].fillna(0).values
                ax_b2.plot(pos_b, actual_yoy_pct_plot, color='seagreen', marker='s', label='Actual YoY %')

            l1, la1 = ax_b.get_legend_handles_labels()
            l2, la2 = ax_b2.get_legend_handles_labels()
            ax_b.legend(l1 + l2, la1 + la2, loc='upper left')

            ax_b.set_title(f'Year-over-Year Budget Variance: {prev_year} → {latest_year}')
            plt.tight_layout()
            st.pyplot(fig_b)

        else:
            st.info(f"Not enough data to compute YoY comparison for {prev_year} → {latest_year}.")

                

        # ----------------------
        # Section C: Quarter-to-Quarter (QoQ) for the latest year
        # ----------------------
        st.markdown("### Section C — Quarter-to-Quarter (QoQ) Variance (Latest Year)")
        latest_year = fq_summary_local['Fiscal_Year'].max()
        q_latest = fq_summary_local[fq_summary_local['Fiscal_Year'] == latest_year].sort_values('Quarter_Num').copy()

        if q_latest.shape[0] == 0:
            st.info(f"No data found for latest year {latest_year}.")
        else:
            q_latest['QoQ_Budget_Var'] = q_latest['Budget'].diff()
            q_latest['QoQ_Budget_%'] = (q_latest['Budget'].pct_change() * 100)
            if 'Actual' in q_latest.columns and q_latest['Actual'].notna().sum() > 0:
                q_latest['QoQ_Actual_Var'] = q_latest['Actual'].diff()
                q_latest['QoQ_Actual_%'] = (q_latest['Actual'].pct_change() * 100)
            else:
                q_latest['QoQ_Actual_Var'] = np.nan
                q_latest['QoQ_Actual_%'] = np.nan

            disp_qoq = q_latest[['Period','Quarter_Num','Budget','Actual','QoQ_Budget_Var','QoQ_Budget_%','QoQ_Actual_Var','QoQ_Actual_%']].copy()
            disp_qoq['Period'] = disp_qoq['Period'].dt.to_period('Q').astype(str)
            disp_qoq['Quarter_Num'] = disp_qoq['Quarter_Num'].apply(lambda q: f"Q{int(q)}")

            # format money and percent (coerce numeric first)
            for col in ['Budget','Actual','QoQ_Budget_Var','QoQ_Actual_Var']:
                disp_qoq[col] = pd.to_numeric(disp_qoq[col], errors='coerce').apply(lambda x: f"${x:,.0f}" if pd.notnull(x) else "")
            for col in ['QoQ_Budget_%','QoQ_Actual_%']:
                disp_qoq[col] = pd.to_numeric(disp_qoq[col], errors='coerce').apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")

            st.write(f"Quarter-to-Quarter Variance for **{latest_year}**")
            st.dataframe(disp_qoq[['Period','Budget','Actual','QoQ_Budget_Var','QoQ_Budget_%','QoQ_Actual_Var','QoQ_Actual_%']], use_container_width=True)

            # Plot QoQ: bar for QoQ_Budget_Var, line for QoQ_Budget_%
            fig_c, ax_c = plt.subplots(figsize=(10,5))
            pos_c = np.arange(len(disp_qoq))
            qoq_budget_var_plot = q_latest['QoQ_Budget_Var'].fillna(0).values
            qoq_budget_pct_plot = q_latest['QoQ_Budget_%'].fillna(0).values

            ax_c.bar(pos_c, qoq_budget_var_plot, color='seagreen', label='QoQ Budget Difference')
            ax_c.set_xticks(pos_c)
            ax_c.set_xticklabels(disp_qoq['Quarter_Num'], rotation=0)
            ax_c.set_xlabel('Quarter')
            ax_c.set_ylabel('Budget Difference')
            ax_c.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))

            ax_c2 = ax_c.twinx()
            ax_c2.plot(pos_c, qoq_budget_pct_plot, color='orange', marker='o', label='% Change')
            ax_c2.set_ylabel('% Change')
            ax_c2.yaxis.set_major_formatter(mtick.PercentFormatter())

            l1, la1 = ax_c.get_legend_handles_labels()
            l2, la2 = ax_c2.get_legend_handles_labels()
            ax_c.legend(l1 + l2, la1 + la2, loc='upper left')

            ax_c.set_title(f'Quarter-to-Quarter Budget Variance ({latest_year})')
            plt.tight_layout()
            st.pyplot(fig_c)



    ####################################
    # Anomalies tab
    ####################################
    ####################################
    # Anomalies tab (redone for 3 variance sections)
    ####################################
    with tabs[4]:
        st.markdown("<h2 style='font-size:26px; margin-bottom:6px'>Flagged Anomalies</h2>", unsafe_allow_html=True)
        st.write("Detect anomalies on the variance metrics for Section A (Annual), Section B (Quarter YoY), and Section C (QoQ latest year). Use rolling z-score and IsolationForest together.")

        # ---------- Build source dataframes (safe, local copies) ----------
        # Quarter-level aggregated (from earlier)
        qagg = quarter_agg.copy() if 'quarter_agg' in locals() else df.groupby('quarter_date').agg(Budget=('Budget','sum')).reset_index()
        qagg = qagg.sort_values('quarter_date').reset_index(drop=True)
        qagg['Period'] = qagg['quarter_date']
        qagg['Fiscal_Year'] = qagg['Period'].dt.year
        qagg['Quarter_Num'] = qagg['Period'].dt.quarter
        qagg['Quarter_Label'] = qagg['Period'].dt.to_period('Q').astype(str)

        # Annual aggregation (Section A)
        ann = df.copy()
        ann['Fiscal_Year'] = ann['quarter_date'].dt.year
        if 'Actual' in df.columns:
            annual_agg = ann.groupby('Fiscal_Year').agg(Budget=('Budget','sum'), Actual=('Actual','sum')).reset_index()
        else:
            annual_agg = ann.groupby('Fiscal_Year').agg(Budget=('Budget','sum')).reset_index()
            annual_agg['Actual'] = np.nan
        annual_agg = annual_agg.sort_values('Fiscal_Year').reset_index(drop=True)
        annual_agg['Variance'] = annual_agg['Actual'] - annual_agg['Budget']
        annual_agg['Pct_Variance'] = (annual_agg['Variance'] / annual_agg['Budget'].replace(0, np.nan)) * 100
        # YoY for annual
        annual_agg['Prev_Budget'] = annual_agg['Budget'].shift(1)
        annual_agg['YoY_Budget_%'] = (annual_agg['Budget'] - annual_agg['Prev_Budget']) / annual_agg['Prev_Budget'].replace(0, np.nan) * 100
        annual_agg['Prev_Actual'] = annual_agg['Actual'].shift(1)
        annual_agg['YoY_Actual_%'] = (annual_agg['Actual'] - annual_agg['Prev_Actual']) / annual_agg['Prev_Actual'].replace(0, np.nan) * 100

        # Section B: Quarter YoY (recompute safely)
        fq = qagg.copy()
        pivot_budget = fq.pivot(index='Quarter_Num', columns='Fiscal_Year', values='Budget')
        pivot_actual = fq.pivot(index='Quarter_Num', columns='Fiscal_Year', values='Actual') if 'Actual' in fq.columns else pd.DataFrame()
        # We'll compute YoY df for the latest pair of years if available
        fy_latest = fq['Fiscal_Year'].max() if not fq['Fiscal_Year'].isnull().all() else None
        fy_prev = fy_latest - 1 if fy_latest is not None else None

        # Build a safe quarter-YoY dataframe (yoy_quarter_df) if possible
        yoy_quarter_df = None
        if (fy_prev is not None) and (fy_prev in pivot_budget.columns) and (fy_latest in pivot_budget.columns):
            yoy_quarter_df = pd.DataFrame(index=pivot_budget.index)
            yoy_quarter_df[f'Budget_{fy_prev}'] = pivot_budget[fy_prev].astype(float)
            yoy_quarter_df[f'Budget_{fy_latest}'] = pivot_budget[fy_latest].astype(float)
            yoy_quarter_df['Budget_YoY_Var'] = yoy_quarter_df[f'Budget_{fy_latest}'] - yoy_quarter_df[f'Budget_{fy_prev}']
            yoy_quarter_df['Budget_YoY_%'] = (yoy_quarter_df['Budget_YoY_Var'] / yoy_quarter_df[f'Budget_{fy_prev}'].replace(0, np.nan)) * 100

            # Actuals if present
            if (not pivot_actual.empty) and (fy_prev in pivot_actual.columns) and (fy_latest in pivot_actual.columns):
                yoy_quarter_df[f'Actual_{fy_prev}'] = pivot_actual[fy_prev].astype(float)
                yoy_quarter_df[f'Actual_{fy_latest}'] = pivot_actual[fy_latest].astype(float)
                yoy_quarter_df['Actual_YoY_Var'] = yoy_quarter_df[f'Actual_{fy_latest}'] - yoy_quarter_df[f'Actual_{fy_prev}']
                yoy_quarter_df['Actual_YoY_%'] = (yoy_quarter_df['Actual_YoY_Var'] / yoy_quarter_df[f'Actual_{fy_prev}'].replace(0, np.nan)) * 100
            else:
                # ensure columns exist but as NaN
                yoy_quarter_df['Actual_YoY_Var'] = np.nan
                yoy_quarter_df['Actual_YoY_%'] = np.nan

        # Section C: QoQ for latest year
        q_latest = fq[fq['Fiscal_Year'] == fy_latest].sort_values('Quarter_Num').copy() if fy_latest is not None else pd.DataFrame()
        if not q_latest.empty:
            q_latest['QoQ_Budget_Var'] = q_latest['Budget'].diff()
            q_latest['QoQ_Budget_%'] = q_latest['Budget'].pct_change() * 100
            if 'Actual' in q_latest.columns and q_latest['Actual'].notna().sum() > 0:
                q_latest['QoQ_Actual_Var'] = q_latest['Actual'].diff()
                q_latest['QoQ_Actual_%'] = q_latest['Actual'].pct_change() * 100
            else:
                q_latest['QoQ_Actual_Var'] = np.nan
                q_latest['QoQ_Actual_%'] = np.nan

        # ---------- UI controls ----------
        section = st.selectbox("Select section to run anomaly detection on", [
            "Section A — Annual (Variance or YoY%)",
            "Section B — Quarter YoY (Budget/Actual)",
            "Section C — QoQ (Latest Year)"
        ])

        detector_choice = st.multiselect("Detectors to apply", ["Rolling z-score", "IsolationForest"], default=["Rolling z-score","IsolationForest"])

        st.markdown("#### Detector parameters")
        # z-score controls
        z_window = st.slider("Z-score rolling window (periods)", min_value=2, max_value=12, value=4, step=1)
        z_thresh = st.slider("Z-score threshold (abs)", min_value=1.0, max_value=6.0, value=3.0, step=0.1)

        # IsolationForest controls
        iso_cont = st.slider("IsolationForest contamination (fraction)", min_value=0.001, max_value=0.2, value=0.02, step=0.001, format="%.3f")
        iso_n = st.number_input("IsolationForest n_estimators", min_value=50, max_value=500, value=100, step=10)

        # Choose metric depending on section
        metric_options = []
        if section.startswith("Section A"):
            metric_options = ["Annual Variance (Actual - Budget)", "Annual YoY Budget %", "Annual YoY Actual %"]
        elif section.startswith("Section B"):
            metric_options = []
            if yoy_quarter_df is not None:
                metric_options += ["Budget_YoY_Var", "Budget_YoY_%"]
                # only include actual metrics if actuals present
                if 'Actual_YoY_%' in yoy_quarter_df.columns or 'Actual_YoY_Var' in yoy_quarter_df.columns:
                    metric_options += ["Actual_YoY_Var", "Actual_YoY_%"]
            else:
                st.info("Not enough data to compute Quarter YoY metrics (need two years of quarter data).")
        else:
            # Section C
            if not q_latest.empty:
                metric_options = ["QoQ_Budget_Var", "QoQ_Budget_%"]
                if 'QoQ_Actual_%' in q_latest.columns:
                    metric_options += ["QoQ_Actual_Var", "QoQ_Actual_%"]
            else:
                st.info("Not enough QoQ data for latest year.")

        if len(metric_options) == 0:
            st.stop()

        metric = st.selectbox("Choose metric to analyze for anomalies", metric_options)

        # ---------- Prepare series_to_use: index (period) and numeric values ----------
        if section.startswith("Section A"):
            if metric == "Annual Variance (Actual - Budget)":
                series_index = annual_agg['Fiscal_Year']
                series_vals = annual_agg['Variance'].astype(float).copy()
                index_labels = series_index.astype(str)
            elif metric == "Annual YoY Budget %":
                series_index = annual_agg['Fiscal_Year']
                series_vals = annual_agg['YoY_Budget_%'].astype(float).copy()
                index_labels = series_index.astype(str)
            else:  # Annual YoY Actual %
                series_index = annual_agg['Fiscal_Year']
                series_vals = annual_agg['YoY_Actual_%'].astype(float).copy()
                index_labels = series_index.astype(str)

        elif section.startswith("Section B"):
            # quarter YOY: use yoy_quarter_df rows and quarter number as label
            series_index = yoy_quarter_df.index if (yoy_quarter_df is not None) else pd.Index([])
            series_vals = pd.Series(dtype=float)
            index_labels = []
            if metric in yoy_quarter_df.columns:
                series_vals = yoy_quarter_df[metric].astype(float).copy()
                index_labels = [f"Q{int(q)}" for q in series_index]
            else:
                st.error("Chosen metric not available.")
                st.stop()

        else:  # Section C
            if metric in q_latest.columns:
                series_vals = q_latest[metric].astype(float).copy()
                index_labels = q_latest['Quarter_Label'].tolist()
                series_index = q_latest['Period']
            else:
                st.error("Chosen metric not available for QoQ.")
                st.stop()

        # drop NaNs
        s = series_vals.copy()
        s.index = index_labels
        s = s.dropna()
        if s.shape[0] < 3:
            st.warning("Not enough non-NA points to run anomaly detection (need >=3).")
            st.stop()

        st.write(f"Running detection on `{metric}` — {s.shape[0]} points")

        # ---------- Rolling z-score ----------
        z_flag = pd.Series(False, index=s.index)
        if "Rolling z-score" in detector_choice:
            roll_mean = s.rolling(window=z_window, min_periods=1).mean()
            roll_std = s.rolling(window=z_window, min_periods=1).std().replace(0, np.nan)
            zscore = (s - roll_mean) / roll_std
            z_flag = zscore.abs() > z_thresh
        else:
            zscore = pd.Series(np.nan, index=s.index)

        # ---------- IsolationForest ----------
        iso_flag = pd.Series(False, index=s.index)
        if "IsolationForest" in detector_choice:
            from sklearn.ensemble import IsolationForest
            # build feature matrix with value, lag1, rolling mean/std
            feat = pd.DataFrame({'v': s})
            feat['lag1'] = feat['v'].shift(1).fillna(method='bfill')
            feat['rmean3'] = feat['v'].rolling(3, min_periods=1).mean()
            feat['rstd3'] = feat['v'].rolling(3, min_periods=1).std().fillna(0)
            X = feat.fillna(0).values
            iso = IsolationForest(n_estimators=int(iso_n), contamination=float(iso_cont), random_state=0)
            try:
                pred = iso.fit_predict(X)
                iso_flag = pd.Series(pred == -1, index=s.index)
            except Exception as e:
                st.warning(f"IsolationForest failed: {e}")
                iso_flag = pd.Series(False, index=s.index)

        # ---------- Combine and produce output ----------
        combined = pd.DataFrame({
            'Period': s.index,
            'Value': s.values,
            'zscore': zscore.reindex(s.index),
            'z_flag': z_flag.reindex(s.index).fillna(False),
            'iso_flag': iso_flag.reindex(s.index).fillna(False)
        })
        combined['anomaly'] = combined['z_flag'] | combined['iso_flag']
        def reason(r):
            rs = []
            if r['z_flag']:
                rs.append(f"z>{z_thresh}")
            if r['iso_flag']:
                rs.append("IsolationForest")
            return "; ".join(rs)
        combined['reason'] = combined.apply(reason, axis=1)

        n_anom = combined['anomaly'].sum()
        st.write(f"Anomalies detected: **{int(n_anom)}**")

        # ---------- Plot ----------
        fig, ax = plt.subplots(figsize=(12,5))
        x_pos = np.arange(len(combined))
        ax.plot(x_pos, combined['Value'], marker='o', label=metric)
        # mark zscore anomalies
        if combined['z_flag'].any():
            ax.scatter(x_pos[combined['z_flag']], combined.loc[combined['z_flag'],'Value'], color='red', s=90, label=f"Z > {z_thresh}")
        # mark iso anomalies with open marker
        if combined['iso_flag'].any():
            ax.scatter(x_pos[combined['iso_flag']], combined.loc[combined['iso_flag'],'Value'], facecolors='none', edgecolors='orange', s=140, linewidths=2, label='IsolationForest')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(combined['Period'], rotation=45)
        ax.set_ylabel(metric)
        # format y-axis nicely when it's money-like
        # try to detect if metric has large absolute values -> use currency formatting
        if combined['Value'].abs().median() > 1000:
            ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        else:
            ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.2f}'))

        ax.legend()
        ax.set_title(f"Anomaly detection on {metric}")
        plt.tight_layout()
        st.pyplot(fig)

        # ---------- Table of flagged anomalies ----------
        if n_anom > 0:
            flagged = combined[combined['anomaly']].copy()
            # friendly formatting of values
            flagged['Value_fmt'] = flagged['Value'].apply(lambda x: f"${x:,.0f}" if abs(x) >= 1000 else f"{x:.2f}")
            flagged = flagged[['Period','Value_fmt','zscore','z_flag','iso_flag','reason']].rename(columns={'Value_fmt':'Value'})
            st.markdown("#### Flagged anomalies")
            st.dataframe(flagged.sort_values('Period', ascending=False), use_container_width=True)
            csv_bytes = flagged.to_csv(index=False).encode('utf-8')
            st.download_button("Download flagged anomalies (CSV)", data=csv_bytes, file_name="flagged_anomalies.csv", mime="text/csv")
        else:
            st.info("No anomalies flagged. Try lowering the z-score threshold or bumping up IsolationForest contamination.")


else:
    st.info("Upload a CSV file to run BudgetVision analysis.")

import streamlit as st
import pandas as pd
import numpy as np 
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
from river.drift import PageHinkley
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_class_weight
from governance.routing import route_predictions
from governance.metrics import calculate_governance_metrics
from governance.adaptive import adaptive_governance_thresholds
from models.thresholding import find_best_threshold


PLOT_THEME = "plotly_dark"
np.random.seed(42)

st.set_page_config(layout="wide")
st.title("🚨 FinSentinel — Fraud Detection System")

uploaded_file = st.file_uploader("Upload your dataset (CSV)")

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # =====================================================
    # SIDEBAR UI REFORM
    # =====================================================
    
    st.sidebar.markdown("""
    <style>
    
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }
    
    .sidebar-card {
        background-color: #1F2937;
        padding: 18px;
        border-radius: 14px;
        margin-bottom: 18px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    
    .big-value {
        font-size: 20px;
        font-weight: 700;
        color: white;
    }
    
    div[data-testid="stButton"] > button {
        border-radius: 10px;
        height: 48px;
        font-weight: 600;
    }
    
    </style>
    """, unsafe_allow_html=True)
    
    # =====================================================
    # SESSION STATE
    # =====================================================

    if "approve_threshold" not in st.session_state:
        st.session_state["approve_threshold"] = 0.30
    
    if "block_threshold" not in st.session_state:
        st.session_state["block_threshold"] = 0.75
    
    # ====================================
    # APPLY PENDING THRESHOLD UPDATE
    # BEFORE WIDGET CREATION
    # ====================================

    if "pending_approve" in st.session_state:

        st.session_state.approve_threshold=(
            st.session_state.pending_approve
        )

        del st.session_state.pending_approve


    if "pending_block" in st.session_state:

        st.session_state.block_threshold=(
            st.session_state.pending_block
        )

        del st.session_state.pending_block
    
    if "recommended_approve" not in st.session_state:
        st.session_state["recommended_approve"] = None
    
    if "recommended_block" not in st.session_state:
        st.session_state["recommended_block"] = None

    if "recommendation_ready" not in st.session_state:
        st.session_state["recommendation_ready"] = False
    
    if "last_governance_config" not in st.session_state:
        st.session_state["last_governance_config"] = None
    
    
    
    # =====================================================
    # MODEL + HYPERPARAMETERS
    # =====================================================
    
    with st.sidebar.container():
    
        st.markdown("""
        <div class="sidebar-card">
        <div class="big-value">🤖 Model Configuration</div>
        </div>
        """, unsafe_allow_html=True)
    
        model_choice = st.selectbox(
            "Choose Model",
            [
                "Logistic Regression",
                "Random Forest",
                "XGBoost",
                "LightGBM",
                "CatBoost",
                "Neural Network"
            ]
        )
    
        experiment_mode = st.checkbox(
            "🧪 Run Model Comparison"
        )
    
        st.markdown("### ⚙️ Hyperparameters")
    
        if model_choice == "Logistic Regression":
    
            C = st.slider(
                "Regularization Strength (C)",
                0.001,
                10.0,
                1.0
            )
    
            penalty = st.selectbox(
                "Penalty",
                ["l2", "l1"]
            )
    
            if penalty == "l1":
                solver_options = ["liblinear"]
            else:
                solver_options = ["liblinear", "lbfgs"]
    
            solver = st.selectbox(
                "Solver",
                solver_options
            )
    
            max_iter = st.slider(
                "Max Iterations",
                100,
                2000,
                500
            )
    
        elif model_choice == "Random Forest":
    
            n_estimators = st.slider(
                "Number of Trees",
                50,
                500,
                100
            )
    
            max_depth = st.slider(
                "Max Depth",
                2,
                50,
                10
            )
    
            min_samples_split = st.slider(
                "Min Samples Split",
                2,
                20,
                2
            )
    
            min_samples_leaf = st.slider(
                "Min Samples Leaf",
                1,
                20,
                1
            )
    
            bootstrap = st.checkbox(
                "Bootstrap",
                True
            )
    
        elif model_choice == "XGBoost":
    
            n_estimators = st.slider(
                "Trees",
                50,
                500,
                100
            )
    
            max_depth = st.slider(
                "Max Depth",
                2,
                20,
                6
            )
    
            learning_rate = st.slider(
                "Learning Rate",
                0.001,
                0.3,
                0.1
            )
    
            subsample = st.slider(
                "Subsample",
                0.5,
                1.0,
                1.0
            )
    
            colsample_bytree = st.slider(
                "Feature Sampling",
                0.5,
                1.0,
                1.0
            )

        elif model_choice == "LightGBM":

            lgbm_estimators = st.slider(
                "Trees",
                50,
                1000,
                300
            )
        
            lgbm_depth = st.slider(
                "Max Depth",
                2,
                20,
                8
            )
        
            lgbm_lr = st.slider(
                "Learning Rate",
                0.001,
                0.3,
                0.05
            )
        
            lgbm_subsample = st.slider(
                "Subsample",
                0.5,
                1.0,
                0.8
            )
        
            lgbm_colsample = st.slider(
                "Feature Sampling",
                0.5,
                1.0,
                0.8
            )

        elif model_choice == "CatBoost":

            cat_estimators = st.slider(
                "Trees",
                50,
                1000,
                300
            )
        
            cat_depth = st.slider(
                "Depth",
                2,
                12,
                6
            )
        
            cat_lr = st.slider(
                "Learning Rate",
                0.001,
                0.3,
                0.05
            )
    
        elif model_choice == "Neural Network":
    
            hidden_layer_size = st.slider(
                "Hidden Layer Size",
                10,
                200,
                100
            )
    
            alpha = st.slider(
                "Regularization (alpha)",
                0.0001,
                0.01,
                0.0001
            )
    
            max_iter_nn = st.slider(
                "NN Max Iterations",
                100,
                1000,
                300
            )
    
    # =====================================================
    # GOVERNANCE SECTION
    # =====================================================
    
    with st.sidebar.container():
    
        st.markdown("""
        <div class="sidebar-card">
        <div class="big-value">🛡️ Governance System</div>
        </div>
        """, unsafe_allow_html=True)
    
        st.markdown("### Manual Threshold Controls")

        


        st.number_input(
            "Auto Approve Threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.0001,
            format="%.4f",
            key="approve_threshold"
        )

        st.number_input(
            "Auto Block Threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.0001,
            format="%.4f",
            key="block_threshold"
        )

        approve_threshold=st.session_state["approve_threshold"]
        block_threshold=st.session_state["block_threshold"]


        if approve_threshold>=block_threshold:

            st.error(
                "Approve threshold must remain lower than block threshold"
            )

            st.stop()
    
        
    
        st.markdown("---")
    
        optimization_goal = st.selectbox(

            "Optimization Goal",
        
            [
                "Balanced Governance",
                "Aggressive Fraud Containment",
                "Conservative Governance",
                "Operational Efficiency"
            ]
        )
    
        max_review_rate = st.slider(
    
            "Maximum Human Review Rate",
    
            min_value=0.01,
            max_value=0.50,
            value=0.15,
            step=0.01
    
        )

        include_drift = st.checkbox(

            "📡 Include Drift Intelligence",

            value=st.session_state.get(
                "include_drift",
                False
            ),

            key="include_drift",

            help="""
            Uses behavioural drift monitoring
            to adjust threshold recommendations.
            """
        )

        sidebar_config=(

            optimization_goal,

            round(max_review_rate,3),

            include_drift,

            model_choice,

            experiment_mode
        )

        if st.session_state.get(
            "last_sidebar_config"
        ) != sidebar_config:

            st.session_state["recommendation_ready"] = False

            st.session_state["recommended_approve"] = None

            st.session_state["recommended_block"] = None

            st.session_state["last_sidebar_config"] = (
                sidebar_config
            )
            
        

        recommend_button = st.button(
            "⚡ Recommend Governance Policy",
            use_container_width=True
        )
    
        st.markdown("### Recommended Governance Thresholds")



        if include_drift:

            suggested_ai=st.session_state.get(
                "suggested_ai_threshold",
                None
            )

            current_drift=st.session_state.get(
                "drift_score",
                0
            )

            if suggested_ai is not None:

                st.info(f"""

        📡 Drift Intelligence Active

        Current Drift Score:
        {round(current_drift,4)}

        Suggested AI Threshold:
        {suggested_ai}

        Expected impact:

        • Increased analyst oversight

        • Reduced false approvals

        • Improved robustness under uncertainty

        """)

        col1, col2 = st.columns(2)

        if st.session_state["recommendation_ready"]:

            col1.metric(
                "Recommended Approve",
                st.session_state["recommended_approve"]
            )
        
            col2.metric(
                "Recommended Block",
                st.session_state["recommended_block"]
            )
        
        else:
        
            col1.metric(
                "Recommended Approve",
                "--"
            )
        
            col2.metric(
                "Recommended Block",
                "--"
            )
        








        
        
        if st.button(
            "✅ Apply Recommended Thresholds",
            use_container_width=True
        ):

            if (

                st.session_state.get(
                    "recommended_approve"
                ) is not None

                and

                st.session_state.get(
                    "recommended_block"
                ) is not None

            ):

                st.session_state.pending_approve=float(

                    st.session_state[
                        "recommended_approve"
                    ]

                )

                st.session_state.pending_block=float(

                    st.session_state[
                        "recommended_block"
                    ]

                )

                st.session_state.pop(
                    "governance_cache",
                    None
                )

                st.rerun()
    
            
    
            
                
              

    # ---------------------------
    # Data preview + checks
    # ---------------------------
    st.write("### Dataset Preview")
    st.dataframe(df, use_container_width=True, height=300)

    st.write("### 🧹 Data Health Check")
    st.write("Shape:", df.shape)
    missing = df.isnull().sum().sum()
    st.write("Total Missing Values:", missing)
    if missing > 0:
        st.warning("Dataset has missing values — model may behave unpredictably")
    else:
        st.success("No missing values detected")

    target_col = st.selectbox("Select Target Column (Fraud Label)", df.columns)

    if target_col:

        df = df.select_dtypes(include=np.number)
        if target_col not in df.columns:
            st.error("Target column must be numeric (0/1).")
        else:
            X = df.drop(target_col, axis=1)
            y = df[target_col]
 
            suspicious_keywords = [

                "fraud",
                "chargeback",
                "blocked",
                "review",
                "label",
                "target",
                "outcome",
                "decision",
                "flag"
            
            ]
            
            suspicious_cols = [
            
                col for col in X.columns
            
                if any(
                    keyword in col.lower()
                    for keyword in suspicious_keywords
                )
            ]
            
            if suspicious_cols:
            
                st.warning(f"""
                Potential leakage-sensitive columns detected:
                
                {suspicious_cols}
                
                Please verify these do not contain future fraud outcome information.
                """)
            st.write("### Class Distribution")
            st.write(y.value_counts())

            if st.button("Train Model"):

                # RESET OLD STATE
                keys_to_clear = [

                    "probs",
                    "preds",
                    "comparison",
                    "model",
                    "X_sample",
                
                    "shap_Logistic Regression",
                    "shap_Random Forest",
                    "shap_XGBoost",
                
                    "drift_cache",
                    "drift_metrics",
                    "drift_score",
                
                    "governance_df",
                
                    "shap_values",
                    "shap_cache",
                    "monitoring_cache",
                    "routing_cache",
                
                    "baseline_auc",
                    "precalibration_model",
                    "shap_explainer",

                    "governance_cache",
                    "recommended_approve",
                    "recommended_block",
                    "recommendation_ready",
                    "last_governance_config",
                    "last_sidebar_config",
                
                ]
            
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
            
                st.session_state["trained"] = True
                
            if st.session_state.get("trained", False):

                # =========================
                # TRAIN MODEL ONLY ONCE
                # =========================
                required_state_keys = [

                    "probs",
                    "preds",
                    "best_threshold",
                    "y_test",
                    "X_test",
                    "model",
                    "model_choice",
                    "X_train",
                    "y_train"
                
                ]
                
                missing_state = any(
                    key not in st.session_state
                    for key in required_state_keys
                )
                
                if missing_state:

                    with st.spinner("Training model..."):
                
                        feature_names = X.columns
                
                        # =========================
                        # TRAIN / VALIDATION / TEST SPLIT
                        # =========================

                        # =========================
                        # PRESERVE TEMPORAL ORDER FOR DRIFT ANALYSIS
                        # =========================
                        
                        # =========================
                        # TEMPORAL ORDER HANDLING
                        # =========================
                        
                        potential_time_cols = [

                            col for col in X.columns
                        
                            if any(
                                keyword in col.lower()
                                for keyword in [
                                    "time",
                                    "date",
                                    "timestamp",
                                    "transactiondt"
                                ]
                            )
                        ]
                        
                        if len(potential_time_cols) > 0:
                        
                            selected_time_col = potential_time_cols[0]
                        
                            df = df.sort_values(by=selected_time_col)
                            df = df.reset_index(drop=True)

                            X = df.drop(target_col, axis=1)
                            y = df[target_col]
                        
                            st.success(f"""
                            Temporal ordering automatically detected using:
                            {selected_time_col}
                            """)
                        
                        else:
                        
                            df = df.reset_index(drop=True)
                        
                            st.warning("""
                            No explicit temporal column detected.
                            
                            Original dataset ordering is being preserved
                            as an approximation for sequential drift analysis.
                            """)
                        
                        split_idx = int(len(X) * 0.8)

                        X_temp = X.iloc[:split_idx]
                        X_test = X.iloc[split_idx:]
                        
                        y_temp = y.iloc[:split_idx]
                        y_test = y.iloc[split_idx:]


                        
                        
                        val_idx = int(len(X_temp) * 0.75)

                        X_train = X_temp.iloc[:val_idx]
                        X_val = X_temp.iloc[val_idx:]
                        
                        y_train = y_temp.iloc[:val_idx]
                        y_val = y_temp.iloc[val_idx:]

                        st.write("Train Fraud Rate:", round(y_train.mean(), 4))
                        st.write("Validation Fraud Rate:", round(y_val.mean(), 4))
                        st.write("Test Fraud Rate:", round(y_test.mean(), 4))
                
                        scaler = StandardScaler()
                        st.session_state["scaler"] = scaler
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_val_scaled = scaler.transform(X_val)
                        X_test_scaled = scaler.transform(X_test)
                
                        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
                        X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_names)
                        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)
                        scale_pos_weight = (
                            len(y_train[y_train == 0])
                            /
                            len(y_train[y_train == 1])
                        )
                
                        scaled_models = [
                            "Logistic Regression",
                            "Neural Network",
                            "LogReg_C1",
                            "LogReg_C0.1",
                            "LogReg_C10",
                            "MLP_Default",
                            "MLP_Tuned",
                            "MLP_Deep"
                        ]
                
                        results = []
                
                        def evaluate_model(
                            name: str,
                            model,
                            X_tr,
                            X_val_data,
                            X_te
                        ):
                            """
                            Trains and evaluates a fraud detection model
                            using validation threshold optimization.

                            Parameters:
                            ----------
                            name : str
                                Name of the model.

                            model :
                                Machine learning classifier instance.

                            X_tr :
                                Training feature set.

                            X_val_data :
                                Validation feature set.

                            X_te :
                                Test feature set.

                            Returns:
                            -------
                            dict
                                Dictionary containing evaluation metrics.
                            """
                        
                            # =========================
                            # TRAIN
                            # =========================
                            model.fit(X_tr, y_train)
                        
                            # =========================
                            # VALIDATION PROBABILITIES
                            # =========================
                            val_probs = model.predict_proba(X_val_data)[:, 1]
                        
                            best_threshold = find_best_threshold(
                                y_val,
                                val_probs
                            )
                        
                            # =========================
                            # FINAL TEST EVALUATION
                            # =========================
                            probs = model.predict_proba(X_te)[:, 1]
                            preds = (probs > best_threshold).astype(int)
                        
                            results.append({
                                "Model": name,
                                "ROC-AUC": roc_auc_score(y_test, probs),
                                "PR-AUC": average_precision_score(y_test, probs),
                                "Precision": precision_score(y_test, preds, zero_division=0),
                                "Recall": recall_score(y_test, preds, zero_division=0),
                                "F1": f1_score(y_test, preds, zero_division=0),
                                "Threshold": round(best_threshold, 3)
                            })
                        
                            return model, probs, preds, best_threshold
                
                        # =========================
                        # EXPERIMENT MODE
                        # =========================
                        if experiment_mode:
                
                            from sklearn.linear_model import LogisticRegression
                            from sklearn.ensemble import RandomForestClassifier
                            from xgboost import XGBClassifier
                
                            best_score = -1




                        
                            # =========================
                            # EXPERIMENT CONFIGURATIONS
                            # =========================
                            
                            experiment_configs = [
                            
                                {
                                    "name": "LogReg_C1",
                                    "model": LogisticRegression(
                                        C=1,
                                        max_iter=500,
                                        random_state=42,
                                        class_weight="balanced"
                                    ),
                                    "train": X_train_scaled,
                                    "val": X_val_scaled,
                                    "test": X_test_scaled,
                                    "params": {
                                        "C": 1
                                    }
                                },
                            
                                {
                                    "name": "LogReg_C0.1",
                                    "model": LogisticRegression(
                                        C=0.1,
                                        max_iter=500,
                                        random_state=42,
                                        class_weight="balanced"
                                    ),
                                    "train": X_train_scaled,
                                    "val": X_val_scaled,
                                    "test": X_test_scaled,
                                    "params": {
                                        "C": 0.1
                                    }
                                },

                                {
                                    "name": "LogReg_C10",
                                
                                    "model": LogisticRegression(
                                        C=10,
                                        max_iter=1000,
                                        random_state=42,
                                        class_weight="balanced"
                                    ),
                                
                                    "train": X_train_scaled,
                                    "val": X_val_scaled,
                                    "test": X_test_scaled,
                                
                                    "params": {
                                        "C": 10
                                    }
                                },
                                                            
                                {
                                    "name": "RF_100_10",
                                    "model": RandomForestClassifier(
                                        n_estimators=100,
                                        random_state=42,
                                        max_depth=10,
                                        class_weight="balanced"
                                    ),
                                    "train": X_train,
                                    "val": X_val,
                                    "test": X_test,
                                    "params": {
                                        "Trees": 100,
                                        "Depth": 10
                                    }
                                },
                            
                                {
                                    "name": "RF_300_20",
                                    "model": RandomForestClassifier(
                                        n_estimators=300,
                                        random_state=42,
                                        max_depth=20,
                                        class_weight="balanced"
                                    ),
                                    "train": X_train,
                                    "val": X_val,
                                    "test": X_test,
                                    "params": {
                                        "Trees": 300,
                                        "Depth": 20
                                    }
                                },

                                {
                                    "name": "RF_500_None",
                                
                                    "model": RandomForestClassifier(
                                        n_estimators=500,
                                        random_state=42,
                                        max_depth=None,
                                        class_weight="balanced"
                                    ),
                                
                                    "train": X_train,
                                    "val": X_val,
                                    "test": X_test,
                                
                                    "params": {
                                        "Trees": 500,
                                        "Depth": "None"
                                    }
                                },
                            
                                {
                                    "name": "XGB_Default",
                                    "model": XGBClassifier(
                                        eval_metric="logloss",
                                        random_state=42,
                                        scale_pos_weight=scale_pos_weight,
                                    ),
                                    "train": X_train,
                                    "val": X_val,
                                    "test": X_test,
                                    "params": {
                                        "Learning Rate": "default"
                                    }
                                },
                            
                                {
                                    "name": "XGB_Tuned",
                                    "model": XGBClassifier(
                                        n_estimators=300,
                                        random_state=42,
                                        max_depth=8,
                                        learning_rate=0.05,
                                        subsample=0.8,
                                        eval_metric="logloss",
                                        scale_pos_weight=scale_pos_weight,
                                    ),
                                    "train": X_train,
                                    "val": X_val,
                                    "test": X_test,
                                    "params": {
                                        "Trees": 300,
                                        "Depth": 8,
                                        "LR": 0.05
                                    }
                                },

                                {
                                    "name": "XGB_Deep",
                                
                                    "model": XGBClassifier(
                                        n_estimators=500,
                                        random_state=42,
                                        max_depth=12,
                                        learning_rate=0.03,
                                        subsample=0.9,
                                        colsample_bytree=0.9,
                                        eval_metric="logloss",
                                        scale_pos_weight=scale_pos_weight,
                                    ),
                                
                                    "train": X_train,
                                    "val": X_val,
                                    "test": X_test,
                                
                                    "params": {
                                        "Trees": 500,
                                        "Depth": 12,
                                        "LR": 0.03
                                    }
                                },

                                {
                                    "name": "LGBM_Balanced",
                                
                                    "model": LGBMClassifier(
                                
                                        n_estimators=300,
                                        learning_rate=0.05,
                                        max_depth=8,
                                        subsample=0.8,
                                        colsample_bytree=0.8,
                                        random_state=42,
                                        class_weight="balanced"
                                    ),
                                
                                    "train": X_train,
                                    "val": X_val,
                                    "test": X_test,
                                
                                    "params": {
                                        "Trees": 300,
                                        "LR": 0.05,
                                        "Depth": 8
                                    }
                                },

                                {
                                    "name": "LGBM_Recall",
                                
                                    "model": LGBMClassifier(
                                
                                        n_estimators=500,
                                        learning_rate=0.03,
                                        max_depth=12,
                                        subsample=0.9,
                                        colsample_bytree=0.9,
                                        random_state=42,
                                        class_weight="balanced"
                                    ),
                                
                                    "train": X_train,
                                    "val": X_val,
                                    "test": X_test,
                                
                                    "params": {
                                        "Trees": 500,
                                        "LR": 0.03,
                                        "Depth": 12
                                    }
                                },

                                {
                                    "name": "LGBM_Deep",
                                
                                    "model": LGBMClassifier(
                                
                                        n_estimators=700,
                                        learning_rate=0.02,
                                        max_depth=15,
                                        subsample=0.9,
                                        colsample_bytree=0.9,
                                        random_state=42,
                                        class_weight="balanced"
                                    ),
                                
                                    "train": X_train,
                                    "val": X_val,
                                    "test": X_test,
                                
                                    "params": {
                                        "Trees": 700,
                                        "LR": 0.02,
                                        "Depth": 15
                                    }
                                },

                                {
                                    "name": "CAT_Default",
                                
                                    "model": CatBoostClassifier(
                                        verbose=0,
                                        random_state=42,
                                        auto_class_weights="Balanced"
                                    ),
                                
                                    "train": X_train,
                                    "val": X_val,
                                    "test": X_test,
                                
                                    "params": {
                                        "Mode": "Default"
                                    }
                                },
                                
                                {
                                    "name": "CAT_Tuned",
                                
                                    "model": CatBoostClassifier(
                                        iterations=500,
                                        learning_rate=0.05,
                                        depth=8,
                                        verbose=0,
                                        random_state=42,
                                        auto_class_weights="Balanced"
                                    ),
                                
                                    "train": X_train,
                                    "val": X_val,
                                    "test": X_test,
                                
                                    "params": {
                                        "Iterations": 500,
                                        "Depth": 8
                                    }
                                },
                                
                                {
                                    "name": "CAT_Deep",
                                
                                    "model": CatBoostClassifier(
                                        iterations=700,
                                        learning_rate=0.03,
                                        depth=10,
                                        verbose=0,
                                        random_state=42,
                                        auto_class_weights="Balanced"
                                    ),
                                
                                    "train": X_train,
                                    "val": X_val,
                                    "test": X_test,
                                
                                    "params": {
                                        "Iterations": 700,
                                        "Depth": 10
                                    }
                                },
                                
                                {
                                    "name": "MLP_Default",
                                    "model": MLPClassifier(
                                        hidden_layer_sizes=(100,),
                                        alpha=0.0001,
                                        max_iter=300,
                                        random_state=42
                                    ),
                                    "train": X_train_scaled,
                                    "val": X_val_scaled,
                                    "test": X_test_scaled,
                                    "params": {
                                        "Hidden": 100,
                                        "Alpha": 0.0001
                                    }
                                },
                                
                                {
                                    "name": "MLP_Tuned",
                                    "model": MLPClassifier(
                                        hidden_layer_sizes=(128, 64),
                                        alpha=0.001,
                                        max_iter=500,
                                        random_state=42
                                    ),
                                    "train": X_train_scaled,
                                    "val": X_val_scaled,
                                    "test": X_test_scaled,
                                    "params": {
                                        "Hidden": "(128,64)",
                                        "Alpha": 0.001
                                    }
                                },

                                {
                                    "name": "MLP_Deep",
                                
                                    "model": MLPClassifier(
                                        hidden_layer_sizes=(256,128,64),
                                        alpha=0.0005,
                                        max_iter=700,
                                        random_state=42
                                    ),
                                
                                    "train": X_train_scaled,
                                    "val": X_val_scaled,
                                    "test": X_test_scaled,
                                
                                    "params": {
                                        "Hidden": "(256,128,64)",
                                        "Alpha": 0.0005
                                    }
                                },
                            
                            ]
                        
                            
                            
                            for config in experiment_configs:
                            
                                m, probs, preds, th = evaluate_model(
                                    config["name"],
                                    config["model"],
                                    config["train"],
                                    config["val"],
                                    config["test"]
                                )
                            
                                current_score = (

                                    (
                                        f1_score(y_test, preds)
                                        * 0.5
                                    )
                                
                                    +
                                
                                    (
                                        average_precision_score(y_test, probs)
                                        * 0.5
                                    )
                                )
                            
                                # =========================
                                # STORE PARAMETERS
                                # =========================
                            
                                results[-1]["Parameters"] = str(config["params"])
                            
                                if current_score > best_score:

                                    best_model=m
                                    best_probs=probs
                                    best_preds=preds

                                    best_threshold_val=th
                                    best_name=config["name"]

                                    best_model_type=config["name"]

                                    best_score=current_score
                            
                            st.session_state["comparison"] = pd.DataFrame(results)
                            comparison_df = st.session_state["comparison"]
    
                            comparison_df = comparison_df.sort_values(
                                by=["F1", "PR-AUC"],
                                ascending=False
                            )
                            
                            st.subheader("🏆 Ranked Experiment Results")
                            
                            st.dataframe(comparison_df)
                            
                            best_row = comparison_df.iloc[0]
                            
                            st.success(f"""
                            Best Performing Configuration:
                            
                            Model: {best_row['Model']}
                            
                            F1 Score: {round(best_row['F1'], 3)}
                            
                            PR-AUC: {round(best_row['PR-AUC'], 3)}
                            
                            Parameters: {best_row['Parameters']}
                            """)
                            
                            model = best_model
                            probs = best_probs
                            preds = best_preds
                            best_threshold = best_threshold_val
                            model_choice = best_model_type

                            st.session_state.update({
                                "probs": probs,
                                "preds": preds,
                                "best_threshold": best_threshold,
                                "y_test": y_test,
                                "X_test": X_test,
                                "model": model,
                                "model_choice": model_choice,
                                "X_train": X_train,
                                "y_train": y_train,
                                "X_train_scaled": X_train_scaled,
                                "X_test_scaled": X_test_scaled,
                            })
        
                        # =========================
                        # CUSTOM MODE
                        # =========================
                        else:
                
                            if model_choice == "Logistic Regression":
                                from sklearn.linear_model import LogisticRegression
                                model = LogisticRegression(
                                    C=C, penalty=penalty, solver=solver,random_state=42,
                                    max_iter=max_iter, class_weight="balanced"
                                )
                                X_tr, X_te = X_train_scaled, X_test_scaled
                
                            elif model_choice == "Random Forest":
                                from sklearn.ensemble import RandomForestClassifier
                                model = RandomForestClassifier(
                                    n_estimators=n_estimators,
                                    max_depth=max_depth, random_state=42,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    bootstrap=bootstrap,
                                    class_weight="balanced"
                                )
                                X_tr, X_te = X_train, X_test
                
                            elif model_choice == "XGBoost":
                                from xgboost import XGBClassifier
                                model = XGBClassifier(
                                    n_estimators=n_estimators, random_state=42,
                                    max_depth=max_depth,
                                    learning_rate=learning_rate,
                                    subsample=subsample,
                                    colsample_bytree=colsample_bytree,
                                    eval_metric="logloss"
                                )
                                X_tr, X_te = X_train, X_test

                            elif model_choice == "LightGBM":

                                model = LGBMClassifier(
                            
                                    n_estimators=lgbm_estimators,
                                    max_depth=lgbm_depth,
                                    learning_rate=lgbm_lr,
                                    subsample=lgbm_subsample,
                                    colsample_bytree=lgbm_colsample,
                            
                                    random_state=42,
                            
                                    class_weight="balanced"
                                )
                            
                                X_tr, X_te = X_train, X_test

                            elif model_choice == "CatBoost":

                                model = CatBoostClassifier(
                            
                                    iterations=cat_estimators,
                                    depth=cat_depth,
                                    learning_rate=cat_lr,
                            
                                    verbose=0,
                                    random_state=42,
                                    auto_class_weights="Balanced"
                                )
                            
                                X_tr, X_te = X_train, X_test

                            elif model_choice == "Neural Network":

                                model = MLPClassifier(
                                    hidden_layer_sizes=(hidden_layer_size,),
                                    alpha=alpha,
                                    max_iter=max_iter_nn,
                                    random_state=42
                                )
                            
                                X_tr, X_te = X_train_scaled, X_test_scaled
            
                            # 🔥 TRAIN + GENERATE OUTPUTS (THIS WAS MISSING)
    
                            # =========================
                            # TRAIN MODEL
                            # =========================
                            
                            # =========================
                            # TRAIN MODEL
                            # =========================
                            
                            model.fit(X_tr, y_train)
                            
                            # =========================
                            # PROBABILITY CALIBRATION
                            # =========================
                            
                            calibration_models = [
                                "Random Forest",
                                "XGBoost",
                                "LightGBM",
                                "CatBoost"
                            ]
                            
                            if model_choice in calibration_models:
                            
                                try:
                            
                                    calibration_input = (
                                        X_val_scaled
                                        if model_choice in scaled_models
                                        else X_val
                                    )
                            
                                    calibration_base_model = model
                                    st.session_state["precalibration_model"] = calibration_base_model
                            
                                    calibrated_model = CalibratedClassifierCV(
                                        estimator=calibration_base_model,
                                        method="sigmoid",
                                        cv=3
                                    )
                            
                                    calibrated_model.fit(
                                        calibration_input,
                                        y_val
                                    )
                            
                                    model = calibrated_model
                            
                                    st.success(
                                        "Probability calibration applied successfully."
                                    )
                            
                                except Exception as e:
                            
                                    st.warning(f"""
                                    Calibration skipped due to compatibility issue:
                            
                                    {e}
                                    """)
                        
                            # =========================
                            # VALIDATION THRESHOLD TUNING
                            # =========================
                            
                            if model_choice in scaled_models:
                                val_probs = model.predict_proba(X_val_scaled)[:, 1]
                            else:
                                val_probs = model.predict_proba(X_val)[:, 1]
                            
                            thresholds = np.linspace(0.01, 0.99, 1000)
                            
                            f1_scores = []
    
                            for t in thresholds:
                                preds_temp = (val_probs > t).astype(int)
                                f1_scores.append(f1_score(y_val, preds_temp))
    
                            best_threshold = thresholds[np.argmax(f1_scores)]
    
                            # =========================
                            # FINAL TEST EVALUATION
                            # =========================
    
                            probs = model.predict_proba(X_te)[:, 1]
    
                            preds = (probs > best_threshold).astype(int)
    
                            st.session_state.update({
                                "probs": probs,
                                "preds": preds,
                                "best_threshold": best_threshold,
                                "y_test": y_test,
                                "X_test": X_test,
                                "model": model,
                                "model_choice": model_choice,
                                "X_train": X_train,
                                "y_train": y_train,
                                "X_train_scaled": X_train_scaled,
                                "X_test_scaled": X_test_scaled,
                            })
            
                # =========================
                # LOAD FROM SESSION
                # =========================
                probs = st.session_state["probs"]
                preds = st.session_state["preds"]
                best_threshold = st.session_state.get(
                    "best_threshold",
                    0.50
                )
                X_test = st.session_state["X_test"]
                y_test = st.session_state["y_test"]
                model = st.session_state["model"]
                model_choice = st.session_state["model_choice"]
                X_train = st.session_state["X_train"]
                y_train = st.session_state["y_train"]
                X_train_scaled = st.session_state["X_train_scaled"]
                X_test_scaled = st.session_state["X_test_scaled"]

             
                roc_auc = roc_auc_score(y_test, probs)

                pr_auc = average_precision_score(y_test, probs)

                brier = brier_score_loss(y_test, probs)

                baseline_auc = 0.5

                # =====================================================
                # GOVERNANCE ROUTING ENGINE
                # =====================================================
                
                routing_actions=route_predictions(

                    probs,

                    st.session_state[
                        "approve_threshold"
                    ],

                    st.session_state[
                        "block_threshold"
                    ]

                )
                
                # =====================================================
                # RESULTS DATAFRAME
                # =====================================================
                
                df_results = X_test.copy()
                
                df_results["Actual Label"] = y_test.values
                
                df_results["Fraud Probability"] = probs
                
                df_results["Raw ML Prediction"] = preds
                
                df_results["Routing Action"] = routing_actions
                
                df_results["Operational Risk Index"] = probs
                
                # =====================================================
                # GOVERNANCE METRICS
                # =====================================================
                
                human_review_rate = round(
                    (
                        routing_actions.count("Human Review")
                        / len(routing_actions)
                    ) * 100,
                    2
                )
                
                auto_decision_rate = round(
                    (
                        routing_actions.count("Auto Approve")
                        +
                        routing_actions.count("Auto Block")
                    ) / max(len(routing_actions), 1) * 100,
                    2
                )
                
                
                
                # =====================================================
                # GOVERNANCE ROUTING METRICS
                # =====================================================

                routing_array = np.array(routing_actions)
                
                governance_metrics = calculate_governance_metrics(
                    y_test,
                    routing_actions
                )

                legit_auto_approve = governance_metrics["legit_auto_approve"]
                legit_review = governance_metrics["legit_review"]
                legit_block = governance_metrics["legit_block"]

                fraud_auto_approve = governance_metrics["fraud_auto_approve"]
                fraud_review = governance_metrics["fraud_review"]
                fraud_block = governance_metrics["fraud_block"]
                
                # =====================================================
                # GOVERNANCE OPERATIONAL METRICS
                # =====================================================
                
                fraud_capture_rate = round(
                    (
                        fraud_block
                        /
                        max((y_test == 1).sum(), 1)
                    ) * 100,
                    2
                )
                
                fraud_intervention_rate = round(
                    (
                        (fraud_review + fraud_block)
                        /
                        max((y_test == 1).sum(), 1)
                    ) * 100,
                    2
                )
                
                legitimate_intervention_rate = round(
                    (
                        (legit_review + legit_block)
                        /
                        max((y_test == 0).sum(), 1)
                    ) * 100,
                    2
                )




                # =====================================================
                # GOVERNANCE CACHE KEY
                # =====================================================
                
                governance_cache_key=(

                    optimization_goal,

                    round(max_review_rate,3),

                    include_drift,

                    round(approve_threshold,4),

                    round(block_threshold,4),

                    len(probs),

                    round(np.mean(probs),6),

                    round(np.std(probs),6),

                    model_choice,

                    experiment_mode
                )

                current_key = st.session_state.get(
                    "last_governance_config",
                    None
                )

                if current_key != governance_cache_key:

                    st.session_state["recommendation_ready"] = False

                    st.session_state[
                        "last_governance_config"
                    ] = governance_cache_key

                # =====================================================
                # GOVERNANCE THRESHOLD OPTIMIZATION
                # =====================================================
                
                if recommend_button:

                    st.session_state["recommendation_ready"] = False

                    st.session_state.pop(
                        "governance_cache",
                        None
                    )

                    st.session_state.pop(
                        "recommended_approve",
                        None
                    )

                    st.session_state.pop(
                        "recommended_block",
                        None
                    )

                    # force recalculation

                    with st.spinner("Optimizing governance policy..."):

                        best_score=-999999

                        best_approve=approve_threshold
                        best_block=block_threshold

                        best_metrics={}

                        governance_experiments = []
             
                        # ====================================
                        # HYBRID THRESHOLD SEARCH SPACE
                        # ====================================

                        candidate_thresholds=np.round(

                            np.concatenate([

                                np.linspace(
                                    0.05,
                                    0.40,
                                    15
                                ),

                                np.linspace(
                                    0.40,
                                    0.85,
                                    20
                                ),

                                np.percentile(

                                    probs,

                                    [5,10,15,20,
                                    25,30,35,40,
                                    45,50,55,60,
                                    65,70,75,80,
                                    85,90,95]

                                )

                            ]),

                        4)

                        candidate_thresholds=np.unique(
                            candidate_thresholds
                        )
            
                        # KEEP YOUR ENTIRE OPTIMIZATION LOOP HERE
            
                        for approve_t in candidate_thresholds:
    
                            for block_t in candidate_thresholds:
                    
                                # -------------------------
                                # VALID CONFIGURATION
                                # -------------------------
                    
                                if approve_t >= block_t:
                                    continue

                                # =========================================
                                # MINIMUM GOVERNANCE GAP
                                # Prevent unstable routing overlap
                                # =========================================
                                
                                if (block_t - approve_t) < 0.05:
                                    continue
                    
                                governance_preds = []
                                governance_actions = []
                    
                                for p in probs:
                    
                                    if p < approve_t:
                    
                                        governance_preds.append(0)
                                        governance_actions.append(
                                            "Auto Approve"
                                        )
                    
                                    elif p >= block_t:
    
                                        governance_preds.append(1)
                                        governance_actions.append("Auto Block")
                                    
                                    else:

                                        governance_preds.append(-1)
                                    
                                        governance_actions.append("Human Review")
                    
                                governance_preds = np.array(
                                    governance_preds
                                )
                    
                                

                                # =========================================
                                # GOVERNANCE ROUTING EVALUATION
                                # =========================================
                                
                                routing_array_opt = np.array(governance_actions)
                                
                                legit_auto_approve = (
                                    (y_test == 0) &
                                    (routing_array_opt == "Auto Approve")
                                ).sum()
                                
                                legit_review = (
                                    (y_test == 0) &
                                    (routing_array_opt == "Human Review")
                                ).sum()
                                
                                legit_block = (
                                    (y_test == 0) &
                                    (routing_array_opt == "Auto Block")
                                ).sum()
                                
                                fraud_auto_approve = (
                                    (y_test == 1) &
                                    (routing_array_opt == "Auto Approve")
                                ).sum()
                                
                                fraud_review = (
                                    (y_test == 1) &
                                    (routing_array_opt == "Human Review")
                                ).sum()
                                
                                fraud_block = (
                                    (y_test == 1) &
                                    (routing_array_opt == "Auto Block")
                                ).sum()
                                
                                # =========================================
                                # GOVERNANCE PERFORMANCE
                                # =========================================

                                # Human review is intervention only
                                # Final automated actions define confusion metrics

                                tp = fraud_block

                                fn = fraud_auto_approve + fraud_review

                                fp = legit_block

                                tn = legit_auto_approve + legit_review


                                # =====================================================
                                # GOVERNANCE-AWARE COST LABELS
                                # =====================================================

                                review_cases = legit_review + fraud_review

                                wrongful_blocks = legit_block


                                precision_val = (

                                    tp/(tp+fp)

                                    if (tp+fp)>0
                                    else 0

                                )


                                recall_val = (

                                    tp/(tp+fn)

                                    if (tp+fn)>0
                                    else 0

                                )


                                f1_val = (

                                    2*
                                    precision_val*
                                    recall_val
                                    /
                                    (
                                        precision_val+
                                        recall_val
                                    )

                                ) if (

                                    precision_val+
                                    recall_val

                                ) >0 else 0


                                # Intervention effectiveness
                                capture_rate = (

                                    (fraud_review + fraud_block)

                                    /

                                    max(
                                        fraud_review +
                                        fraud_block +
                                        fraud_auto_approve,
                                        1
                                    )

                                )


                                false_positive_rate = (

                                    fp/
                                    (fp+tn)

                                ) if (fp+tn)>0 else 0


                                human_review_rate = (

                                    review_cases/
                                    len(y_test)

                                )


                                automation_rate = (

                                    governance_actions.count(
                                        "Auto Approve"
                                    )

                                    +

                                    governance_actions.count(
                                        "Auto Block"
                                    )

                                )/len(governance_actions)


                                review_penalty = (

                                    human_review_rate * 0.8
                                )

                                operational_cost = (

                                    wrongful_blocks * 25

                                    +

                                    review_cases * 3

                                    +

                                    fn * 15

                                    +

                                    human_review_rate * 150
                                )
                    
                                # -------------------------
                                # HARD CONSTRAINT
                                # -------------------------
                    
                                if human_review_rate > max_review_rate:
                                    continue
                    
                                # ==========================================
                                # GOVERNANCE POLICY OBJECTIVES
                                # ==========================================

                                if optimization_goal=="Balanced Governance":

                                    governance_score=(

                                        0.30*capture_rate
                                        +0.25*precision_val
                                        +0.20*automation_rate
                                        +0.15*f1_val

                                        -0.15*human_review_rate
                                        -0.20*false_positive_rate

                                    )

                                elif optimization_goal=="Aggressive Fraud Containment":

                                    governance_score=(

                                        0.55*recall_val
                                        +0.30*capture_rate
                                        +0.10*precision_val

                                        -0.05*human_review_rate

                                    )

                                elif optimization_goal=="Conservative Governance":

                                    governance_score=(

                                        0.60*precision_val
                                        +0.20*(1-false_positive_rate)
                                        +0.10*human_review_rate

                                        +0.10*automation_rate

                                    )

                                elif optimization_goal=="Operational Efficiency":

                                    governance_score=(

                                        0.55*automation_rate

                                        +0.20*(1-human_review_rate)

                                        +0.15*precision_val

                                        +0.10*capture_rate

                                    )
                    
                                # -------------------------
                                # BEST CONFIGURATION
                                # -------------------------

                                governance_experiments.append({

                                    "Approve Threshold": approve_t,
                                
                                    "Block Threshold": block_t,
                                
                                    "Fraud Capture (%)": round(capture_rate * 100, 2),
                                
                                    "Human Review (%)": round(human_review_rate * 100, 2),
                                
                                    "Automation (%)": round(automation_rate * 100, 2),
                                
                                    "False Positives": fp,
                                
                                    "F1": round(f1_val, 3),
                                
                                    "Operational Cost": round(operational_cost, 2)
                                })
                    
                                if governance_score > best_score:
                    
                                    best_score = governance_score
                    
                                    best_approve = approve_t
                                    best_block = block_t
                    
                                    best_metrics = {

                                        "capture_rate": capture_rate,
                                    
                                        "precision": precision_val,
                                    
                                        "f1": f1_val,
                                    
                                        "review_rate": human_review_rate,
                                    
                                        "automation_rate": automation_rate,
                                    
                                        "false_positive_rate": false_positive_rate,
                                    
                                        "operational_cost": operational_cost,
                                    
                                        "false_negatives": fn,
                                    
                                        "false_positives": fp
                                    }




                        





                        # =====================================================
                        # DRIFT-AWARE RECOMMENDATION LAYER
                        # =====================================================

                        if include_drift:

                            drift_score = st.session_state.get(
                                "drift_score",
                                None
                            )

                            if drift_score is None:

                                st.warning(
                                    "No drift analysis available yet. "
                                    "Run Trust & Monitoring first."
                                )

                                drift_adjustment=0

                            else:

                                drift_adjustment=max(
                                    min(
                                        drift_score*3,
                                        0.10
                                    ),
                                    0.02
                                )

                            # Governance adjustment

                            best_approve=max(
                                best_approve-drift_adjustment,
                                0.10
                            )

                            best_block=max(
                                best_block-drift_adjustment,
                                best_approve+0.10
                            )

                            # AI threshold recommendation

                            suggested_ai_threshold=min(
                                best_threshold+
                                drift_adjustment,
                                0.80
                            )

                            st.session_state[
                                "suggested_ai_threshold"
                            ]=round(
                                suggested_ai_threshold,
                                4
                            )
                        # =====================================================
                        # STORE RECOMMENDED VALUES
                        # =====================================================
                        
                        # =====================================================
                        # FINAL STORE OF RECOMMENDATIONS
                        # =====================================================

                        final_approve = best_approve
                        final_block = best_block

                        if include_drift:

                            current_drift = st.session_state.get(
                                "drift_score",
                                0
                            )

                            adjustment = max(

                                min(
                                    current_drift * 3,
                                    0.10
                                ),

                                0.02
                            )

                            final_approve = max(

                                best_approve - adjustment,

                                0.10
                            )

                            final_block = max(

                                best_block - adjustment,

                                final_approve + 0.10
                            )


                        st.session_state["recommended_approve"] = round(

                            final_approve,

                            4
                        )

                        st.session_state["recommended_block"] = round(

                            final_block,

                            4
                        )
                        
                        st.session_state["recommendation_ready"] = True


                        
                        st.session_state["governance_cache"] = {

                            "key": governance_cache_key,

                            "approve": round(final_approve,4),

                            "block": round(final_block,4),

                            "metrics": best_metrics

                        }


                        
                        st.rerun()
        
                            
                           
        
        
                                    
                # =========================
                # UI (ALWAYS SHOW)
                # =========================
                st.success("Model trained successfully!")
            
                
            
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "🚨 AI Control Center",
                    "📊 Model Evaluation",
                    "🛡️ Trust & Monitoring",
                    "📘 Framework Guide",
                    "🧠 Decision Intelligence"   
                ])

                

                # =====================================================
                # TAB 1 — AI CONTROL CENTER (RESTRUCTURED)
                # =====================================================
                with tab1:
                
                    st.title("🚨 AI Control Center")

                    st.info("""
                    FinSentinel operates through two coordinated intelligence layers:
                    
                    • Predictive Intelligence → evaluates raw fraud detection capability
                    
                    • Governance Intelligence → evaluates how AI predictions are operationally routed using human-in-the-loop decision policies
                    """)
                
                    with st.expander("📘 Understanding Evaluation Metrics"):
                
                        st.markdown("""
                
                        ### ROC-AUC
                        Measures how well the model separates fraud from legitimate transactions.
                
                        • Closer to 1 = better separation  
                        • 0.5 = random guessing
                
                        ---
                
                        ### PR-AUC
                        Precision-Recall AUC is especially important for imbalanced fraud datasets.
                
                        • High PR-AUC = strong fraud detection quality  
                        • More useful than ROC-AUC when fraud cases are rare
                
                        ---
                
                        ### Precision
                        Of all predicted fraud cases:
                        how many were actually fraud?
                
                        High precision = fewer false alarms.
                
                        ---
                
                        ### Recall
                        Of all actual fraud cases:
                        how many did the model detect?
                
                        High recall = fewer missed frauds.
                
                        ---
                
                        ### F1 Score
                        Harmonic balance between precision and recall.
                
                        Used when both:
                        - false positives
                        - false negatives
                        matter.
                
                        ---
                
                        ### Threshold Routing
                        Thresholds do NOT retrain the model.
                
                        They only affect:
                        • Auto Approval
                        • Human Review
                        • Auto Blocking
                
                        This creates a human-in-the-loop fraud governance workflow.
                        """)
                
                    
                
                   

                    # =====================================================
                    # SIDE-BY-SIDE INTELLIGENCE COMPARISON
                    # =====================================================
                    
                    st.markdown("---")
                    
                    left_ai, right_gov = st.columns(2)
                    
                    # =====================================================
                    # LEFT SIDE — AI CAPABILITY
                    # =====================================================
                    
                    with left_ai:
                    
                        st.subheader("🤖 AI Capability")
                    
                        st.caption("""
                        Evaluates raw predictive intelligence BEFORE governance routing.
                        """)
                    
                        ai_k1, ai_k2, ai_k3 = st.columns(3)
                    
                        ai_k1.metric("ROC-AUC", round(roc_auc, 3))
                        ai_k2.metric("PR-AUC", round(pr_auc, 3))
                        ai_k3.metric(
                            "F1 Score",
                            round(
                                f1_score(y_test, preds, zero_division=0),
                                3
                            )
                        )
                    
                        ai_k4, ai_k5 = st.columns(2)
                    
                        ai_k4.metric(
                            "Optimal Threshold",
                            round(best_threshold, 3)
                        )
                    
                        ai_k5.metric(
                            "Brier Score",
                            round(brier, 4)
                        )
                    
                    # =====================================================
                    # RIGHT SIDE — GOVERNANCE
                    # =====================================================
                    
                    with right_gov:
                    
                        st.subheader("🛡️ Human-AI Governance")
                    
                        st.caption("""
                        Evaluates operational routing AFTER governance controls.
                        """)
                    
                        gov_k1, gov_k2 = st.columns(2)
                    
                        gov_k1.metric(
                            "Fraud Capture Rate",
                            f"{fraud_capture_rate}%"
                        )
                        
                        gov_k2.metric(
                            "Fraud Intervention Rate",
                            f"{fraud_intervention_rate}%"
                        )
                    
                        gov_k3, gov_k4, gov_k5 = st.columns(3)
                        gov_k6 = st.columns(1)[0]
                    
                        gov_k3.metric(
                            "Human Review Rate",
                            f"{human_review_rate}%"
                        )
                    
                        gov_k4.metric(
                            "Auto Decision Rate",
                            f"{auto_decision_rate}%"
                        )

                        total_auto_blocked = routing_actions.count("Auto Block")

                        wrongful_block_rate = round(
                            (
                                legit_block
                                /
                                total_auto_blocked
                            ) * 100,
                            2
                        ) if total_auto_blocked > 0 else 0
                        
                        gov_k5.metric(
                            "Wrongful Block Rate",
                            f"{wrongful_block_rate}%"
                        )

                        gov_k6.metric(
                            "Legitimate Intervention Rate",
                            f"{legitimate_intervention_rate}%"
                        )
                    
                    # =====================================================
                    # MATRICES ROW
                    # =====================================================
                    
                    matrix_left, matrix_right = st.columns(2)
                    
                    with matrix_left:
                    
                        st.markdown("### 🧩 AI Confusion Matrix")
                    
                        ai_cm = confusion_matrix(y_test, preds)
                    
                        fig_ai_cm = px.imshow(
                            ai_cm,
                            text_auto=True,
                            color_continuous_scale="Blues",
                            labels=dict(x="Predicted", y="Actual"),
                            x=["Legitimate", "Fraud"],
                            y=["Legitimate", "Fraud"]
                        )
                    
                        fig_ai_cm.update_layout(
                            template=PLOT_THEME,
                            height=600,
                            margin=dict(l=40, r=40, t=60, b=40)
                        )
                    
                        st.plotly_chart(
                            fig_ai_cm,
                            use_container_width=True
                        )
                    
                    with matrix_right:
                    
                        st.markdown("### 🧩 Governance Routing Matrix")
                    
                        governance_matrix = np.array([

                            [
                                ((y_test == 0) &
                                 (routing_array == "Auto Approve")).sum(),
                        
                                ((y_test == 0) &
                                 (routing_array == "Human Review")).sum(),
                        
                                ((y_test == 0) &
                                 (routing_array == "Auto Block")).sum()
                            ],
                        
                            [
                                ((y_test == 1) &
                                 (routing_array == "Auto Approve")).sum(),
                        
                                ((y_test == 1) &
                                 (routing_array == "Human Review")).sum(),
                        
                                ((y_test == 1) &
                                 (routing_array == "Auto Block")).sum()
                            ]
                        ])
                    
                        fig_gov_cm = px.imshow(
                            governance_matrix,
                            text_auto=True,
                            color_continuous_scale="Blues",
                            labels=dict(x="Routing Decision", y="Actual"),
                            x=[
                                "Auto Approve",
                                "Human Review",
                                "Auto Block"
                            ],
                            y=["Legitimate", "Fraud"]
                        )
                    
                        fig_gov_cm.update_layout(
                            template=PLOT_THEME,
                            height=600,
                            margin=dict(l=40, r=40, t=60, b=40)
                        )
                    
                        st.plotly_chart(
                            fig_gov_cm,
                            use_container_width=True
                        )
                    
                    # =====================================================
                    # DONUTS ROW
                    # =====================================================
                    
                    donut_left, donut_right = st.columns(2)
                    
                    with donut_left:
                    
                        st.markdown("### 🍩 AI Prediction Allocation")
                    
                        ai_prediction_df = pd.DataFrame({
                            "Prediction": [
                                "Predicted Legitimate",
                                "Predicted Fraud"
                            ],
                            "Count": [
                                (preds == 0).sum(),
                                (preds == 1).sum()
                            ]
                        })
                    
                        fig_ai_donut = px.pie(
                            ai_prediction_df,
                            names="Prediction",
                            values="Count",
                            template=PLOT_THEME
                        )
                    
                        fig_ai_donut.update_traces(
                            hole=0.60,
                            textinfo="percent+label",
                            textposition="outside"
                        )
                    
                        fig_ai_donut.update_layout(
                            height=500
                        )
                    
                        st.plotly_chart(
                            fig_ai_donut,
                            use_container_width=True
                        )
                    
                    with donut_right:
                    
                        st.markdown("### 🍩 Governance Routing Allocation")
                    
                        routing_counts = pd.Series(
                            routing_actions
                        ).value_counts()
                    
                        routing_df = pd.DataFrame({
                            "Routing Action": routing_counts.index,
                            "Count": routing_counts.values
                        })
                    
                        fig_gov_donut = px.pie(
                            routing_df,
                            names="Routing Action",
                            values="Count",
                            template=PLOT_THEME
                        )
                    
                        fig_gov_donut.update_traces(
                            hole=0.60,
                            textinfo="percent+label",
                            textposition="outside"
                        )
                    
                        fig_gov_donut.update_layout(
                            height=500
                        )
                    
                        st.plotly_chart(
                            fig_gov_donut,
                            use_container_width=True
                        )
 
                    # =====================================================
                    # OPERATIONAL IMPACT
                    # =====================================================
                    
                    ai_flagged = int((preds == 1).sum())
                    
                    ai_legit = int((preds == 0).sum())
                    
                    fraud_detected = fraud_block
                    
                    review_cases = routing_actions.count("Human Review")
                    
                    approved_cases = routing_actions.count("Auto Approve")
                    
                    st.markdown("### 💼 Operational Impact")
                    
                    impact1, impact2, impact3 = st.columns(3)
                    
                    # =====================================================
                    # CARD 1
                    # =====================================================
                    
                    impact1.success(f"""
                    ### 🚫 Fraud Intervention
                    
                    ### AI Capability
                    Flagged as Fraud:
                    {ai_flagged}
                    
                    ### Human-AI Governance
                    Auto Blocked:
                    {fraud_detected}
                    
                    AI identifies suspicious activity.
                    
                    Governance autonomously blocks
                    only high-confidence fraud.
                    """)
                    
                    # =====================================================
                    # CARD 2
                    # =====================================================
                    
                    impact2.info(f"""
                    ### 👨‍💻 Human Dependency
                    
                    ### AI Capability
                    Human Review:
                    N/A
                    
                    ### Human-AI Governance
                    Escalated Cases:
                    {review_cases}
                    
                    Pure AI has no operational oversight.
                    
                    Governance introduces analyst
                    intervention for uncertain cases.
                    """)
                    
                    # =====================================================
                    # CARD 3
                    # =====================================================
                    
                    impact3.warning(f"""
                    ### ⚙️ Autonomous Decisions
                    
                    ### AI Capability
                    Predicted Legitimate:
                    {ai_legit}
                    
                    ### Human-AI Governance
                    Auto Approved:
                    {approved_cases}
                    
                    Governance safely automates
                    low-risk legitimate transactions.
                    """)







                    
                    
                    # =====================================================
                    # RESULTS OVERVIEW
                    # =====================================================
                
                    st.subheader("📋 Results Overview")
                
                    st.caption("""
                    Fraud probabilities and operational routing decisions
                    generated by the AI governance system.
                    """)
                
                    st.dataframe(
                        df_results.reset_index(drop=True),
                        use_container_width=True,
                        height=700
                    )
                
                    # =====================================================
                    # TOP RISKY TRANSACTIONS
                    # =====================================================
                
                    st.subheader("🔺 Highest Risk Transactions")
                
                    st.caption("""
                    Transactions with the highest predicted fraud probability.
                    """)
                
                    st.dataframe(
                        df_results
                        .sort_values(
                            by="Operational Risk Index",
                            ascending=False
                        )
                        .head(30)
                        .reset_index(drop=True),
                    
                        use_container_width=True
                    )
                        
                        
                            
                            


                # =====================================================
                # TAB 2 — MODEL INTELLIGENCE
                # =====================================================
                with tab2:
                
                    st.title("📊 Predictive Intelligence")
                
                    st.caption("""
                    Evaluates model discrimination capability,
                    probability reliability, and threshold sensitivity.
                    """)
                
                    # =========================
                    # MODEL COMPARISON
                    # =========================
                
                    if "comparison" in st.session_state:
                
                        st.subheader("🧪 Model Comparison")
                
                        st.dataframe(
                            st.session_state["comparison"],
                            use_container_width=True
                        )
                
                    # =========================
                    # ROC + PR CURVES
                    # =========================
                
                    st.subheader("📈 Classification Performance Curves")
                
                    fpr, tpr, _ = roc_curve(y_test, probs)
                
                    precision_curve, recall_curve, _ = precision_recall_curve(
                        y_test,
                        probs
                    )
                 
                    col1, col2 = st.columns(2)
                
                    with col1:
                
                        fig = go.Figure()
                
                        fig.add_trace(go.Scatter(
                            x=fpr,
                            y=tpr,
                            mode="lines",
                            name="ROC Curve"
                        ))
                
                        fig.update_layout(
                            title="ROC Curve",
                            xaxis_title="False Positive Rate",
                            yaxis_title="True Positive Rate",
                            template=PLOT_THEME,
                            height=450
                        )
                
                        st.plotly_chart(fig, use_container_width=True)
                
                    with col2:
                
                        fig = go.Figure()
                
                        fig.add_trace(go.Scatter(
                            x=recall_curve,
                            y=precision_curve,
                            mode="lines",
                            name="PR Curve"
                        ))
                
                        fig.update_layout(
                            title="Precision-Recall Curve",
                            xaxis_title="Recall",
                            yaxis_title="Precision",
                            template=PLOT_THEME,
                            height=450
                        )
                
                        st.plotly_chart(fig, use_container_width=True)
                
                    # =========================
                    # BASELINE COMPARISON
                    # =========================
                
                    st.subheader("⚖️ Baseline Comparison")
                
                    col1, col2 = st.columns(2)
                
                    col1.metric(
                        "Baseline ROC-AUC",
                        round(baseline_auc, 4)
                    )
                
                    col2.metric(
                        "Selected Model ROC-AUC",
                        round(roc_auc, 4)
                    )
                
                    # =========================
                    # CALIBRATION ANALYSIS
                    # =========================
                
                    st.subheader("📈 Calibration Analysis")
                
                    bins = min(10, max(3, len(np.unique(probs))))

                    prob_true, prob_pred = calibration_curve(
                        y_test,
                        probs,
                        n_bins=bins
                    )
                
                    fig = go.Figure()
                
                    fig.add_trace(go.Scatter(
                        x=prob_pred,
                        y=prob_true,
                        mode="lines+markers",
                        name="Model Calibration"
                    ))
                
                    fig.add_trace(go.Scatter(
                        x=[0,1],
                        y=[0,1],
                        mode="lines",
                        name="Perfect Calibration",
                        line=dict(dash="dash")
                    ))
                
                    fig.update_layout(
                        title="Calibration Curve",
                        xaxis_title="Predicted Probability",
                        yaxis_title="Observed Fraud Frequency",
                        template=PLOT_THEME,
                        height=500
                    )
                
                    st.plotly_chart(fig, use_container_width=True)
                
                    st.info("""
                    Calibration evaluates whether predicted fraud probabilities
                    align with actual observed fraud likelihood.
                
                    Better calibrated models produce more trustworthy confidence scores.
                    """)
                
                    # =========================
                    # THRESHOLD ANALYSIS
                    # =========================
                
                    st.subheader("🎯 Threshold Sensitivity Analysis")
                
                    thresholds = np.linspace(0.01, 0.99, 25)
                
                    threshold_results = []
                
                    for t in thresholds:

                        preds_temp = np.where(
                            probs >= t,
                            1,
                            0
                        )
                    
                        threshold_results.append({
                    
                            "Threshold": round(t, 2),
                    
                            "Precision": round(
                                precision_score(
                                    y_test,
                                    preds_temp,
                                    zero_division=0
                                ),
                                3
                            ),
                    
                            "Recall": round(
                                recall_score(
                                    y_test,
                                    preds_temp,
                                    zero_division=0
                                ),
                                3
                            ),
                    
                            "F1": round(
                                f1_score(
                                    y_test,
                                    preds_temp,
                                    zero_division=0
                                ),
                                3
                            )
                        })
                
                   
                
                    threshold_df = pd.DataFrame(threshold_results)
                
                    fig = go.Figure()
                
                    fig.add_trace(go.Scatter(
                        x=threshold_df["Threshold"],
                        y=threshold_df["Precision"],
                        mode='lines',
                        name='Precision'
                    ))
                
                    fig.add_trace(go.Scatter(
                        x=threshold_df["Threshold"],
                        y=threshold_df["Recall"],
                        mode='lines',
                        name='Recall'
                    ))
                
                    fig.add_trace(go.Scatter(
                        x=threshold_df["Threshold"],
                        y=threshold_df["F1"],
                        mode='lines',
                        name='F1'
                    ))
                
                    fig.update_layout(
                        title="Threshold Sensitivity Analysis",
                        template=PLOT_THEME,
                        height=500
                    )
                
                    st.plotly_chart(fig, use_container_width=True)





                

                

                # =====================================================
                # TAB 4 — DRIFT (FULL ORIGINAL LOGIC)
                # =====================================================
                with tab3:
                    st.info("""
                    Behavioural drift is detected using Page-Hinkley.
                    
                    This monitors changes in feature distributions over time and identifies 
                    when the model may no longer reflect current data patterns.
                    """)
                    st.write("## 📡 Monitoring Intelligence")
                    st.caption("""
                    Page-Hinkley detects shifts in data distribution over time.
                    This is important in fraud detection where behaviour evolves.
                    """)

                    if "drift_cache" not in st.session_state:

                        feature_detectors = {
                    
                            col: PageHinkley(
                                     min_instances=50,
                                     delta=0.02,
                                     threshold=50
                                 )
                    
                            for col in X_test.columns
                        }
                    
                        drift_timeline = []
                        prediction_history = []
                        drift_events = []
                        drift_episodes = []
                        current_start = None
                    
                        X_test_mean = X_test.mean()
                        X_test_std = X_test.std() + 1e-8
                    
                        for i in range(len(X_test)):
                    
                            row = X_test.iloc[i]
                    
                            prediction_history.append(probs[i])
                    
                            row_flag = 0
                            triggered = []

                            drift_trigger_count = 0
                    
                            for col in X_test.columns:
                    
                                val = (
                                    row[col] - X_test_mean[col]
                                ) / X_test_std[col]
                    
                                detector = feature_detectors[col]
                    
                                detector.update(val)
                    
                                if detector.drift_detected:

                                    drift_trigger_count += 1
                                    triggered.append(col)
                                
                                if drift_trigger_count >= 1:
                                
                                    row_flag = 1
                    
                            drift_timeline.append(row_flag)
                    
                            if row_flag == 1:
                    
                                drift_events.append({
                                    "Index": i,
                                    "Triggered Features": ", ".join(triggered)
                                })
                    
                                if current_start is None:
                                    current_start = i
                    
                            else:
                    
                                if current_start is not None:
                                    drift_episodes.append((current_start, i))
                                    current_start = None
                    
                        if current_start is not None:
                            drift_episodes.append((current_start, len(X_test) - 1))
                    
                        st.session_state["drift_cache"] = {
                            "drift_timeline": drift_timeline,
                            "prediction_history": prediction_history,
                            "drift_events": drift_events,
                            "drift_episodes": drift_episodes
                        }
                    
                    drift_timeline = st.session_state["drift_cache"]["drift_timeline"]
                    prediction_history = st.session_state["drift_cache"]["prediction_history"]
                    drift_events = st.session_state["drift_cache"]["drift_events"]
                    drift_episodes = st.session_state["drift_cache"]["drift_episodes"]







                    

                    # Root cause
                    st.subheader("Root Cause Analysis")
                    st.caption("Features contributing most to detected drift")
                    episode_analysis = []

                    for start, end in drift_episodes:
                        counter = {}
                    
                        for event in drift_events:
                            idx = event["Index"]
                            if start <= idx <= end:
                                feats = event["Triggered Features"].split(", ")
                                for f in feats:
                                    counter[f] = counter.get(f, 0) + 1
                    
                        # ✅ Compute contribution OUTSIDE dictionary
                        if len(counter) > 0:
                            total = sum(counter.values())
                    
                            top_feats = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:5]
                    
                            top_feats_str = ", ".join([
                                f"{k} ({round((v/total)*100, 1)}%)"
                                for k, v in top_feats
                            ])
                        else:
                            top_feats_str = "None"
                    
                        # ✅ THEN append clean dictionary
                        episode_analysis.append({
                            "Start": start,
                            "End": end,
                            "Primary Drift Drivers": top_feats_str
                        })

                    if len(episode_analysis) > 0:
                        st.dataframe(pd.DataFrame(episode_analysis))
                    else:
                        st.info("No root cause data")

                    # Timeline
                    
                    st.info("""
                    This table identifies which transaction features changed most
                    during detected drift periods.
                    
                    Strong drift in important features may indicate evolving fraud behaviour
                    or changing transaction patterns.
                    """)

                    total_drift_points = sum([
                        end - start
                        for start, end in drift_episodes
                    ])
                    
                    drift_score = total_drift_points / len(X_test)
                    st.session_state["drift_score"] = drift_score
                    

                    # =====================================================
                    # ADAPTIVE GOVERNANCE POLICY
                    # =====================================================

                    adaptive_policy = adaptive_governance_thresholds(

                        drift_score,

                        approve_threshold,

                        block_threshold

                    )

                    st.session_state[
                        "adaptive_policy"
                    ] = adaptive_policy

                    # =====================================================
                    # DRIFT SEVERITY INTERPRETATION
                    # =====================================================
                    
                    if drift_score < 0.10:
                    
                        drift_severity = "Low"
                    
                    elif drift_score < 0.30:
                    
                        drift_severity = "Moderate"
                    
                    else:
                    
                        drift_severity = "High"
                    
                    

                    # =====================================================
                    # MONITORING ALERT PANEL
                    # =====================================================

                    st.subheader(
                        "🚨 Monitoring Alert"
                    )

                    driver_names=[]

                    global_counter = st.session_state.get(
                        "global_counter",
                        {}
                    )

                    if len(global_counter)>0:

                        driver_names=list(
                            global_counter.keys()
                        )[:3]


                    alert_df=pd.DataFrame({

                        "Attribute":[

                            "Severity",

                            "Top Drift Drivers",

                            "Recommended Action"

                        ],

                        "Value":[

                            adaptive_policy[
                                "severity"
                            ],

                            ", ".join(
                                driver_names
                            ),

                            (

                            "Increase analyst review and schedule retraining"

                            if adaptive_policy[
                                "severity"
                            ]!="Low"

                            else

                            "Continue monitoring"

                            )
                        ]
                    })


                    st.dataframe(
                        alert_df,
                        use_container_width=True
                    )

                    if drift_severity=="Low":

                        st.success("""
                        Transaction behaviour remains relatively stable.

                        Current fraud patterns appear consistent
                        with historical behaviour.
                        """)
                    
                    if drift_severity == "Low":
                    
                        st.success("""
                        Transaction behaviour remains relatively stable.
                        
                        Current fraud patterns appear consistent with historical behaviour.
                        """)
                    
                    elif drift_severity == "Moderate":
                    
                        st.warning("""
                        Moderate behavioural evolution detected.
                        
                        Fraud strategies or customer behaviour may be changing gradually.
                        """)
                    
                    else:
                    
                        st.error("""
                        Significant behavioural drift detected.
                        
                        Model reliability may degrade without retraining or policy adjustment.
                        """)
                    st.subheader("📊 Drift Summary")

                    avg_drift_length = (
                        np.mean([end-start for start, end in drift_episodes])
                        if len(drift_episodes) > 0
                        else 0
                    )

                    col1, col2, col3, col4 = st.columns(4)

                    col1.metric(
                        "Drift Score",
                        round(drift_score,4)
                    )

                    col2.metric(
                        "Drift Severity",
                        drift_severity
                    )

                    col3.metric(
                        "Drift Episodes",
                        len(drift_episodes)
                    )

                    col4.metric(
                        "Avg Episode Length",
                        round(avg_drift_length,1)
                    )
                    st.info(f"""
                    Approximately {round(drift_score*100,2)}% of the evaluated transaction
                    timeline experienced measurable behavioural drift.
                    
                    Frequent or extended drift periods may reduce long-term
                    model reliability and indicate retraining requirements.
                    """)
                    
                    st.subheader("📈 Drift Timeline")

                    drift_plot_df = pd.DataFrame({
                        "Transaction Index": range(len(drift_timeline)),
                        "Drift Detected": drift_timeline
                    })
                    
                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=drift_plot_df["Transaction Index"],
                        y=drift_plot_df["Drift Detected"],
                        mode="lines",
                        line=dict(width=1),
                        name="Drift Signal"
                    ))
                    
                    fig.update_layout(
                        template=PLOT_THEME,
                        height=350,
                        yaxis_title="Drift Signal",
                        xaxis_title="Transaction Timeline",
                        yaxis=dict(tickvals=[0,1])
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("""
                    This timeline highlights periods where transaction behaviour
                    changed significantly from historical patterns.
                    
                    Sustained drift regions may indicate evolving fraud strategies
                    or changing customer behaviour.
                    """)

                    # =========================
                    # DRIFT IMPACT EXPERIMENT
                    # =========================
                    st.info("""
                    This experiment evaluates whether model performance changes
                    during periods of behavioural drift.
                    
                    Performance degradation during drift may indicate
                    retraining requirements or reduced operational reliability.
                    """)
                    
                    st.subheader("📉 Drift Impact on Model Performance")

                    f1_before = None
                    f1_after = None
                    
                    if len(drift_episodes) > 0:

                        # =========================
                        # BUILD DRIFT REGIONS
                        # =========================
                    
                        drift_indices = []
                    
                        for start, end in drift_episodes:
                            drift_indices.extend(list(range(start, end + 1)))
                    
                        drift_indices = sorted(list(set(drift_indices)))
                    
                        non_drift_indices = [
                            i for i in range(len(X_test))
                            if i not in drift_indices
                        ]
                    
                        # =========================
                        # SAFETY CHECKS
                        # =========================
                    
                        if len(drift_indices) == 0:
                    
                            st.warning("No valid drift indices found.")
                    
                        elif len(non_drift_indices) == 0:
                    
                            st.warning("Entire dataset classified as drift.")
                    
                        else:
                    
                            X_after = X_test.iloc[drift_indices]
                            y_after = y_test.iloc[drift_indices]
                    
                            X_before = X_test.iloc[non_drift_indices]
                            y_before = y_test.iloc[non_drift_indices]


                        # =====================================================
                        # DRIFT SAMPLE VALIDATION
                        # =====================================================
                        
                        fraud_before = int(y_before.sum())
                        fraud_after = int(y_after.sum())
                        
                        st.write("Fraud Samples Before Drift:", fraud_before)
                        st.write("Fraud Samples During Drift:", fraud_after)
                        
                        if fraud_before < 10 or fraud_after < 10:
                        
                            st.warning("""
                            Drift comparison may be statistically unstable.
                        
                            Very small fraud populations were detected
                            in either the stable or drift regions.
                        
                            F1 scores may become unreliable.
                            """)
                    
                            # =========================
                            # EMPTY DATA PROTECTION
                            # =========================
                    
                            if len(X_after) == 0 or len(X_before) == 0:
                    
                                st.warning("Insufficient data for drift comparison.")
                    
                            else:
                    
                                try:
                    
                                    scaler = st.session_state["scaler"]
                    
                                    X_before_scaled = scaler.transform(X_before)
                                    X_after_scaled = scaler.transform(X_after)
                    
                                    if model_choice in scaled_models:
                                        Xb, Xa = X_before_scaled, X_after_scaled
                                    else:
                                        Xb, Xa = X_before, X_after
                    
                                    probs_before = model.predict_proba(Xb)[:, 1]
                                    probs_after = model.predict_proba(Xa)[:, 1]
                    
                                    preds_before = (
                                        probs_before >= best_threshold
                                    ).astype(int)
                                    
                                    preds_after = (
                                        probs_after >= best_threshold
                                    ).astype(int)

                                    st.write(
                                        "Fraud Predictions During Drift:",
                                        int(preds_after.sum())
                                    )
                    
                                    f1_before = f1_score(
                                        y_before,
                                        preds_before,
                                        zero_division=0
                                    )
                    
                                    f1_after = f1_score(
                                        y_after,
                                        preds_after,
                                        zero_division=0
                                    )

                                    # =====================================================
                                    # EXTENDED DRIFT PERFORMANCE METRICS
                                    # =====================================================
                                    
                                    precision_before = precision_score(
                                        y_before,
                                        preds_before,
                                        zero_division=0
                                    )
                                    
                                    precision_after = precision_score(
                                        y_after,
                                        preds_after,
                                        zero_division=0
                                    )
                                    
                                    recall_before = recall_score(
                                        y_before,
                                        preds_before,
                                        zero_division=0
                                    )
                                    
                                    recall_after = recall_score(
                                        y_after,
                                        preds_after,
                                        zero_division=0
                                    )
                                    
                                    pr_auc_before = average_precision_score(
                                        y_before,
                                        probs_before
                                    )
                                    
                                    pr_auc_after = average_precision_score(
                                        y_after,
                                        probs_after
                                    )
                                    
                                    # =====================================================
                                    # STORE FOR REPORTING
                                    # =====================================================
                                    
                                    st.session_state["drift_metrics"] = {
                                    
                                        "F1 Before": round(f1_before, 3),
                                        "F1 After": round(f1_after, 3),
                                    
                                        "Precision Before": round(precision_before, 3),
                                        "Precision After": round(precision_after, 3),
                                    
                                        "Recall Before": round(recall_before, 3),
                                        "Recall After": round(recall_after, 3),
                                    
                                        "PR-AUC Before": round(pr_auc_before, 3),
                                        "PR-AUC After": round(pr_auc_after, 3)
                                    }
                    
                                    col1, col2 = st.columns(2)
                    
                                    if f1_before is not None and f1_after is not None:

                                        col1.metric(
                                            "F1 Before Drift",
                                            round(f1_before, 3)
                                        )
                                    
                                        col2.metric(
                                            "F1 After Drift",
                                            round(f1_after, 3)
                                        )
                                    
                                        st.write(
                                            "📉 Performance Change:",
                                            round(f1_after - f1_before, 3)
                                        )
                                    
                                        if f1_after > f1_before:
                                    
                                            st.success(
                                                "Model performance improved after drift."
                                            )
                                    
                                        else:
                                    
                                            st.warning(
                                                "Model performance decreased after drift."
                                            )

                                    # =====================================================
                                    # DRIFT METRIC COMPARISON TABLE
                                    # =====================================================
                                    
                                    if "drift_metrics" in st.session_state:
                                    
                                        st.subheader("📊 Drift Performance Comparison")
                                    
                                        drift_metrics_df = pd.DataFrame({
                                    
                                            "Metric": [
                                                "F1 Score",
                                                "Precision",
                                                "Recall",
                                                "PR-AUC"
                                            ],
                                    
                                            "Before Drift": [
                                    
                                                st.session_state["drift_metrics"]["F1 Before"],
                                                st.session_state["drift_metrics"]["Precision Before"],
                                                st.session_state["drift_metrics"]["Recall Before"],
                                                st.session_state["drift_metrics"]["PR-AUC Before"]
                                            ],
                                    
                                            "After Drift": [
                                    
                                                st.session_state["drift_metrics"]["F1 After"],
                                                st.session_state["drift_metrics"]["Precision After"],
                                                st.session_state["drift_metrics"]["Recall After"],
                                                st.session_state["drift_metrics"]["PR-AUC After"]
                                            ]
                                        })
                                    
                                        st.dataframe(
                                            drift_metrics_df,
                                            use_container_width=True
                                        )


                                    # =====================================================
                                    # GOVERNANCE BEHAVIOUR DURING DRIFT
                                    # =====================================================
                                    
                                    routing_array_full = np.array(routing_actions)
                                    
                                    before_actions = routing_array_full[non_drift_indices]
                                    after_actions = routing_array_full[drift_indices]
                                    
                                    before_labels = y_test.iloc[non_drift_indices]
                                    after_labels = y_test.iloc[drift_indices]
                                    
                                    # =====================================================
                                    # HUMAN REVIEW RATES
                                    # =====================================================
                                    
                                    review_before = (
                                        (before_actions == "Human Review").sum()
                                        / len(before_actions)
                                    ) * 100
                                    
                                    review_after = (
                                        (after_actions == "Human Review").sum()
                                        / len(after_actions)
                                    ) * 100
                                    
                                    # =====================================================
                                    # AUTO BLOCK RATES
                                    # =====================================================
                                    
                                    block_before = (
                                        (before_actions == "Auto Block").sum()
                                        / len(before_actions)
                                    ) * 100
                                    
                                    block_after = (
                                        (after_actions == "Auto Block").sum()
                                        / len(after_actions)
                                    ) * 100
                                    
                                    # =====================================================
                                    # FALSE POSITIVE ANALYSIS
                                    # =====================================================
                                    
                                    fp_before = (
                                        (
                                            (before_labels == 0)
                                            &
                                            (
                                                (before_actions == "Auto Block")
                                                |
                                                (before_actions == "Human Review")
                                            )
                                        ).sum()
                                    )
                                    
                                    fp_after = (
                                        (
                                            (after_labels == 0)
                                            &
                                            (
                                                (after_actions == "Auto Block")
                                                |
                                                (after_actions == "Human Review")
                                            )
                                        ).sum()
                                    )
                                    
                                    # =====================================================
                                    # DISPLAY
                                    # =====================================================
                                    
                                    st.subheader("🛡️ Governance Behaviour During Drift")
                                    
                                    gov_drift_df = pd.DataFrame({
                                    
                                        "Metric": [
                                    
                                            "Human Review Rate (%)",
                                            "Auto Block Rate (%)",
                                            "False Positives"
                                        ],
                                    
                                        "Stable Period": [
                                    
                                            round(review_before, 2),
                                            round(block_before, 2),
                                            int(fp_before)
                                        ],
                                    
                                        "Drift Period": [
                                    
                                            round(review_after, 2),
                                            round(block_after, 2),
                                            int(fp_after)
                                        ]
                                    })
                                    
                                    st.dataframe(
                                        gov_drift_df,
                                        use_container_width=True
                                    )
                                    
                                    st.info("""
                                    
                                    This analysis evaluates how governance behaviour changes
                                    during periods of behavioural drift.
                                    
                                    Higher review escalation during drift suggests that the
                                    human-AI governance framework adapts to uncertainty by
                                    increasing human oversight.
                                    
                                    """)



                    
                                except Exception as e:
                    
                                    st.warning(
                                        f"Could not compute drift impact: {e}"
                                    )

                    

                    # =========================
                    # GLOBAL DRIFT ANALYSIS (STRONG FIX)
                    # =========================
                    st.subheader("📌 Overall Drift Drivers")
                    global_counter = {}

                    for event in drift_events:
                    
                        feats = event["Triggered Features"].split(", ")
                    
                        for f in feats:
                    
                            global_counter[f] = global_counter.get(f, 0) + 1

                    
                    global_sorted = sorted(
                        global_counter.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )

                    st.session_state[
                        "global_counter"
                    ] = global_counter

                    global_df = pd.DataFrame(
                        global_sorted,
                        columns=["Feature", "Drift Frequency"]
                    )
                    
                    fig = px.bar(
                        global_df,
                        x="Drift Frequency",
                        y="Feature",
                        orientation="h",
                        template=PLOT_THEME,
                        text="Drift Frequency"
                    )
                    
                    fig.update_layout(
                        yaxis=dict(autorange="reversed"),
                        height=450,
                        showlegend=False
                    )
                    
                    fig.update_traces(
                        textposition="outside"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.info("""
                    Features with high drift frequency experienced the largest behavioural
                    changes over time.
                    
                    Persistent drift in important transaction variables may indicate:
                    • evolving fraud strategies
                    • changing customer behaviour
                    • reduced model generalization capability
                    
                    These features should be monitored closely during production deployment.
                    """)


                    try:
                            
                            if "X_sample" not in st.session_state:
    
                                fraud_cases = X_test[y_test == 1]
                                legit_cases = X_test[y_test == 0]
                            
                                fraud_sample_size = min(100, len(fraud_cases))
                                legit_sample_size = min(400, len(legit_cases))
                            
                                fraud_sample = fraud_cases.sample(
                                    n=fraud_sample_size,
                                    random_state=42
                                )
                                
                                legit_sample = legit_cases.sample(
                                    n=legit_sample_size,
                                    random_state=42
                                )
                                
                                X_sample = pd.concat([
                                    fraud_sample,
                                    legit_sample
                                ])

                                X_sample_scaled = st.session_state["scaler"].transform(X_sample)
                                
                                sample_indices = X_sample.index
                                
                                sample_probs = pd.Series(
                                    model.predict_proba(
                                        X_sample if model_choice not in scaled_models
                                        else X_sample_scaled
                                    )[:, 1]
                                ).reset_index(drop=True)
                                
                                st.session_state["sample_probs"] = sample_probs
                            
                                st.session_state["X_sample"] = X_sample
                            
                            X_sample = st.session_state["X_sample"]
                            X_sample_scaled = st.session_state["scaler"].transform(X_sample)
    
                            cache_key = (
                                f"shap_"
                                f"{model_choice}_"
                                f"{str(model.get_params())}"
                            )
    
                            if cache_key not in st.session_state:
                            
                                with st.spinner("Computing SHAP explanations..."):
                            
                                    # =====================================================
                                    # SHAP MODEL HANDLING
                                    # =====================================================

                                    try:

                                        if any(
                                            tree_name in model_choice
                                            for tree_name in [
                                                "XGB",
                                                "Random Forest",
                                                "RF",
                                                "LGBM",
                                                "LightGBM",
                                                "CAT",
                                                "CatBoost"
                                            ]
                                        ):

                                            if isinstance(
                                                model,
                                                CalibratedClassifierCV
                                            ):

                                                shap_model = st.session_state.get(
                                                    "precalibration_model",
                                                    model
                                                )

                                            else:

                                                shap_model=model


                                            explainer=shap.TreeExplainer(
                                                shap_model
                                            )

                                            shap_values=explainer.shap_values(
                                                X_sample,
                                                check_additivity=False
                                            )


                                        elif (
                                            "LogReg" in model_choice
                                            or
                                            model_choice=="Logistic Regression"
                                        ):

                                            explainer=shap.LinearExplainer(
                                                model,
                                                X_sample_scaled
                                            )

                                            shap_values=explainer.shap_values(
                                                X_sample_scaled
                                            )


                                        elif (
                                            "MLP" in model_choice
                                            or
                                            model_choice=="Neural Network"
                                        ):

                                            background=shap.sample(
                                                X_sample_scaled,
                                                100
                                            )

                                            explainer=shap.KernelExplainer(
                                                model.predict_proba,
                                                background
                                            )

                                            shap_values=explainer.shap_values(
                                                X_sample_scaled.iloc[:50]
                                            )


                                        else:

                                            st.warning(
                                                "SHAP unsupported for selected model"
                                            )

                                            shap_values=None
                                            explainer=None


                                    except Exception as e:

                                        st.warning(
                                            f"SHAP explanation failed:\n{e}"
                                        )

                                        shap_values=None
                                        explainer=None


                                    cache_key=(
                                        f"shap_"
                                        f"{model_choice}_"
                                        f"{str(model.get_params())}"
                                    )

                                    st.session_state[cache_key]={

                                        "explainer":explainer,
                                        "shap_values":shap_values

                                    }

                                    st.session_state["shap_values"]=shap_values

                     
                            
                            explainer = st.session_state[cache_key]["explainer"]
                            
                            shap_values = st.session_state[cache_key]["shap_values"]
    
                            if shap_values is not None:
                                st.markdown("---")
                                st.markdown("## 🧠 Explainability Intelligence")
                                st.subheader("🌍 Global Feature Importance")




    
                                # ==================================================
                                # NORMALIZE SHAP OUTPUT
                                # ==================================================

                                if isinstance(shap_values,list):

                                    values=shap_values[1]

                                elif len(np.array(shap_values).shape)==3:

                                    values=shap_values[:,:,1]

                                else:

                                    values=shap_values


                                mean_shap=np.abs(
                                    values
                                ).mean(axis=0)


                                importance_df=pd.DataFrame({

                                    "Feature":X_sample.columns,
                                    "Importance":mean_shap

                                }).sort_values(

                                    by="Importance",
                                    ascending=False

                                )







                                feature_importance_df = importance_df.copy()
                                
                                fig = px.bar(
                                    importance_df,
                                    x="Importance",
                                    y="Feature",
                                    orientation='h',
                                    color="Importance",
                                    height=min(900, max(500, len(importance_df) * 25))
                                )
                                
                                fig.update_layout(
                                    template=PLOT_THEME,
                                    yaxis=dict(autorange="reversed"),
                                    title="Global SHAP Importance"
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)

                                # =====================================================
                                # SHAP STABILITY UNDER DRIFT
                                # =====================================================
                                
                                try:
                                
                                    if len(drift_indices) > 0 and len(non_drift_indices) > 0:
                                
                                        stable_sample_idx = [
                                            i for i in non_drift_indices
                                            if i in X_sample.index
                                        ]
                                
                                        drift_sample_idx = [
                                            i for i in drift_indices
                                            if i in X_sample.index
                                        ]
                                
                                        if len(stable_sample_idx) > 5 and len(drift_sample_idx) > 5:
                                
                                            stable_positions = [
                                                list(X_sample.index).index(i)
                                                for i in stable_sample_idx
                                            ]
                                
                                            drift_positions = [
                                                list(X_sample.index).index(i)
                                                for i in drift_sample_idx
                                            ]





                                
                                            # ==================================================
                                            # NORMALIZE SHAP FOR DRIFT COMPARISON
                                            # ==================================================

                                            if isinstance(shap_values,list):

                                                values=shap_values[1]

                                            elif len(np.array(shap_values).shape)==3:

                                                values=shap_values[:,:,1]

                                            else:

                                                values=shap_values


                                            stable_shap=np.abs(

                                                values[stable_positions]

                                            ).mean(axis=0)


                                            drift_shap=np.abs(

                                                values[drift_positions]

                                            ).mean(axis=0)






                                
                                            shap_drift_df = pd.DataFrame({
                                
                                                "Feature": X_sample.columns,
                                
                                                "Stable Importance": stable_shap,
                                
                                                "Drift Importance": drift_shap
                                            })
                                
                                            shap_drift_df["Importance Shift"] = (
                                                shap_drift_df["Drift Importance"]
                                                -
                                                shap_drift_df["Stable Importance"]
                                            )
                                
                                            shap_drift_df = shap_drift_df.sort_values(
                                                by="Importance Shift",
                                                ascending=False
                                            )
                                
                                            st.subheader("🧠 Explainability Stability Analysis")
                                
                                            st.caption("""
                                            Evaluates how feature importance changes
                                            during behavioural drift periods.
                                            """)
                                
                                            st.dataframe(
                                                shap_drift_df.head(15),
                                                use_container_width=True
                                            )
                                
                                except Exception as e:
                                
                                    st.warning(
                                        f"Could not compute SHAP drift analysis: {e}"
                                    )
                               
    
                                st.subheader("📍 Transaction-Level Explainability")
                                
                                left_col, right_col = st.columns([1,1.5])
    
                                
    
                                with right_col:
    
                                    index = st.selectbox(
                                        "Select Transaction",
                                        range(len(X_sample)),
                                        key="shap_index"
                                    )
                                    
                                    sample = X_sample.iloc[index]





                                    
                                    # ==================================================
                                    # NORMALIZE SHAP FOR LOCAL EXPLANATION
                                    # ==================================================

                                    if isinstance(shap_values,list):

                                        values=shap_values[1]

                                    elif len(np.array(shap_values).shape)==3:

                                        values=shap_values[:,:,1]

                                    else:

                                        values=shap_values


                                    values=values[index]






        
                                    explanation = pd.DataFrame({
                                        "Feature": X_sample.columns,
                                        "Impact": values,
                                        "Value": sample.values
                                    }).sort_values(by="Impact", key=abs, ascending=False)
                                    positive = explanation[explanation["Impact"] > 0]
                                    negative = explanation[explanation["Impact"] < 0]
                                    
                                    top_positive = positive.head(5)
                                    top_negative = negative.head(5)
                                    prediction = st.session_state["sample_probs"].iloc[index]
                                    
                                    fraud_percent = round(prediction * 100, 2)
        
                                    if prediction < approve_threshold:
    
                                        prediction_label = "Legitimate"
                                    
                                    elif prediction >= block_threshold:
                                    
                                        prediction_label = "Potential Fraud"
                                    
                                    else:
                                    
                                        prediction_label = "Human Review"
                                    
                                    
                                    
                                    col1, col2, col3 = st.columns([1, 2, 1])
                                    
                                    col1.metric(
                                        "Fraud Probability",
                                        f"{fraud_percent}%"
                                    )
                                    
                                    col2.metric(
                                        "Model Decision",
                                        prediction_label
                                    )
                                    
                                    col3.metric(
                                        "Risk Score",
                                        round(prediction, 3)
                                    )
                                
                                    st.subheader("📊 Feature Contribution Analysis")
                                    
                                    top_explanation = explanation.copy()
                                    
                                    top_explanation["Direction"] = np.where(
                                        top_explanation["Impact"] > 0,
                                        "Increases Fraud Risk",
                                        "Reduces Fraud Risk"
                                    )
                                    
                                    fig = px.bar(
                                        top_explanation,
                                        x="Impact",
                                        y="Feature",
                                        orientation='h',
                                        color="Direction",
                                        hover_data=["Value"]
                                    )
                                    
                                    fig.update_layout(
                                        template=PLOT_THEME,
                                        yaxis=dict(
                                            autorange="reversed",
                                            automargin=True
                                        ),
                                        margin=dict(l=170, r=20, t=40, b=40),
                                        height=min(
                                            850,
                                            max(500, len(top_explanation) * 28)
                                        ),
                                        showlegend=True
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
    
                                with left_col:
    
    
                                    if shap_values is not None:
    
                                        st.markdown(f"""
                                        ### 🧠 Analyst Narrative
                                        
                                        The model classified this transaction as **{prediction_label}**
                                        with a predicted fraud probability of **{fraud_percent}%**.
                                        
                                        ---
                                        
                                        ##### Reduced Fraud Risk
                                        """)
                                        
                                        for _, row in top_negative.iterrows():
                                            st.markdown(
                                                f"- **{row['Feature']}** reduced fraud likelihood"
                                            )
                                        
                                        st.markdown("""
                                        ---
                                        
                                        ##### Increased Fraud Risk
                                        """)
                                        
                                        for _, row in top_positive.iterrows():
                                            st.markdown(
                                                f"- **{row['Feature']}** increased fraud suspicion"
                                            )
            
                                       
                                    
                                # =====================================================
                                # SHAP WATERFALL
                                # =====================================================

                                if explainer is not None:

                                    st.subheader("🔬 SHAP Waterfall Analysis")

                                    plt.style.use("dark_background")
                                    plt.close('all')

                                    fig=plt.figure(figsize=(16,8))

                                    try:

                                        base_value=(
                                            explainer.expected_value[1]
                                            if isinstance(
                                                explainer.expected_value,
                                                (list,np.ndarray)
                                            )
                                            else explainer.expected_value
                                        )

                                        shap.plots.waterfall(

                                            shap.Explanation(

                                                values=values,
                                                base_values=base_value,
                                                data=sample,
                                                feature_names=X_sample.columns

                                            ),

                                            max_display=len(X_sample.columns),
                                            show=False

                                        )

                                        st.pyplot(fig)

                                    except Exception as e:

                                        st.warning(
                                            f"Waterfall generation failed: {e}"
                                        )

                                    plt.close(fig)

                                else:

                                    st.info(
                                        "SHAP waterfall unavailable for current model."
                                    )



    
    
                    except Exception as e:
                        st.error(f"SHAP error: {e}")

                # =====================================================
                # TAB 4 — FRAMEWORK GUIDE + SHAP
                # =====================================================
                with tab4:

                    st.title("📘 FinSentinel Framework")

                    st.subheader("🏗️ FinSentinel Intelligence Architecture")
                    
                    architecture_df = pd.DataFrame({
                    
                        "Intelligence Layer": [
                    
                            "Predictive Intelligence",
                            "Governance Intelligence",
                            "Monitoring Intelligence",
                            "Explainability Intelligence"
                    
                        ],
                    
                        "Core Objective": [
                    
                            "Detect fraud accurately",
                            "Route operational decisions safely",
                            "Monitor behavioural stability",
                            "Explain AI decisions transparently"
                    
                        ],
                    
                        "Key Components": [
                    
                            "ROC, PR, F1, Calibration",
                            "Threshold Routing, Human Review",
                            "Drift Detection, Drift Impact",
                            "SHAP Feature Attribution"
                    
                        ]
                    })
                    
                    st.dataframe(
                        architecture_df,
                        use_container_width=True
                    )
                
                    st.markdown("""
                
                    ## System Architecture
                
                    FinSentinel is a trust-aware fraud intelligence framework
                    combining:
                
                    • Machine Learning Prediction  
                    • Human-in-the-Loop Governance  
                    • Explainable AI (SHAP)  
                    • Drift Detection  
                    • Dynamic Threshold Routing  
                
                    ---
                
                    ## Operational Workflow
                
                    Step 1 — Fraud Probability Prediction  
                    The model predicts fraud likelihood.
                
                    Step 2 — Dynamic Routing  
                    Transactions are categorized into:
                
                    • Auto Approve  
                    • Human Review  
                    • Auto Block  
                
                    based on configurable thresholds.
                
                    Step 3 — Explainability  
                    SHAP explanations identify
                    which features influenced predictions.
                
                    Step 4 — Drift Monitoring  
                    Behavioural drift detection identifies
                    changing fraud patterns over time.
                
                    Step 5 — Human Oversight  
                    Medium-risk transactions are escalated
                    to analysts before automated action occurs.
                
                    ---
                
                    ## Why This Framework Matters
                
                    Traditional fraud systems focus only on prediction.
                
                    FinSentinel extends beyond prediction into:
                
                    • operational governance  
                    • explainability  
                    • adaptive monitoring  
                    • human-AI collaboration  
                    • deployment reliability
                
                    """)

                    st.subheader("📜 Regulatory Alignment Matrix")

                    reg_df = pd.DataFrame({
                    
                        "Principle": [
                            "Transparency",
                            "Human Oversight",
                            "Auditability",
                            "Adaptive Monitoring"
                        ],
                    
                        "System Component": [
                            "SHAP Explainability",
                            "Human Review Routing",
                            "Decision Intelligence",
                            "Drift Detection"
                        ]
                    })
                    
                    st.dataframe(
                        reg_df,
                        use_container_width=True
                    )


                # =====================================================
                # TAB 5 — DECISION INTELLIGENCE
                # =====================================================
                
                with tab5:
                
                    st.title("🧠 Decision Intelligence")
                
                    st.markdown("""
                    This intelligence layer compares:
                
                    • Ground Truth (real transaction outcomes)
                
                    • AI Capability (pure machine learning predictions)
                
                    • Human-AI Governance (threshold-based operational routing)
                
                    The purpose is to evaluate how operational governance
                    changes fraud detection behaviour, analyst dependency,
                    automation efficiency, and false positive exposure.
                    """)
                
                    st.divider()
                
                    # =====================================================
                    # GOVERNANCE CONFIGURATION
                    # =====================================================
                
                    st.subheader("🎛️ Active Governance Thresholds")
                
                    st.info(f"""
                    Current Operational Configuration
                
                    • Auto Approve Threshold: {round(approve_threshold,4)}
                
                    • Auto Block Threshold: {round(block_threshold,4)}
                
                    These thresholds affect ONLY governance routing.
                    They do NOT retrain the AI model.
                    """)
                
                    st.divider()
                
                    # =====================================================
                    # BASELINE MODEL OUTPUTS
                    # =====================================================
                
                    baseline_preds = st.session_state["preds"]
                
                    validation_threshold = best_threshold
                
                    # =====================================================
                    # GOVERNANCE ROUTING
                    # =====================================================
                
                    governance_actions = []
                    governance_preds = []
                
                    for p in probs:
                
                        if p < approve_threshold:
                
                            governance_actions.append("Auto Approve")
                            governance_preds.append(0)
                
                        elif p >= block_threshold:

                            governance_preds.append(1)
                            governance_actions.append("Auto Block")
                        
                        else:
                        
                            # Human review inherits AI prediction
                            # instead of automatically assuming fraud
                        
                            governance_preds.append(
                                1 if p >= validation_threshold else 0
                            )
                        
                            governance_actions.append("Human Review")
                
                    governance_preds = np.array(governance_preds)
                
                    # =====================================================
                    # CONFUSION MATRICES
                    # =====================================================
                
                    ground_cm = np.array([
                        [(y_test == 0).sum(), 0],
                        [0, (y_test == 1).sum()]
                    ])
                
                    baseline_cm = confusion_matrix(
                        y_test,
                        baseline_preds
                    )
                
                    
                
                    tn_b, fp_b, fn_b, tp_b = baseline_cm.ravel()
                
                    # =====================================================
                    # GOVERNANCE ROUTING MATRIX (3-STATE)
                    # =====================================================
                    
                    gov_cm = np.array([
                    
                        [
                            ((y_test == 0) &
                             (np.array(governance_actions) == "Auto Approve")).sum(),
                    
                            ((y_test == 0) &
                             (np.array(governance_actions) == "Human Review")).sum(),
                    
                            ((y_test == 0) &
                             (np.array(governance_actions) == "Auto Block")).sum()
                        ],
                    
                        [
                            ((y_test == 1) &
                             (np.array(governance_actions) == "Auto Approve")).sum(),
                    
                            ((y_test == 1) &
                             (np.array(governance_actions) == "Human Review")).sum(),
                    
                            ((y_test == 1) &
                             (np.array(governance_actions) == "Auto Block")).sum()
                        ]
                    ])
                    
                    # =====================================================
                    # GOVERNANCE METRICS
                    # =====================================================
                    
                    tn_g = gov_cm[0][0]
                    
                    fp_g = gov_cm[0][1] + gov_cm[0][2]
                    
                    fn_g = gov_cm[1][0]
                    
                    tp_g = gov_cm[1][1] + gov_cm[1][2]
                    
                    global_zmax = max(
                        ground_cm.max(),
                        baseline_cm.max(),
                        gov_cm.max()
                    )
                
                    global_zmax = max(
                        ground_cm.max(),
                        baseline_cm.max(),
                        gov_cm.max()
                    )
                
                    # =====================================================
                    # MASTER COMPARISON DATAFRAME
                    # =====================================================
                
                    comparison_df = pd.DataFrame({
                
                        "Transaction ID": range(len(y_test)),
                
                        "Actual Reality": np.where(
                            y_test.values == 1,
                            "Fraud",
                            "Legitimate"
                        ),
                
                        "AI Prediction": np.where(
                            baseline_preds == 1,
                            "Fraud",
                            "Legitimate"
                        ),
                
                        "Governance Decision": governance_actions,
                
                        "Fraud Probability": np.round(probs, 4)
                
                    })
                
                    # =====================================================
                    # OUTCOME ENGINE
                    # =====================================================
                
                    def classify_outcome(row):
                
                        if (
                            row["Actual Reality"] == "Fraud"
                            and
                            row["Governance Decision"] == "Auto Block"
                        ):
                            return "Fraud Intercepted"
                
                        elif (
                            row["Actual Reality"] == "Fraud"
                            and
                            row["Governance Decision"] == "Auto Approve"
                        ):
                            return "Missed Fraud"
                
                        elif (
                            row["Actual Reality"] == "Legitimate"
                            and
                            row["Governance Decision"] == "Auto Block"
                        ):
                            return "False Positive"
                
                        elif row["Governance Decision"] == "Human Review":
                            return "Human Escalation"
                
                        else:
                            return "Correct Approval"
                
                    comparison_df["Outcome"] = comparison_df.apply(
                        classify_outcome,
                        axis=1
                    )
                
                    # =====================================================
                    # COMPARATIVE KPI TABLE
                    # =====================================================
                
                    st.subheader("📊 Comparative Intelligence Matrix")
                
                    comparative_df = pd.DataFrame({
                
                        "Metric": [
                            "Total Frauds",
                            "Legitimate Transactions",
                            "False Positives",
                            "Missed Frauds",
                            "Human Reviews",
                            "Fraud Capture Rate"
                        ],
                
                        "Ground Truth": [
                            int((y_test == 1).sum()),
                            int((y_test == 0).sum()),
                            "N/A",
                            "N/A",
                            "N/A",
                            "100%"
                        ],
                
                        "AI Capability": [
                            int(tp_b),
                            int(tn_b),
                            int(fp_b),
                            int(fn_b),
                            "N/A",
                            f"{round((tp_b/(tp_b+fn_b))*100,2)}%"
                        ],
                
                        "Human-AI Governance": [
                            int(tp_g),
                            int(tn_g),
                            int(fp_g),
                            int(fn_g),
                            governance_actions.count("Human Review"),
                            f"{round((tp_g/(tp_g+fn_g))*100,2)}%"
                        ]
                    })
                
                    st.dataframe(
                        comparative_df,
                        use_container_width=True,
                        height=320
                    )
                
                    st.divider()
                
                    # =====================================================
                    # THREE COLUMN COMPARISON
                    # =====================================================
                
                    col1, col2, col3 = st.columns(3)
                
                    # =====================================================
                    # COLUMN 1 — GROUND TRUTH
                    # =====================================================
                
                    with col1:
                
                        st.markdown("## 🌍 Ground Truth")
                
                        gt1, gt2 = st.columns(2)
                
                        gt1.metric(
                            "Frauds",
                            int((y_test == 1).sum())
                        )
                
                        gt2.metric(
                            "Legitimate",
                            int((y_test == 0).sum())
                        )
                
                        # -------------------------
                        # DONUT
                        # -------------------------
                
                        ground_donut = pd.DataFrame({
                            "Category": ["Legitimate", "Fraud"],
                            "Count": [
                                int((y_test == 0).sum()),
                                int((y_test == 1).sum())
                            ]
                        })
                
                        fig_ground_donut = px.pie(
                            ground_donut,
                            names="Category",
                            values="Count",
                            template=PLOT_THEME
                        )
                
                        fig_ground_donut.update_traces(
                            hole=0.60,
                            textinfo="percent+label",
                            textfont_size=14,
                            textposition="outside"
                        )
                
                        fig_ground_donut.update_layout(
                            height=320,
                            showlegend=True,
                            uniformtext_minsize=12,
                            uniformtext_mode='hide'
                        )
                
                        st.plotly_chart(
                            fig_ground_donut,
                            use_container_width=True
                        )

                        
                
                        # -------------------------
                        # MATRIX
                        # -------------------------
                
                        fig_ground_cm = px.imshow(
                            ground_cm,
                            text_auto=True,
                            color_continuous_scale="Blues",
                            zmin=0,
                            zmax=global_zmax,
                            x=["Legitimate", "Fraud"],
                            y=["Legitimate", "Fraud"],
                            labels=dict(
                                x="Reality",
                                y="Reality"
                            )
                        )
                
                        fig_ground_cm.update_layout(
                            template=PLOT_THEME,
                            height=350
                        )
                
                        st.plotly_chart(
                            fig_ground_cm,
                            use_container_width=True
                        )
                
                        st.info("""
                        Represents true labelled transaction outcomes.
                        No AI inference is involved here.
                        """)
                
                    # =====================================================
                    # COLUMN 2 — AI CAPABILITY
                    # =====================================================
                
                    with col2:
                
                        st.markdown("## 🤖 AI Capability")
                
                        ai1, ai2 = st.columns(2)
                
                        ai1.metric(
                            "Recall",
                            round(
                                recall_score(
                                    y_test,
                                    baseline_preds
                                ),
                                3
                            )
                        )
                
                        ai2.metric(
                            "Precision",
                            round(
                                precision_score(
                                    y_test,
                                    baseline_preds,
                                    zero_division=0
                                ),
                                3
                            )
                        )
                
                        ai3, ai4 = st.columns(2)
                
                        ai3.metric(
                            "F1",
                            round(
                                f1_score(
                                    y_test,
                                    baseline_preds,
                                    zero_division=0
                                ),
                                3
                            )
                        )
                
                        ai4.metric(
                            "Optimal Decision Threshold",
                            round(validation_threshold, 3)
                        )
                
                        # -------------------------
                        # DONUT
                        # -------------------------
                
                        ai_donut = pd.DataFrame({
                            "Category": ["Predicted Legitimate", "Predicted Fraud"],
                            "Count": [
                                (baseline_preds == 0).sum(),
                                (baseline_preds == 1).sum()
                            ]
                        })
                
                        fig_ai_donut = px.pie(
                            ai_donut,
                            names="Category",
                            values="Count",
                            template=PLOT_THEME
                        )
                
                        fig_ai_donut.update_traces(
                            hole=0.60,
                            textinfo="percent+label",
                            textfont_size=14,
                            textposition="outside"
                        )
                
                        fig_ai_donut.update_layout(
                            height=320,
                            showlegend=True,
                            uniformtext_minsize=12,
                            uniformtext_mode='hide'
                        )
                
                        st.plotly_chart(
                            fig_ai_donut,
                            use_container_width=True
                        )
                
                        # -------------------------
                        # MATRIX
                        # -------------------------
                
                        fig_ai_cm = px.imshow(
                            baseline_cm,
                            text_auto=True,
                            color_continuous_scale="Blues",
                            zmin=0,
                            zmax=global_zmax,
                            x=["Pred Legit", "Pred Fraud"],
                            y=["Actual Legit", "Actual Fraud"]
                        )
                
                        fig_ai_cm.update_layout(
                            template=PLOT_THEME,
                            height=350
                        )
                
                        st.plotly_chart(
                            fig_ai_cm,
                            use_container_width=True
                        )
                
                        st.info("""
                        Represents raw machine learning classification
                        capability independent of governance routing.
                        """)
                
                    # =====================================================
                    # COLUMN 3 — GOVERNANCE
                    # =====================================================
                
                    with col3:
                
                        st.markdown("## 🛡️ Human-AI Governance")
                
                        gov1, gov2 = st.columns(2)
                
                        gov1.metric(
                            "Auto Approved",
                            governance_actions.count("Auto Approve")
                        )
                
                        gov2.metric(
                            "Human Reviews",
                            governance_actions.count("Human Review")
                        )
                
                        gov3, gov4 = st.columns(2)
                
                        gov3.metric(
                            "Auto Blocked",
                            governance_actions.count("Auto Block")
                        )
                
                        gov4.metric(
                            "Capture Rate",
                            f"{round((tp_g/(tp_g+fn_g))*100,2)}%"
                        )
                
                        # -------------------------
                        # DONUT
                        # -------------------------
                
                        gov_donut = pd.DataFrame({
                            "Category": [
                                "Auto Approve",
                                "Human Review",
                                "Auto Block"
                            ],
                            "Count": [
                                governance_actions.count("Auto Approve"),
                                governance_actions.count("Human Review"),
                                governance_actions.count("Auto Block")
                            ]
                        })
                
                        fig_gov_donut = px.pie(
                            gov_donut,
                            names="Category",
                            values="Count",
                            template=PLOT_THEME
                        )
                
                        fig_gov_donut.update_traces(
                            hole=0.60,
                            textinfo="percent+label",
                            textfont_size=14,
                            textposition="outside"
                        )
                
                        fig_gov_donut.update_layout(
                            height=320,
                            showlegend=True,
                            uniformtext_minsize=12,
                            uniformtext_mode='hide'
                        )
                
                        st.plotly_chart(
                            fig_gov_donut,
                            use_container_width=True
                        )
                
                        # -------------------------
                        # MATRIX
                        # -------------------------
                
                        fig_gov_cm = px.imshow(
                            gov_cm,
                            text_auto=True,
                            color_continuous_scale="Blues",
                            zmin=0,
                            zmax=global_zmax,
                            x=[
                                "Auto Approve",
                                "Human Review",
                                "Auto Block"
                            ],
                            
                            y=[
                                "Actual Legit",
                                "Actual Fraud"
                            ]
                        )
                
                        fig_gov_cm.update_layout(
                            template=PLOT_THEME,
                            height=350
                        )
                
                        st.plotly_chart(
                            fig_gov_cm,
                            use_container_width=True
                        )
                
                        st.info("""
                        Represents operational governance outcomes
                        after threshold-based routing decisions.
                        """)
                
                    st.divider()


                    # =====================================================
                    # RISK ZONE DISTRIBUTION
                    # =====================================================
                    
                    st.subheader("🎯 Governance Risk Zone Distribution")
                    
                    risk_zone_df = pd.DataFrame({
                    
                        "Risk Zone": [
                    
                            "Low Risk",
                            "Medium Risk",
                            "High Risk"
                        ],
                    
                        "Transactions": [
                    
                            (probs < approve_threshold).sum(),
                    
                            (
                                (probs >= approve_threshold)
                                &
                                (probs < block_threshold)
                            ).sum(),
                    
                            (probs >= block_threshold).sum()
                        ]
                    })
                    
                    fig = px.bar(
                    
                        risk_zone_df,
                    
                        x="Risk Zone",
                        y="Transactions",
                    
                        template=PLOT_THEME,
                    
                        text="Transactions"
                    )
                    
                    fig.update_layout(
                        height=450
                    )
                    
                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )
                    
                    st.info("""
                    
                    Transactions are operationally separated into:
                    • low-risk automated approvals
                    • medium-risk human review cases
                    • high-risk automated fraud blocks
                    
                    This creates a risk-based human-AI governance workflow.
                    
                    """)
                
                    # =====================================================
                    # MASTER TRANSACTION TABLE
                    # =====================================================
                
                    st.subheader("🔎 Unified Transaction Intelligence Table")
                
                    f1, f2, f3, f4 = st.columns(4)
                
                    with f1:
                
                        reality_filter = st.multiselect(
                            "Actual Reality",
                            comparison_df["Actual Reality"].unique(),
                            default=comparison_df["Actual Reality"].unique()
                        )
                
                    with f2:
                
                        ai_filter = st.multiselect(
                            "AI Prediction",
                            comparison_df["AI Prediction"].unique(),
                            default=comparison_df["AI Prediction"].unique()
                        )
                
                    with f3:
                
                        gov_filter = st.multiselect(
                            "Governance Decision",
                            comparison_df["Governance Decision"].unique(),
                            default=comparison_df["Governance Decision"].unique()
                        )
                
                    with f4:
                
                        outcome_filter = st.multiselect(
                            "Outcome",
                            comparison_df["Outcome"].unique(),
                            default=comparison_df["Outcome"].unique()
                        )
                
                    prob_range = st.slider(
                        "Fraud Probability Range",
                        0.0,
                        1.0,
                        (0.0, 1.0)
                    )
                
                    search_id = st.text_input(
                        "Search Transaction ID"
                    )
                
                    filtered_df = comparison_df[
                
                        (comparison_df["Actual Reality"].isin(reality_filter))
                        &
                        (comparison_df["AI Prediction"].isin(ai_filter))
                        &
                        (comparison_df["Governance Decision"].isin(gov_filter))
                        &
                        (comparison_df["Outcome"].isin(outcome_filter))
                        &
                        (
                            comparison_df["Fraud Probability"]
                            >= prob_range[0]
                        )
                        &
                        (
                            comparison_df["Fraud Probability"]
                            <= prob_range[1]
                        )
                
                    ]
                
                    if search_id != "":
                
                        try:
                
                            filtered_df = filtered_df[
                                filtered_df["Transaction ID"]
                                == int(search_id)
                            ]
                
                        except:
                            pass
                
                    st.dataframe(
                        filtered_df,
                        use_container_width=True,
                        height=700
                    )
                
                    st.divider()
                
                    # =====================================================
                    # SMART GOVERNANCE INSIGHTS
                    # =====================================================
                
                    st.subheader("🧠 Governance Intelligence Insights")
                
                    fraud_intercepted = (
                        comparison_df["Outcome"]
                        == "Fraud Intercepted"
                    ).sum()
                
                    missed_fraud = (
                        comparison_df["Outcome"]
                        == "Missed Fraud"
                    ).sum()
                
                    false_positive = (
                        comparison_df["Outcome"]
                        == "False Positive"
                    ).sum()
                
                    review_count = governance_actions.count(
                        "Human Review"
                    )
                
                    st.success(f"""
                    • {fraud_intercepted} fraud cases were successfully intercepted.
                
                    • {missed_fraud} fraud cases bypassed governance controls.
                
                    • {false_positive} legitimate transactions experienced operational friction.
                
                    • {review_count} transactions required human analyst escalation.
                
                    • Current governance thresholds balance fraud capture,
                      analyst workload, and automation efficiency.
                    """)  

                    # =====================================================
                    # GOVERNANCE PERMUTATION EXPERIMENT
                    # =====================================================

                    from governance.routing import route_predictions

                    st.write(
                        "## ⚖ Governance Threshold Experiment"
                    )

                    experiment_results=[]

                    approve_range=np.arange(
                        0.05,
                        0.6,
                        0.05
                    )

                    block_range=np.arange(
                        0.50,
                        0.95,
                        0.05
                    )


                    for exp_approve_threshold in approve_range:

                        for exp_block_threshold in block_range:

                            # ------------------------
                            # VALID CONFIGURATIONS ONLY
                            # ------------------------

                            if exp_approve_threshold >= exp_block_threshold:

                                continue


                            routed=route_predictions(
                                probs,
                                exp_approve_threshold,
                                exp_block_threshold
                            )


                            routed=np.array(routed)


                            auto_approve=(

                                routed=="Auto Approve"

                            )


                            human_review=(

                                routed=="Human Review"

                            )


                            auto_block=(

                                routed=="Auto Block"

                            )


                            approve_count=int(
                                auto_approve.sum()
                            )


                            review_count=int(
                                human_review.sum()
                            )


                            block_count=int(
                                auto_block.sum()
                            )


                            fraud_capture=int(

                                (

                                    auto_block

                                    &

                                    (y_test==1)

                                ).sum()

                            )


                            total_fraud=max(

                                int(

                                    (y_test==1).sum()

                                ),

                                1

                            )


                            capture_rate=round(

                                fraud_capture/
                                total_fraud,

                                4

                            )


                            false_positive=int(

                                (

                                    auto_block

                                    &

                                    (y_test==0)

                                ).sum()

                            )


                            experiment_results.append({

                                "Approve":
                                round(
                                    exp_approve_threshold,
                                    2
                                ),

                                "Block":
                                round(
                                    exp_block_threshold,
                                    2
                                ),

                                "Human Reviews":

                                review_count,

                                "Auto Approvals":

                                approve_count,

                                "Auto Blocks":

                                block_count,

                                "Capture Rate":

                                capture_rate,

                                "False Positives":

                                false_positive

                            })


                    tradeoff_df=pd.DataFrame(
                        experiment_results
                    )

                    st.dataframe(
                        tradeoff_df,
                        use_container_width=True
                    )
 

                    # =====================================================
                    # GOVERNANCE POLICY EXPLORER
                    # =====================================================

                    st.subheader(
                        "📊 Governance Policy Explorer"
                    )

                    st.info(
                    """
                    Select a governance policy to view its complete
                    operational profile and performance behaviour.
                    """
                    )

                    # ---------------------------------------
                    # Create configuration labels
                    # ---------------------------------------

                    tradeoff_df["Configuration"]=(

                        "A:"+

                        tradeoff_df["Approve"].astype(str)

                        +

                        " | B:"+

                        tradeoff_df["Block"].astype(str)

                    )

                    # ---------------------------------------
                    # Policy selection
                    # ---------------------------------------

                    selected_config=st.selectbox(

                        "Select Governance Policy",

                        tradeoff_df["Configuration"]

                    )

                    selected_row=tradeoff_df[

                        tradeoff_df["Configuration"]

                        ==

                        selected_config

                    ]

                    row=selected_row.iloc[0]


                    # =====================================================
                    # GOVERNANCE POLICY PROFILE
                    # =====================================================

                    st.subheader(
                        "📡 Governance Policy Profile"
                    )

                    row = selected_row.iloc[0]

                    categories=[

                        f"Human Reviews\n({int(row['Human Reviews']):,})",

                        f"Auto Approvals\n({int(row['Auto Approvals']):,})",

                        f"Auto Blocks\n({int(row['Auto Blocks']):,})",

                        f"Capture Rate\n({round(row['Capture Rate']*100,1)}%)",

                        f"False Positives\n({int(row['False Positives']):,})"

                    ]

                    # Normalized only for drawing shape
                    values=[

                        row["Human Reviews"]/
                        tradeoff_df["Human Reviews"].max(),

                        row["Auto Approvals"]/
                        tradeoff_df["Auto Approvals"].max(),

                        row["Auto Blocks"]/
                        tradeoff_df["Auto Blocks"].max(),

                        row["Capture Rate"],

                        row["False Positives"]/
                        tradeoff_df["False Positives"].max()

                    ]

                    values.append(values[0])
                    categories.append(categories[0])

                    fig=go.Figure()

                    fig.add_trace(

                        go.Scatterpolar(

                            r=values,

                            theta=categories,

                            fill='toself',

                            line=dict(width=3),

                            name=selected_config

                        )

                    )

                    fig.update_layout(

                        polar=dict(

                            radialaxis=dict(

                                visible=True,
                                range=[0,1]

                            )

                        ),

                        height=650,

                        showlegend=False

                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )


                    # =====================================================
                    # EXACT METRICS
                    # =====================================================

                    st.subheader(
                        "📌 Selected Policy Results"
                    )

                    c1,c2,c3=st.columns(3)

                    c1.metric(

                        "Human Reviews",

                        f"{int(row['Human Reviews']):,}"

                    )

                    c2.metric(

                        "Auto Approvals",

                        f"{int(row['Auto Approvals']):,}"

                    )

                    c3.metric(

                        "Auto Blocks",

                        f"{int(row['Auto Blocks']):,}"

                    )


                    c1,c2=st.columns(2)

                    c1.metric(

                        "Capture Rate",

                        f"{row['Capture Rate']:.2%}"

                    )

                    c2.metric(

                        "False Positives",

                        int(row["False Positives"])

                    )
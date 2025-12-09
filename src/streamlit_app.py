"""
Streamlit Application for Otto Product Classification
Interactive UI for single and batch predictions with confidence scores
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import time
from datetime import datetime
import io
import plotly.graph_objects as go
import plotly.express as px

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Otto Product Classification",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Model Loading Functions
# ============================================================================

@st.cache_resource
def load_model():
    """Load the trained XGBoost model"""
    try:
        # Get absolute path to model file
        script_dir = Path(__file__).parent.absolute()
        model_path = script_dir.parent / "model" / "xgboost_optimized_model.pkl"
        
        if not model_path.exists():
            st.error(f"❌ Model file not found")
            st.error(f"Looking at: {model_path}")
            st.error(f"Script dir: {script_dir}")
            return None
            
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None


@st.cache_resource
def load_label_encoder():
    """Load the label encoder"""
    try:
        # Get absolute path to encoder file
        script_dir = Path(__file__).parent.absolute()
        encoder_path = script_dir.parent / "model" / "label_encoder.pkl"
        
        if not encoder_path.exists():
            st.error(f"❌ Encoder file not found")
            st.error(f"Looking at: {encoder_path}")
            st.error(f"Script dir: {script_dir}")
            return None
            
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)
        
        return encoder
    except Exception as e:
        st.error(f"❌ Error loading encoder: {str(e)}")
        return None


# ============================================================================
# Prediction Functions
# ============================================================================

def validate_features(features):
    """Validate feature inputs"""
    if len(features) != 93:
        return False, f"Expected exactly 93 features, got {len(features)}"
    if any(x < 0 for x in features):
        return False, "All features must be non-negative"
    return True, "Valid"


def make_single_prediction(model, label_encoder, features):
    """Make a single prediction with confidence scores"""
    try:
        # Convert to numpy array
        X = np.array(features).reshape(1, -1)
        
        # Get prediction
        start_time = time.time()
        y_pred = model.predict(X)[0]
        y_pred_proba = model.predict_proba(X)[0]
        end_time = time.time()
        
        # Get predicted class name
        predicted_class = label_encoder.inverse_transform([y_pred])[0]
        
        # Get confidence (max probability)
        confidence = float(np.max(y_pred_proba))
        
        # Get all class probabilities
        all_probabilities = {
            label_encoder.inverse_transform([i])[0]: float(prob)
            for i, prob in enumerate(y_pred_proba)
        }
        
        # Sort probabilities by value
        all_probabilities = dict(sorted(
            all_probabilities.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        prediction_time = (end_time - start_time) * 1000  # ms
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_probabilities": all_probabilities,
            "prediction_time": prediction_time
        }
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None


def make_batch_predictions(model, label_encoder, instances):
    """Make batch predictions with confidence scores"""
    try:
        # Convert to numpy array
        X = np.array(instances)
        
        # Get predictions
        start_time = time.time()
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)
        end_time = time.time()
        
        total_time = (end_time - start_time) * 1000  # ms
        avg_time_per_instance = total_time / len(instances)
        
        # Process each prediction
        predictions = []
        for i in range(len(instances)):
            predicted_class = label_encoder.inverse_transform([y_pred[i]])[0]
            confidence = float(np.max(y_pred_proba[i]))
            
            all_probabilities = {
                label_encoder.inverse_transform([j])[0]: float(prob)
                for j, prob in enumerate(y_pred_proba[i])
            }
            
            all_probabilities = dict(sorted(
                all_probabilities.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            
            predictions.append({
                "predicted_class": predicted_class,
                "confidence": confidence,
                "all_probabilities": all_probabilities,
                "prediction_time": avg_time_per_instance
            })
        
        return predictions, total_time
        
    except Exception as e:
        st.error(f"Error in batch prediction: {str(e)}")
        return None, 0


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_confidence_gauge(confidence):
    """Create a gauge chart for confidence score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Score", 'font': {'size': 24}},
        delta = {'reference': 50, 'increasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 60], 'color': '#ffcccc'},
                {'range': [60, 80], 'color': '#fff4cc'},
                {'range': [80, 90], 'color': '#ccffcc'},
                {'range': [90, 100], 'color': '#ccffee'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def plot_probability_distribution(probabilities):
    """Create a bar chart for probability distribution"""
    classes = list(probabilities.keys())
    probs = list(probabilities.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=probs,
            text=[f'{p:.2%}' for p in probs],
            textposition='auto',
            marker_color=['#1f77b4' if i == 0 else '#7fcdbb' for i in range(len(classes))]
        )
    ])
    
    fig.update_layout(
        title="Probability Distribution Across All Classes",
        xaxis_title="Product Class",
        yaxis_title="Probability",
        yaxis_tickformat='.0%',
        height=400,
        showlegend=False
    )
    
    return fig


def get_confidence_interpretation(confidence):
    """Get interpretation of confidence score"""
    if confidence >= 0.9:
        return "🟢 Very High Confidence", "The model is very certain about this prediction."
    elif confidence >= 0.8:
        return "🟢 High Confidence", "The model is confident about this prediction."
    elif confidence >= 0.7:
        return "🟡 Moderate Confidence", "The model has reasonable confidence in this prediction."
    elif confidence >= 0.6:
        return "🟡 Low-Moderate Confidence", "The prediction should be verified."
    else:
        return "🔴 Low Confidence", "This prediction is uncertain and should be carefully reviewed."


# ============================================================================
# Main Application
# ============================================================================

def main():
    # Load model and encoder
    model = load_model()
    label_encoder = load_label_encoder()
    
    # Sidebar
    st.sidebar.title("📦 Otto Classification")
    st.sidebar.markdown("---")
    
    # Model Status
    st.sidebar.subheader("🔧 System Status")
    if model is not None and label_encoder is not None:
        st.sidebar.success("✅ Model Loaded")
        st.sidebar.success("✅ Encoder Loaded")
    else:
        st.sidebar.error("❌ Model Loading Failed")
        st.stop()
    
    # Model Info
    with st.sidebar.expander("📊 Model Information", expanded=False):
        st.write(f"**Model Type:** {type(model).__name__}")
        st.write(f"**Features:** 93")
        st.write(f"**Classes:** {len(label_encoder.classes_)}")
        st.write(f"**Classes:** {', '.join(label_encoder.classes_)}")
        
        st.markdown("**Hyperparameters:**")
        hyperparams = {
            "n_estimators": 473,
            "max_depth": 15,
            "learning_rate": 0.048,
            "subsample": 0.831,
            "colsample_bytree": 0.576,
            "gamma": 0.172,
            "min_child_weight": 3
        }
        for param, value in hyperparams.items():
            st.write(f"- {param}: {value}")
    
    st.sidebar.markdown("---")
    
    # Navigation
    st.sidebar.subheader("🧭 Navigation")
    page = st.sidebar.radio(
        "Select Mode:",
        ["🏠 Home", "🔮 Single Prediction", "📊 Batch Prediction", "📁 CSV Upload", "📖 Help"]
    )
    
    # Main Content
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔮 Single Prediction":
        show_single_prediction_page(model, label_encoder)
    elif page == "📊 Batch Prediction":
        show_batch_prediction_page(model, label_encoder)
    elif page == "📁 CSV Upload":
        show_csv_upload_page(model, label_encoder)
    elif page == "📖 Help":
        show_help_page()


def show_home_page():
    """Display home page"""
    st.title("📦 Otto Product Classification System")
    st.markdown("### XGBoost-based Multi-Class Classification with Confidence Scores")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🔮 Single Prediction")
        st.info("Classify individual products by entering 93 feature values. Get instant predictions with confidence scores and probability distributions.")
    
    with col2:
        st.markdown("#### 📊 Batch Prediction")
        st.info("Process multiple products at once. Perfect for analyzing several products simultaneously with detailed results.")
    
    with col3:
        st.markdown("#### 📁 CSV Upload")
        st.info("Upload a CSV file with product features and download predictions with confidence scores for all products.")
    
    st.markdown("---")
    
    st.markdown("### 📊 Model Performance")
    st.success("""
    This model is trained on the Otto product dataset and optimized using Optuna hyperparameter tuning.
    It provides multi-class classification across 9 product categories with high accuracy.
    """)


def show_single_prediction_page(model, label_encoder):
    """Display single prediction page"""
    st.title("🔮 Single Product Prediction")
    st.markdown("Enter 93 feature values to classify a product.")
    
    st.markdown("---")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Manual Entry (Grid)", "Paste Values", "Random Sample"]
    )
    
    features = []
    
    if input_method == "Manual Entry (Grid)":
        st.markdown("#### Enter Feature Values (0-92)")
        
        # Create grid layout for feature inputs
        num_cols = 5
        num_features = 93
        
        for row in range(0, num_features, num_cols):
            cols = st.columns(num_cols)
            for idx, col in enumerate(cols):
                feat_idx = row + idx
                if feat_idx < num_features:
                    with col:
                        val = st.number_input(
                            f"F{feat_idx}",
                            min_value=0.0,
                            value=0.0,
                            step=1.0,
                            key=f"feat_{feat_idx}"
                        )
                        features.append(val)
    
    elif input_method == "Paste Values":
        st.markdown("#### Paste 93 comma-separated values")
        input_text = st.text_area(
            "Features (comma-separated):",
            height=150,
            placeholder="1.0, 0.0, 0.0, 2.0, 1.0, 3.0, ..."
        )
        
        if input_text:
            try:
                features = [float(x.strip()) for x in input_text.split(',')]
                if len(features) != 93:
                    st.warning(f"Expected 93 values, got {len(features)}")
            except ValueError:
                st.error("Invalid input. Please enter numeric values separated by commas.")
    
    elif input_method == "Random Sample":
        st.markdown("#### Generate Random Sample Data")
        
        # Initialize session state for random features
        if 'random_features' not in st.session_state:
            st.session_state.random_features = []
        
        if st.button("🎲 Generate Random Features"):
            st.session_state.random_features = np.random.randint(0, 10, 93).tolist()
            st.success(f"Generated {len(st.session_state.random_features)} random features")
        
        if st.session_state.random_features:
            st.code(", ".join([f"{x:.1f}" for x in st.session_state.random_features[:20]]) + ", ...")
            features = st.session_state.random_features
    
    st.markdown("---")
    
    # Predict button
    if st.button("🔮 Predict", type="primary", use_container_width=True):
        if len(features) != 93:
            st.error(f"Please provide exactly 93 features. Currently have {len(features)}.")
        else:
            # Validate features
            is_valid, message = validate_features(features)
            if not is_valid:
                st.error(message)
            else:
                # Make prediction
                with st.spinner("Making prediction..."):
                    result = make_single_prediction(model, label_encoder, features)
                
                if result:
                    st.success("✅ Prediction Complete!")
                    
                    # Display results
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown("### 🎯 Result")
                        st.markdown(f"#### Predicted Class: **{result['predicted_class']}**")
                        st.markdown(f"**Confidence:** {result['confidence']:.2%}")
                        
                        # Confidence interpretation
                        interpretation, description = get_confidence_interpretation(result['confidence'])
                        st.markdown(f"**Level:** {interpretation}")
                        st.info(description)
                        
                        st.markdown(f"**Prediction Time:** {result['prediction_time']:.2f}ms")
                    
                    with col2:
                        # Confidence gauge
                        st.plotly_chart(
                            plot_confidence_gauge(result['confidence']),
                            use_container_width=True
                        )
                    
                    # Probability distribution
                    st.markdown("### 📊 Probability Distribution")
                    st.plotly_chart(
                        plot_probability_distribution(result['all_probabilities']),
                        use_container_width=True
                    )
                    
                    # Detailed probabilities table
                    with st.expander("📋 Detailed Probabilities", expanded=False):
                        prob_df = pd.DataFrame([
                            {"Class": cls, "Probability": prob, "Percentage": f"{prob:.4%}"}
                            for cls, prob in result['all_probabilities'].items()
                        ])
                        st.dataframe(prob_df, use_container_width=True, hide_index=True)


def show_batch_prediction_page(model, label_encoder):
    """Display batch prediction page"""
    st.title("📊 Batch Product Prediction")
    st.markdown("Enter multiple products (up to 10,000) for batch classification.")
    
    st.markdown("---")
    
    # Number of instances
    num_instances = st.number_input(
        "Number of instances:",
        min_value=1,
        max_value=10000,
        value=5,
        step=1
    )
    
    # Input method
    input_method = st.radio(
        "Choose input method:",
        ["Manual Entry", "Generate Random Data"]
    )
    
    instances = []
    
    if input_method == "Manual Entry":
        st.markdown("#### Enter features for each instance")
        st.info("Each row should contain 93 comma-separated values")
        
        for i in range(num_instances):
            with st.expander(f"Instance {i+1}", expanded=(i < 3)):
                input_text = st.text_area(
                    f"Features for instance {i+1}:",
                    height=100,
                    key=f"batch_{i}",
                    placeholder="1.0, 0.0, 0.0, 2.0, ..."
                )
                
                if input_text:
                    try:
                        features = [float(x.strip()) for x in input_text.split(',')]
                        if len(features) == 93:
                            instances.append(features)
                        else:
                            st.warning(f"Expected 93 values, got {len(features)}")
                    except ValueError:
                        st.error("Invalid input format")
    
    elif input_method == "Generate Random Data":
        # Initialize session state for random instances
        if 'random_instances' not in st.session_state:
            st.session_state.random_instances = []
        
        if st.button("🎲 Generate Random Instances"):
            st.session_state.random_instances = np.random.randint(0, 10, (num_instances, 93)).tolist()
            st.success(f"Generated {len(st.session_state.random_instances)} random instances")
        
        if st.session_state.random_instances:
            instances = st.session_state.random_instances
    
    st.markdown("---")
    
    # Predict button
    if st.button("📊 Predict Batch", type="primary", use_container_width=True):
        if len(instances) == 0:
            st.error("Please provide at least one instance.")
        else:
            # Validate all instances
            all_valid = True
            for i, inst in enumerate(instances):
                is_valid, message = validate_features(inst)
                if not is_valid:
                    st.error(f"Instance {i+1}: {message}")
                    all_valid = False
                    break
            
            if all_valid:
                # Make predictions
                with st.spinner(f"Making predictions for {len(instances)} instances..."):
                    predictions, total_time = make_batch_predictions(model, label_encoder, instances)
                
                if predictions:
                    st.success(f"✅ Batch Prediction Complete! Processed {len(predictions)} instances in {total_time:.2f}ms")
                    
                    # Summary statistics
                    st.markdown("### 📈 Summary Statistics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    confidences = [p['confidence'] for p in predictions]
                    
                    with col1:
                        st.metric("Total Instances", len(predictions))
                    with col2:
                        st.metric("Avg Confidence", f"{np.mean(confidences):.2%}")
                    with col3:
                        st.metric("Min Confidence", f"{np.min(confidences):.2%}")
                    with col4:
                        st.metric("Max Confidence", f"{np.max(confidences):.2%}")
                    
                    # Results table
                    st.markdown("### 📋 Prediction Results")
                    
                    results_df = pd.DataFrame([
                        {
                            "Instance": i+1,
                            "Predicted Class": p['predicted_class'],
                            "Confidence": f"{p['confidence']:.4f}",
                            "Confidence %": f"{p['confidence']:.2%}",
                            "Top 2nd Class": list(p['all_probabilities'].keys())[1],
                            "2nd Probability": f"{list(p['all_probabilities'].values())[1]:.4f}"
                        }
                        for i, p in enumerate(predictions)
                    ])
                    
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    # Class distribution
                    st.markdown("### 📊 Predicted Class Distribution")
                    class_counts = pd.Series([p['predicted_class'] for p in predictions]).value_counts()
                    
                    fig = px.bar(
                        x=class_counts.index,
                        y=class_counts.values,
                        labels={'x': 'Product Class', 'y': 'Count'},
                        title='Distribution of Predicted Classes'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Confidence distribution
                    st.markdown("### 📉 Confidence Distribution")
                    fig = px.histogram(
                        confidences,
                        nbins=20,
                        labels={'value': 'Confidence Score', 'count': 'Frequency'},
                        title='Distribution of Confidence Scores'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )


def show_csv_upload_page(model, label_encoder):
    """Display CSV upload page"""
    st.title("📁 CSV Upload & Prediction")
    st.markdown("Upload a CSV file with product features to get predictions for all products.")
    
    st.markdown("---")
    
    # File format info
    with st.expander("📖 CSV Format Requirements", expanded=True):
        st.markdown("""
        **Required Format:**
        - CSV file with 93 or 94 columns
        - If 94 columns: First column should be `id`, followed by 93 feature columns
        - If 93 columns: Only features (no ID column)
        - All feature values must be non-negative numbers
        - No missing values allowed
        
        **Example:**
        ```
        id,feat_1,feat_2,feat_3,...,feat_93
        1,1.0,0.0,0.0,...,2.0
        2,0.0,1.0,2.0,...,1.0
        ```
        """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with 93 feature columns"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
            
            # Show preview
            st.markdown("### 📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Validate
            if len(df.columns) not in [93, 94]:
                st.error(f"Expected 93 or 94 columns, got {len(df.columns)}")
            else:
                # Extract features
                if 'id' in df.columns:
                    ids = df['id'].values
                    X = df.drop('id', axis=1).values
                else:
                    ids = np.arange(len(df))
                    X = df.values
                
                st.info(f"Found {len(df)} instances with {X.shape[1]} features each")
                
                # Predict button
                if st.button("🚀 Generate Predictions", type="primary", use_container_width=True):
                    # Validate all features
                    if np.any(X < 0):
                        st.error("All feature values must be non-negative")
                    else:
                        # Make predictions
                        with st.spinner(f"Processing {len(df)} instances..."):
                            start_time = time.time()
                            y_pred = model.predict(X)
                            y_pred_proba = model.predict_proba(X)
                            end_time = time.time()
                            
                            total_time = (end_time - start_time) * 1000
                        
                        st.success(f"✅ Predictions completed in {total_time:.2f}ms ({total_time/len(df):.2f}ms per instance)")
                        
                        # Create results dataframe
                        results = pd.DataFrame({
                            'id': ids,
                            'predicted_class': label_encoder.inverse_transform(y_pred),
                            'confidence': np.max(y_pred_proba, axis=1)
                        })
                        
                        # Add probability columns for each class
                        for i, class_name in enumerate(label_encoder.classes_):
                            results[f'prob_{class_name}'] = y_pred_proba[:, i]
                        
                        # Display results
                        st.markdown("### 📊 Prediction Results")
                        
                        # Summary statistics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Predictions", len(results))
                        with col2:
                            st.metric("Avg Confidence", f"{results['confidence'].mean():.2%}")
                        with col3:
                            st.metric("Min Confidence", f"{results['confidence'].min():.2%}")
                        with col4:
                            st.metric("Max Confidence", f"{results['confidence'].max():.2%}")
                        
                        # Results preview
                        st.markdown("#### Preview (First 20 rows)")
                        display_cols = ['id', 'predicted_class', 'confidence'] + [col for col in results.columns if col.startswith('prob_')]
                        st.dataframe(results[display_cols].head(20), use_container_width=True, hide_index=True)
                        
                        # Visualizations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Class distribution
                            st.markdown("#### Predicted Class Distribution")
                            class_counts = results['predicted_class'].value_counts()
                            fig = px.pie(
                                values=class_counts.values,
                                names=class_counts.index,
                                title='Distribution of Predicted Classes'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Confidence distribution
                            st.markdown("#### Confidence Score Distribution")
                            fig = px.histogram(
                                results['confidence'],
                                nbins=30,
                                labels={'value': 'Confidence Score'},
                                title='Distribution of Confidence Scores'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Download button
                        csv = results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions CSV",
                            data=csv,
                            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")


def show_help_page():
    """Display help page"""
    st.title("📖 Help & Documentation")
    
    st.markdown("---")
    
    st.markdown("### 🎯 About This Application")
    st.markdown("""
    This is an interactive web application for **Otto Product Classification** using a trained XGBoost model.
    The model classifies products into 9 different categories based on 93 features.
    """)
    
    st.markdown("---")
    
    st.markdown("### 📊 Features")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Single Prediction", "Batch Prediction", "CSV Upload", "Understanding Results"])
    
    with tab1:
        st.markdown("""
        #### 🔮 Single Prediction
        
        **Purpose:** Classify individual products one at a time.
        
        **How to use:**
        1. Select input method (Manual, Paste, or Random)
        2. Enter 93 feature values for your product
        3. Click "Predict" to get results
        4. View confidence scores and probability distributions
        
        **Best for:** Quick single product classifications
        """)
    
    with tab2:
        st.markdown("""
        #### 📊 Batch Prediction
        
        **Purpose:** Classify multiple products simultaneously (up to 10,000).
        
        **How to use:**
        1. Specify number of instances
        2. Enter features for each instance or generate random data
        3. Click "Predict Batch"
        4. View summary statistics and download results
        
        **Best for:** Processing multiple products at once
        """)
    
    with tab3:
        st.markdown("""
        #### 📁 CSV Upload
        
        **Purpose:** Process large datasets from CSV files.
        
        **How to use:**
        1. Prepare CSV with 93 features (optionally with ID column)
        2. Upload the CSV file
        3. Click "Generate Predictions"
        4. Download results with all probabilities
        
        **Best for:** Large-scale batch processing
        """)
    
    with tab4:
        st.markdown("""
        #### 📈 Understanding Results
        
        **Predicted Class:** The most likely product category
        
        **Confidence Score:** Probability of the predicted class (0-1)
        - 🟢 **0.9-1.0:** Very High Confidence
        - 🟢 **0.8-0.9:** High Confidence  
        - 🟡 **0.7-0.8:** Moderate Confidence
        - 🟡 **0.6-0.7:** Low-Moderate Confidence
        - 🔴 **0.0-0.6:** Low Confidence
        
        **All Probabilities:** Distribution across all 9 classes
        
        **Prediction Time:** Processing time in milliseconds
        """)
    
    st.markdown("---")
    
    st.markdown("### 🎓 Product Classes")
    st.markdown("""
    The model classifies products into the following 9 categories:
    
    - **Class_1** through **Class_9**: Different product categories from the Otto dataset
    
    Each prediction includes the probability distribution across all 9 classes.
    """)
    
    st.markdown("---")
    
    st.markdown("### 🛠️ Model Information")
    st.markdown("""
    - **Algorithm:** XGBoost (Extreme Gradient Boosting)
    - **Input Features:** 93 numeric features
    - **Output Classes:** 9 product categories
    - **Optimization:** Hyperparameter tuning with Optuna
    - **Performance:** Optimized for accuracy and speed
    """)
    
    st.markdown("---")
    
    st.markdown("### ❓ FAQ")
    
    with st.expander("What are the input features?"):
        st.markdown("""
        The model expects 93 numeric features representing various product attributes.
        All feature values must be non-negative numbers (≥ 0).
        """)
    
    with st.expander("How is confidence calculated?"):
        st.markdown("""
        Confidence is the maximum probability among all class predictions.
        It represents how certain the model is about its prediction.
        Higher confidence indicates more certainty.
        """)
    
    with st.expander("What if my confidence is low?"):
        st.markdown("""
        Low confidence (< 0.6) suggests the model is uncertain. Consider:
        - Verifying input data quality
        - Checking if the product fits known categories
        - Looking at the top 2-3 predicted classes
        - Manually reviewing the prediction
        """)
    
    with st.expander("Can I process data programmatically?"):
        st.markdown("""
        Yes! This application has a FastAPI equivalent available.
        Use the API for programmatic access with REST endpoints.
        See the API documentation for more details.
        """)
    
    st.markdown("---")
    
    st.markdown("### 📞 Support")
    st.info("""
    For issues, questions, or feature requests, please contact the development team
    or refer to the project documentation.
    """)


# ============================================================================
# Run Application
# ============================================================================

if __name__ == "__main__":
    main()

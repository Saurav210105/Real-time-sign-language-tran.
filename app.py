import os
import json
import time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
import streamlit as st

# Version-safe Streamlit rerun helper
def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

# Suppress TensorFlow log pollution
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Initialize MediaPipe Solutions
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Page configuration
st.set_page_config(
    page_title="Sign Language Translator & Trainer",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design CSS Injection
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');

/* Main app background */
.stApp {
    background-color: #0b0c10;
    color: #c5c6c7;
    font-family: 'Outfit', sans-serif;
}

/* Glassmorphism Card Container */
.card {
    background: rgba(31, 40, 51, 0.45);
    border: 1px solid rgba(102, 252, 241, 0.15);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 15px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
    backdrop-filter: blur(5px);
    transition: border-color 0.2s ease-in-out;
}

.card:hover {
    border-color: rgba(102, 252, 241, 0.45);
}

/* Logo Design */
.logo-container {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
}
.logo-icon {
    font-size: 2.2rem;
}
.logo-text {
    font-size: 1.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #66fcf1, #45a29e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Status Badges */
.status-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.badge-success {
    background: rgba(102, 252, 241, 0.15);
    color: #66fcf1;
    border: 1px solid rgba(102, 252, 241, 0.3);
}
.badge-danger {
    background: rgba(255, 0, 122, 0.15);
    color: #FF007A;
    border: 1px solid rgba(255, 0, 122, 0.3);
}
.badge-warning {
    background: rgba(255, 193, 7, 0.15);
    color: #ffc107;
    border: 1px solid rgba(255, 193, 7, 0.3);
}

/* Display Boards */
.text-board {
    background: rgba(11, 12, 16, 0.6);
    border-radius: 8px;
    padding: 12px;
    margin-top: 10px;
    border-left: 4px solid #FF007A;
}
.history-board {
    background: rgba(11, 12, 16, 0.6);
    border-radius: 8px;
    padding: 12px;
    margin-top: 10px;
    border-left: 4px solid #66fcf1;
}
.board-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8b9bb4;
    margin-bottom: 2px;
}
.board-content-main {
    font-size: 1.8rem;
    font-weight: 800;
    color: #ffffff;
}
.board-content-history {
    font-size: 1.2rem;
    font-weight: 600;
    color: #66fcf1;
    min-height: 30px;
}

/* Console details */
.console-block {
    background: #000000;
    font-family: monospace;
    font-size: 0.8rem;
    padding: 10px;
    border-radius: 6px;
    border: 1px solid rgba(255,255,255,0.1);
    color: #66fcf1;
    overflow-y: auto;
    max-height: 200px;
    white-space: pre-wrap;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 1. Coordinate Normalization (Matches app.js original logic)
# ----------------------------------------------------
def normalize_landmarks(landmarks):
    if not landmarks or len(landmarks) != 21:
        return None
    
    wrist = landmarks[0]
    
    # Translate wrist to coordinate (0,0,0)
    translated = []
    for lm in landmarks:
        translated.append({
            'x': lm.x - wrist.x,
            'y': lm.y - wrist.y,
            'z': lm.z - wrist.z
        })
        
    # Scale relative to wrist-to-middle-MCP (landmark 9) distance
    mcp = translated[9]
    scale = (mcp['x']**2 + mcp['y']**2 + mcp['z']**2)**0.5 or 1.0
    
    # Flatten to 63-element feature vector
    flattened = []
    for i in range(21):
        flattened.append(translated[i]['x'] / scale)
        flattened.append(translated[i]['y'] / scale)
        flattened.append(translated[i]['z'] / scale)
        
    return flattened

# ----------------------------------------------------
# 2. CNN-LSTM Model Trainer (Matches server.py original parameters)
# ----------------------------------------------------
def train_model_ui(dataset, progress_bar, status_text):
    status_text.text("Parsing dataset sequences...")
    sequence_length = 20
    
    # 1. Group consecutive samples into sequences
    sequences = []
    seq_labels = []
    
    has_seq_id = any('sequenceId' in s for s in dataset)
    
    if has_seq_id:
        from collections import defaultdict
        grouped = defaultdict(list)
        for s in dataset:
            seq_id = s.get('sequenceId', 'default')
            grouped[seq_id].append(s)
            
        for seq_id, seq_samples in grouped.items():
            features = [s['features'] for s in seq_samples]
            label = seq_samples[0]['label']
            if len(features) >= sequence_length:
                for i in range(len(features) - sequence_length + 1):
                    sequences.append(features[i:i+sequence_length])
                    seq_labels.append(label)
    else:
        # Group by blocks of 25 (legacy dataset support)
        current_label = None
        current_seq = []
        for s in dataset:
            label = s['label']
            features = s['features']
            if label != current_label or len(current_seq) >= 25:
                if len(current_seq) >= sequence_length:
                    for i in range(len(current_seq) - sequence_length + 1):
                        sequences.append(current_seq[i:i+sequence_length])
                        seq_labels.append(current_label)
                current_label = label
                current_seq = []
            current_seq.append(features)
        if len(current_seq) >= sequence_length:
            for i in range(len(current_seq) - sequence_length + 1):
                sequences.append(current_seq[i:i+sequence_length])
                seq_labels.append(current_label)

    if not sequences:
        status_text.text(f"Error: Insufficient data. Please record sequences of at least {sequence_length} frames.")
        return None, []
        
    unique_labels = sorted(list(set(seq_labels)))
    if len(unique_labels) < 2:
        status_text.text("Error: You must record at least 2 distinct gestures to train the model.")
        return None, []
        
    # 2. Data Augmentation (Coordinate Jittering)
    status_text.text(f"Augmenting dataset (Jitter scale: 5x) - Base seqs: {len(sequences)}")
    augmented_sequences = []
    augmented_labels = []
    
    for seq, label in zip(sequences, seq_labels):
        augmented_sequences.append(seq)
        augmented_labels.append(label)
        
        # Add copies with Gaussian coordinate noise (jitter)
        for noise_scale in [0.005, 0.010, 0.015, 0.020]:
            noise = np.random.normal(0, noise_scale, size=np.array(seq).shape)
            jittered = np.array(seq) + noise
            augmented_sequences.append(jittered.tolist())
            augmented_labels.append(label)
            
    # 3. Label Encoding
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    y_indices = [label_to_index[l] for l in augmented_labels]
    
    X = np.array(augmented_sequences, dtype=np.float32)
    y = to_categorical(y_indices, num_classes=len(unique_labels))
    
    # Shuffle dataset
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # 4. Build Conv1D-LSTM model
    num_classes = len(unique_labels)
    epochs = 30
    batch_size = 16
    
    tf_model = Sequential([
        Input(shape=(sequence_length, 63)),
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same',
               kernel_regularizer=tf.keras.regularizers.l2(0.002)),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same',
               kernel_regularizer=tf.keras.regularizers.l2(0.002)),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        LSTM(48, dropout=0.2, recurrent_dropout=0.2,
             kernel_regularizer=tf.keras.regularizers.l2(0.002)),
             
        Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.002)),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    
    tf_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    class StreamlitCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            percent = (epoch + 1) / epochs
            progress_bar.progress(percent)
            status_text.text(
                f"Training CNN-LSTM Model...\n"
                f"Epoch {epoch+1}/{epochs}\n"
                f"Loss: {logs.get('loss', 0):.4f} - Accuracy: {logs.get('accuracy', 0):.4f}\n"
                f"Val Loss: {logs.get('val_loss', 0):.4f} - Val Accuracy: {logs.get('val_accuracy', 0):.4f}"
            )
            
    # Fit model
    tf_model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=[StreamlitCallback()],
        verbose=0
    )
    
    # Save model and labels
    tf_model.save("sign_model.h5")
    with open("labels.json", "w") as f:
        json.dump(unique_labels, f)
        
    return tf_model, unique_labels

# ----------------------------------------------------
# 3. Session State Initialization
# ----------------------------------------------------
if 'dataset' not in st.session_state:
    if os.path.exists("dataset.json"):
        with open("dataset.json", "r") as f:
            try:
                st.session_state.dataset = json.load(f)
            except:
                st.session_state.dataset = []
    else:
        st.session_state.dataset = []
        
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.labels = []
    if os.path.exists("sign_model.h5") and os.path.exists("labels.json"):
        try:
            st.session_state.model = load_model("sign_model.h5")
            with open("labels.json", "r") as f:
                st.session_state.labels = json.load(f)
        except Exception as e:
            print("Error loading model:", e)

if 'sentence_history' not in st.session_state:
    st.session_state.sentence_history = []
if 'last_typed_word' not in st.session_state:
    st.session_state.last_typed_word = ""
if 'current_stabilized_word' not in st.session_state:
    st.session_state.current_stabilized_word = ""
if 'stable_counter' not in st.session_state:
    st.session_state.stable_counter = 0
if 'sequence_buffer' not in st.session_state:
    st.session_state.sequence_buffer = []
if 'recording_state' not in st.session_state:
    st.session_state.recording_state = None
if 'recording_label' not in st.session_state:
    st.session_state.recording_label = ""
if 'recording_countdown_start' not in st.session_state:
    st.session_state.recording_countdown_start = 0.0
if 'recording_samples_captured' not in st.session_state:
    st.session_state.recording_samples_captured = 0
if 'recording_last_sample_time' not in st.session_state:
    st.session_state.recording_last_sample_time = 0.0
if 'recording_seq_id' not in st.session_state:
    st.session_state.recording_seq_id = ""
if 'speak_text' not in st.session_state:
    st.session_state.speak_text = None
if 'trigger_training' not in st.session_state:
    st.session_state.trigger_training = False

# ----------------------------------------------------
# 4. Global Action Handler (Training Interception)
# ----------------------------------------------------
if st.session_state.trigger_training:
    # Release camera resource before intensive training
    if 'cap' in st.session_state:
        st.session_state.cap.release()
        del st.session_state.cap
        
    st.markdown('<div class="card"><h2>Training CNN-LSTM Model</h2>', unsafe_allow_html=True)
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    
    new_model, new_labels = train_model_ui(st.session_state.dataset, progress_bar, status_text)
    
    if new_model is not None:
        st.session_state.model = new_model
        st.session_state.labels = new_labels
        st.success("CNN-LSTM Model Trained and Loaded successfully!")
    else:
        st.error("Model training failed.")
        
    st.session_state.trigger_training = False
    time.sleep(2.5)
    safe_rerun()

# ----------------------------------------------------
# 5. Sidebar Navigation Layout
# ----------------------------------------------------
st.sidebar.markdown(
    '<div class="logo-container">'
    '<span class="logo-icon">🤟</span>'
    '<span class="logo-text">SIGN TRANSLATOR</span>'
    '</div>',
    unsafe_allow_html=True
)

page = st.sidebar.radio("Navigation", ["Translate", "Model Trainer", "Dataset Manager"])
camera_enabled = st.sidebar.checkbox("Start Webcam", value=False)

# Auto close/release capture if user untoggles sidebar box
if not camera_enabled and 'cap' in st.session_state:
    st.session_state.cap.release()
    del st.session_state.cap

# ----------------------------------------------------
# 6. Page Execution Switch
# ----------------------------------------------------
if page == "Translate":
    st.markdown(
        '<div>'
        '<h1>Real-time Gesture Translate Mode</h1>'
        '<p style="color:#8b9bb4;">Perform hand gestures in front of the camera. Hold a pose steady for ~0.5 seconds (15 frames) to write it down.</p>'
        '</div>',
        unsafe_allow_html=True
    )
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown('<div class="card"><h3>Webcam Feed</h3></div>', unsafe_allow_html=True)
        frame_placeholder = st.empty()
        if not camera_enabled:
            frame_placeholder.info("Click 'Start Webcam' in the sidebar to enable live translation feed.")
            
    with col2:
        status_placeholder = st.empty()
        prediction_placeholder = st.empty()
        history_placeholder = st.empty()
        
        # Action Buttons
        if st.button("🔊 Speak Text", key="btn_speak"):
            st.session_state.speak_text = "".join(st.session_state.sentence_history) or "No composed text"
            
        if st.button("🗑️ Clear Output", key="btn_clear"):
            st.session_state.sentence_history.clear()
            st.session_state.last_typed_word = ""
            st.session_state.current_stabilized_word = ""
            st.session_state.stable_counter = 0
            
        auto_speak = st.checkbox("Auto-Speak on detect", value=True, key="chk_auto_speak")
        
        # Audio / TTS component anchor
        tts_placeholder = st.empty()

elif page == "Model Trainer":
    st.markdown(
        '<div>'
        '<h1>Gesture Model Trainer</h1>'
        '<p style="color:#8b9bb4;">Record training poses to custom labels. 25 sequence frames will be captured dynamically over 2.5 seconds.</p>'
        '</div>',
        unsafe_allow_html=True
    )
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown('<div class="card"><h3>Webcam Capture</h3></div>', unsafe_allow_html=True)
        frame_placeholder = st.empty()
        if not camera_enabled:
            frame_placeholder.info("Click 'Start Webcam' in the sidebar to activate capture camera.")
            
    with col2:
        trainer_ui_placeholder = st.empty()
        
        # Text input to record pose
        st.markdown('<div class="card"><h3>Record New Pose</h3></div>', unsafe_allow_html=True)
        label_input = st.text_input("Gesture Label (e.g. Hello, ThumbsUp, Space, Backspace)", placeholder="e.g. Namaste", key="label_in")
        
        btn_disabled = (st.session_state.recording_state is not None) or not label_input.strip() or not camera_enabled
        
        if st.button("🔴 Record Pose", disabled=btn_disabled, key="btn_rec"):
            st.session_state.recording_state = "countdown"
            st.session_state.recording_label = label_input.strip()
            st.session_state.recording_countdown_start = time.time()
            
        if not camera_enabled:
            st.warning("Please activate the camera in the sidebar to record.")

elif page == "Dataset Manager":
    st.markdown(
        '<div>'
        '<h1>Dataset Manager</h1>'
        '<p style="color:#8b9bb4;">Manage and inspect your recorded training classes.</p>'
        '</div>',
        unsafe_allow_html=True
    )
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Calculate counts
    counts = {}
    for sample in st.session_state.dataset:
        lbl = sample.get("label", "unknown")
        counts[lbl] = counts.get(lbl, 0) + 1
        
    st.subheader(f"Total Samples: {len(st.session_state.dataset)}")
    
    if not counts:
        st.info("No gestures recorded yet. Go to 'Model Trainer' page to record new poses.")
    else:
        # Table list of gesture labels
        st.markdown(
            '<table style="width:100%; border-collapse: collapse; margin-bottom: 20px;">'
            '<tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">'
            '<th style="text-align:left; padding:8px;">Gesture Label</th>'
            '<th style="text-align:left; padding:8px;">Sample Count</th>'
            '<th style="text-align:right; padding:8px;">Action</th>'
            '</tr>',
            unsafe_allow_html=True
        )
        
        for label, count in sorted(counts.items()):
            col_l, col_c, col_a = st.columns([2, 2, 1])
            with col_l:
                st.write(f"🏷️ **{label}**")
            with col_c:
                st.write(f"📁 {count} samples")
            with col_a:
                if st.button("Delete Class", key=f"del_{label}"):
                    st.session_state.dataset = [s for s in st.session_state.dataset if s.get("label") != label]
                    with open("dataset.json", "w") as f:
                        json.dump(st.session_state.dataset, f, indent=2)
                    st.session_state.trigger_training = True
                    safe_rerun()
                    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Actions Panel
    st.markdown('<div class="card"><h3>Dataset Import/Export</h3></div>', unsafe_allow_html=True)
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        dataset_str = json.dumps(st.session_state.dataset, indent=2)
        st.download_button(
            label="📥 Save & Download Dataset (dataset.json)",
            data=dataset_str,
            file_name="dataset.json",
            mime="application/json"
        )
        
    with col_d2:
        uploaded_file = st.file_uploader("📤 Load Dataset File", type=["json"])
        if uploaded_file is not None:
            try:
                loaded_data = json.load(uploaded_file)
                if isinstance(loaded_data, list) and all("features" in s and "label" in s for s in loaded_data):
                    st.session_state.dataset = loaded_data
                    with open("dataset.json", "w") as f:
                        json.dump(loaded_data, f, indent=2)
                    st.success("Dataset loaded successfully!")
                    st.session_state.trigger_training = True
                    time.sleep(1)
                    safe_rerun()
                else:
                    st.error("Invalid dataset structure. Missing features or labels.")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                
    st.markdown("---")
    if st.button("🗑️ Clear Entire Dataset"):
        st.session_state.dataset = []
        if os.path.exists("dataset.json"):
            os.remove("dataset.json")
        if os.path.exists("labels.json"):
            os.remove("labels.json")
        if os.path.exists("sign_model.h5"):
            os.remove("sign_model.h5")
        st.session_state.model = None
        st.session_state.labels = []
        st.success("Dataset deleted! Model reset.")
        time.sleep(1)
        safe_rerun()

# ----------------------------------------------------
# 7. Real-Time Camera Acquisition Loop
# ----------------------------------------------------
if camera_enabled:
    if 'cap' not in st.session_state or not st.session_state.cap.isOpened():
        st.session_state.cap = cv2.VideoCapture(0)
        
    cap = st.session_state.cap
    
    # Initialize MediaPipe Hands detector
    with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
        
        while camera_enabled:
            ret, frame = cap.read()
            if not ret:
                st.error("Webcam device disconnected or busy.")
                break
                
            # Flip horizontally for mirrored view
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            
            # Process hand landmarks
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            normalized_features = None
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    # Normalize landmarks
                    normalized_features = normalize_landmarks(hand_landmarks.landmark)
                    break # Single hand detection limit
                    
            # Mirror BGR back to RGB for streamlit
            rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Rendering loop logic
            if page == "Translate":
                # Prediction classification
                label_display = "-"
                conf_display = "0%"
                
                if normalized_features is not None:
                    st.session_state.sequence_buffer.append(normalized_features)
                    if len(st.session_state.sequence_buffer) > 20:
                        st.session_state.sequence_buffer.pop(0)
                        
                    if len(st.session_state.sequence_buffer) == 20:
                        if st.session_state.model is not None:
                            input_data = np.expand_dims(np.array(st.session_state.sequence_buffer, dtype=np.float32), axis=0)
                            predictions = st.session_state.model.predict(input_data, verbose=0)[0]
                            max_idx = np.argmax(predictions)
                            confidence = float(predictions[max_idx])
                            predicted_label = st.session_state.labels[max_idx]
                            
                            if predicted_label != "None" and confidence >= 0.6:
                                label_display = predicted_label
                                conf_display = f"{int(confidence * 100)}%"
                                
                                # Stabilization composition rules
                                if predicted_label == st.session_state.current_stabilized_word:
                                    st.session_state.stable_counter += 1
                                    if st.session_state.stable_counter == 15: # threshold (~0.5s)
                                        
                                        # Auto speak
                                        if auto_speak and predicted_label != st.session_state.last_typed_word:
                                            st.session_state.speak_text = predicted_label
                                            
                                        # Special command keys mapping
                                        if predicted_label.lower() == "space":
                                            if st.session_state.sentence_history and st.session_state.sentence_history[-1] != " ":
                                                st.session_state.sentence_history.append(" ")
                                        elif predicted_label.lower() == "backspace":
                                            if st.session_state.sentence_history:
                                                st.session_state.sentence_history.pop()
                                        elif predicted_label.lower() == "clear":
                                            st.session_state.sentence_history.clear()
                                        else:
                                            if predicted_label != st.session_state.last_typed_word:
                                                st.session_state.sentence_history.append(predicted_label)
                                                
                                        st.session_state.last_typed_word = predicted_label
                                else:
                                    st.session_state.current_stabilized_word = predicted_label
                                    st.session_state.stable_counter = 0
                            else:
                                label_display = "-"
                                conf_display = "0%"
                                st.session_state.stable_counter = 0
                                st.session_state.last_typed_word = ""
                        else:
                            label_display = "No Model Loaded"
                            conf_display = "0%"
                    else:
                        label_display = "Analyzing gesture..."
                        conf_display = f"{int(len(st.session_state.sequence_buffer) / 20 * 100)}%"
                else:
                    st.session_state.sequence_buffer.clear()
                    label_display = "-"
                    conf_display = "0%"
                    st.session_state.stable_counter = 0
                    st.session_state.last_typed_word = ""
                    
                # Update visual labels
                status_placeholder.markdown(
                    f'<div class="card">'
                    f'<h3>System Status</h3>'
                    f'<div style="display: flex; gap: 10px;">'
                    f'<span class="status-badge badge-success">Camera Connected</span>'
                    f'<span class="status-badge {"badge-success" if st.session_state.model else "badge-danger"}">'
                    f'{"CNN-LSTM Ready" if st.session_state.model else "Model Not Trained"}</span>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                badge_class = "badge-success" if label_display not in ["-", "No Model Loaded", "Analyzing gesture..."] else "badge-danger"
                if label_display == "Analyzing gesture...":
                    badge_class = "badge-warning"
                    
                prediction_placeholder.markdown(
                    f'<div class="card">'
                    f'<div style="display:flex; justify-content:space-between; align-items:center;">'
                    f'<h2>Current Sign</h2>'
                    f'<span class="status-badge {badge_class}" style="font-size:1.1rem; padding: 8px 16px;">{conf_display}</span>'
                    f'</div>'
                    f'<div class="text-board">'
                    f'<div class="board-label">DETECTED GESTURE</div>'
                    f'<div class="board-content-main">{label_display}</div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                history_placeholder.markdown(
                    f'<div class="card">'
                    f'<div class="history-board">'
                    f'<div class="board-label">COMPOSED SENTENCE</div>'
                    f'<div class="board-content-history">{"".join(st.session_state.sentence_history) or "-"}</div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                # Speak trigger execution
                if st.session_state.speak_text is not None:
                    with tts_placeholder:
                        text_to_speak = st.session_state.speak_text
                        st.components.v1.html(f"""
                            <script>
                            const utterance = new SpeechSynthesisUtterance("{text_to_speak}");
                            utterance.lang = "en-US";
                            window.speechSynthesis.speak(utterance);
                            </script>
                        """, height=0)
                    st.session_state.speak_text = None
                    
            elif page == "Model Trainer":
                status_text = ""
                
                if st.session_state.recording_state == "countdown":
                    elapsed = time.time() - st.session_state.recording_countdown_start
                    remaining = 3.0 - elapsed
                    if remaining > 0:
                        status_text = f"Countdown: {int(remaining)+1}..."
                        # Overlay text onto video frame
                        cv2.putText(rgb_display, str(int(remaining)+1), (w//2 - 20, h//2), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (102, 252, 241), 8)
                        cv2.putText(rgb_display, "HOLD HAND STEADY", (w//2 - 150, h//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (102, 252, 241), 2)
                    else:
                        st.session_state.recording_state = "capturing"
                        st.session_state.recording_samples_captured = 0
                        st.session_state.recording_last_sample_time = time.time()
                        st.session_state.recording_seq_id = str(int(time.time() * 1000))
                        
                elif st.session_state.recording_state == "capturing":
                    captured = st.session_state.recording_samples_captured
                    status_text = f"Recording pose: {captured}/25 samples"
                    
                    # Progress overlay on video frame
                    cv2.putText(rgb_display, f"Capturing: {captured}/25", (40, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 122), 3)
                    
                    if normalized_features is not None:
                        now = time.time()
                        if now - st.session_state.recording_last_sample_time >= 0.1: # 100ms
                            st.session_state.dataset.append({
                                "features": normalized_features,
                                "label": st.session_state.recording_label,
                                "sequenceId": st.session_state.recording_seq_id
                            })
                            st.session_state.recording_samples_captured += 1
                            st.session_state.recording_last_sample_time = now
                            
                            if st.session_state.recording_samples_captured >= 25:
                                # Save dataset
                                with open("dataset.json", "w") as f:
                                    json.dump(st.session_state.dataset, f, indent=2)
                                st.session_state.recording_state = None
                                st.session_state.trigger_training = True
                                # Exits loop automatically on rerun
                                safe_rerun()
                    else:
                        # Warning text if hand disappears
                        cv2.putText(rgb_display, "SHOW HAND IN CAMERA!", (w//2 - 180, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
                
                # Render trainer sidebar/box status
                trainer_ui_placeholder.markdown(
                    f'<div class="card">'
                    f'<h3>Capture Status</h3>'
                    f'<p><strong>Current Mode:</strong> {status_text if status_text else "Webcam Active - Ready"}</p>'
                    f'<p>Total Dataset Samples: <span class="status-badge badge-success">{len(st.session_state.dataset)}</span></p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
            # Render video image placeholder
            frame_placeholder.image(rgb_display, channels="RGB", use_column_width=True)
            
            # Tiny sleep to yield thread execution
            time.sleep(0.01)

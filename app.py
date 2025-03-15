import os
import cv2
import PIL
import numpy as np
import google.generativeai as genai
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from mediapipe.python.solutions import hands, drawing_utils
from dotenv import load_dotenv
from warnings import filterwarnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array
import time

# Filter warnings
filterwarnings(action='ignore')

class Calculator:
    def streamlit_config(self):
        """Configure Streamlit page layout and appearance"""
        # Page configuration
        st.set_page_config(
            page_title='AI-Powered Virtual Calculator',
            page_icon='🧮',
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better appearance
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 600;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 1rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid #f0f2f6;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #424242;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        .result-header {
            font-size: 1.3rem;
            font-weight: 600;
            color: #2E7D32;
            margin-bottom: 0.5rem;
        }
        .result-box {
            background-color: #F1F8E9;
            border-radius: 10px;
            padding: 15px;
            border-left: 5px solid #2E7D32;
        }
        .stAlert {
            background-color: #FFF3E0;
            border-radius: 10px;
            padding: 15px;
            border-left: 5px solid #FF9800;
        }
        [data-testid="stHeader"] {
            background: rgba(255,255,255,0.95);
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .block-container {
            padding-top: 1rem;
        }
        .gesture-item {
            margin-bottom: 10px;
            padding: 5px;
        }
        .stSidebar {
            background-color: #F5F5F5;
        }
        </style>
        """, unsafe_allow_html=True)

        # App title
        st.markdown('<h1 class="main-header">AI-Powered Virtual Calculator</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Solve mathematical equations with hand gestures</p>', unsafe_allow_html=True)

        # Sidebar with instructions
        with st.sidebar:
            st.markdown('<h2 style="text-align: center;">📝 Instructions</h2>', unsafe_allow_html=True)
        
            # Instructions with icons
            instructions = [
                {"gesture": "Thumb + Index☝️", "action": "Draw mode"},
                {"gesture": "Thumb + Index☝️ + Middle🖕", "action": "Stop drawing"},
                {"gesture": "Thumb + Middle Finger🖕", "action": "Erase"},
                {"gesture": "Thumb + Pinky Finger 🤙", "action": "Clear canvas"},
                {"gesture": "Two Fingers Up ✌️", "action": "Calculate"}
            ]
            
            for instruction in instructions:
                st.markdown(
                    f'<div class="gesture-item"><b>{instruction["gesture"]}</b>: {instruction["action"]}</div>',
                    unsafe_allow_html=True
                )
            
            st.image("calculator1.jpg", caption="Calculator Interface", use_column_width=True)
            
            # Add system status indicators
            st.markdown("### System Status")
            self.status_placeholder = st.empty()
            self.fps_placeholder = st.empty()
            
            # Add About section
            with st.expander("About this app"):
                st.write("""
                This AI-powered virtual calculator allows you to write mathematical equations using hand gestures
                and solves them in real-time. The application uses computer vision to track your hand movements
                and Google's Generative AI to interpret and solve the equations.
                
                Created with Streamlit, MediaPipe, and Google Gemini.
                """)

    def __init__(self):
        """Initialize the Calculator application"""
        # Load environment variables
        load_dotenv()
        
        # Initialize status variables
        self.is_webcam_active = False
        self.current_fps = 0
        self.last_calculation_time = 0
        self.cooldown_period = 3  # seconds
        
        # Initialize webcam
        self.cap = None
        self.setup_webcam()
        
        # Initialize Canvas Image
        self.imgCanvas = np.zeros((550, 950, 3), np.uint8)
        
        # Initialize Mediapipe hands
        self.mphands = hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )
        
        # Set Drawing Origin to Zero
        self.p1, self.p2 = 0, 0
        
        # Set Previous Time to Zero for FPS
        self.p_time = 0
        
        # Create Fingers Open/Close Position List
        self.fingers = []
        
        # Load CNN model
        self.cnn_model = self.build_cnn_model()
        
        # Initialize Gemini AI
        try:
            genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
            self.genai_model = genai.GenerativeModel(model_name='gemini-1.5-flash')
            self.genai_available = True
        except Exception as e:
            st.error(f"Error initializing Gemini AI: {e}")
            self.genai_available = False

    def setup_webcam(self):
        """Initialize the webcam with proper settings"""
        try:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 950)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 550)
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 130)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Check if webcam is opened successfully
            if self.cap.isOpened():
                self.is_webcam_active = True
                return True
            else:
                return False
        except Exception as e:
            st.error(f"Error setting up webcam: {e}")
            return False

    def build_cnn_model(self):
        """Build and compile the CNN model for feature extraction"""
        try:
            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(550, 950, 3)),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(10, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            return model
        except Exception as e:
            st.error(f"Error building CNN model: {e}")
            return None

    def preprocess_canvas(self, imgCanvas):
        """
        Preprocess the canvas image for better recognition
        
        Args:
            imgCanvas: The canvas image to preprocess
            
        Returns:
            The preprocessed canvas image
        """
        try:
            # Convert to grayscale for noise reduction
            gray_canvas = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur for noise reduction
            blurred_canvas = cv2.GaussianBlur(gray_canvas, (5, 5), 0)
            
            # Apply thresholding to enhance contrast
            _, thresholded_canvas = cv2.threshold(blurred_canvas, 50, 255, cv2.THRESH_BINARY)
            
            # Normalize pixel values to range [0, 1]
            normalized_canvas = thresholded_canvas / 255.0
            
            # Convert to 3-channel format for CNN compatibility
            processed_canvas = cv2.merge([normalized_canvas] * 3)
            
            # Resize to match CNN input shape
            resized_canvas = cv2.resize(processed_canvas, (950, 550))
            
            return resized_canvas
        except Exception as e:
            st.error(f"Error preprocessing canvas: {e}")
            return imgCanvas

    def extract_features_with_cnn(self, imgCanvas):
        """
        Extract features from the canvas using the CNN model
        
        Args:
            imgCanvas: The canvas image to extract features from
            
        Returns:
            The extracted features
        """
        try:
            # Preprocess the canvas
            preprocessed_canvas = self.preprocess_canvas(imgCanvas)
            
            # Convert to batch format for CNN input
            canvas_batch = np.expand_dims(preprocessed_canvas, axis=0)
            
            # Predict/extract features using the CNN
            features = self.cnn_model.predict(canvas_batch, verbose=0)
            return features
        except Exception as e:
            st.error(f"Error extracting features: {e}")
            return None

    def process_frame(self):
        """
        Process a frame from the webcam
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Reading the Video Capture
            success, img = self.cap.read()
            
            # Check if the frame is valid
            if not success or img is None:
                self.is_webcam_active = False
                return False
            
            # Resize the Image
            img = cv2.resize(src=img, dsize=(950, 550))
            
            # Flip the Image Horizontally for Selfie View
            self.img = cv2.flip(src=img, flipCode=1)
            
            # BGR Image Convert to RGB Image
            self.imgRGB = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            
            # Calculate FPS
            c_time = time.time()
            if self.p_time > 0:
                self.current_fps = 1 / (c_time - self.p_time)
            self.p_time = c_time
            
            return True
        except Exception as e:
            st.error(f"Error processing frame: {e}")
            self.is_webcam_active = False
            return False

    def process_hands(self):
        """Process hand landmarks from the current frame"""
        try:
            # Process the RGB Image to get hand landmarks
            result = self.mphands.process(image=self.imgRGB)
            
            # Initialize landmark list
            self.landmark_list = []
            
            if result.multi_hand_landmarks:
                for hand_lms in result.multi_hand_landmarks:
                    # Draw landmarks on the image
                    drawing_utils.draw_landmarks(
                        image=self.img,
                        landmark_list=hand_lms,
                        connections=hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        connection_drawing_spec=drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                    
                    # Extract coordinates for each landmark
                    for id, lm in enumerate(hand_lms.landmark):
                        h, w, c = self.img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        self.landmark_list.append([id, cx, cy])
            
            return True
        except Exception as e:
            st.error(f"Error processing hands: {e}")
            return False

    def identify_fingers(self):
        """Identify which fingers are extended"""
        try:
            # Initialize fingers list
            self.fingers = []
            
            # Verify hand detection
            if self.landmark_list:
                for id in [4, 8, 12, 16, 20]:  # Thumb, Index, Middle, Ring, Pinky
                    # For fingers other than thumb
                    if id != 4:
                        if self.landmark_list[id][2] < self.landmark_list[id-2][2]:
                            self.fingers.append(1)  # Finger is up
                        else:
                            self.fingers.append(0)  # Finger is down
                    # For thumb
                    else:
                        if self.landmark_list[id][1] < self.landmark_list[id-2][1]:
                            self.fingers.append(1)  # Thumb is up
                        else:
                            self.fingers.append(0)  # Thumb is down
                
                # Highlight extended fingertips
                for i in range(5):
                    if len(self.fingers) > i and self.fingers[i] == 1:
                        finger_id = (i+1) * 4
                        if finger_id < len(self.landmark_list):
                            cx, cy = self.landmark_list[finger_id][1], self.landmark_list[finger_id][2]
                            cv2.circle(
                                img=self.img,
                                center=(cx, cy),
                                radius=8,
                                color=(255, 0, 255),
                                thickness=cv2.FILLED
                            )
            
            return True
        except Exception as e:
            st.error(f"Error identifying fingers: {e}")
            return False

    def handle_drawing_mode(self):

        # Both Thumb and Index Fingers Up in Drwaing Mode
        if sum(self.fingers) == 2 and self.fingers[0]==self.fingers[1]==1:
            cx, cy = self.landmark_list[8][1], self.landmark_list[8][2]
            
            if self.p1 == 0 and self.p2 == 0:
                self.p1, self.p2 = cx, cy

            cv2.line(img=self.imgCanvas, pt1=(self.p1,self.p2), pt2=(cx,cy), color=(255,0,255), thickness=5)

            self.p1,self.p2 = cx,cy
        

        # Thumb, Index & Middle Fingers UP ---> Disable the Points Connection
        elif sum(self.fingers) == 3 and self.fingers[0]==self.fingers[1]==self.fingers[2]==1:
            self.p1, self.p2 = 0, 0
        

        # Both Thumb and Middle Fingers Up ---> Erase the Drawing Lines
        elif sum(self.fingers) == 2 and self.fingers[0]==self.fingers[2]==1:
            cx, cy = self.landmark_list[12][1], self.landmark_list[12][2]
        
            if self.p1 == 0 and self.p2 == 0:
                self.p1, self.p2 = cx, cy

            cv2.line(img=self.imgCanvas, pt1=(self.p1,self.p2), pt2=(cx,cy), color=(0,0,0), thickness=15)

            self.p1,self.p2 = cx,cy
        

        # Both Thumb and Pinky Fingers Up ---> Erase the Whole Thing (Reset)
        elif sum(self.fingers) == 2 and self.fingers[0]==self.fingers[4]==1:
            self.imgCanvas = np.zeros(shape=(550,950,3), dtype=np.uint8)

    def blend_canvas_with_feed(self):
         # Blend the Live Camera Feed and Canvas Images ---> Canvas Image Top on it the Original Transparency Image
        img = cv2.addWeighted(src1=self.img, alpha=0.7, src2=self.imgCanvas, beta=1, gamma=0)

        # Canvas_BGR Image Convert to Gray Scale Image ---> Maintain Intensity of Color Image
        imgGray = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2GRAY)

        # Gray Image Convert to Binary_Inverse Image ---> Gray Shades into only Two Colors (Black/White) based Threshold
        _, imgInv = cv2.threshold(src=imgGray, thresh=50, maxval=255, type=cv2.THRESH_BINARY_INV)

        # Binary_Inverse Image Convert into BGR Image ---> Single Channel Value apply All 3 Channel [0,0,0] or [255,255,255]
        # Bleding need same Channel for Both Images 
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        

        # Blending both Images ---> Binary_Inverse Image Black/White Top on Original Image
        img = cv2.bitwise_and(src1=img, src2=imgInv)

        # Canvas Color added on the Top on Original Image
        self.img = cv2.bitwise_or(src1=img, src2=self.imgCanvas)


    def analyze_image_with_genai(self):
        # Canvas_BGR Image Convert to RGB Image 
        imgCanvas = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2RGB)

        # Numpy Array Convert to PIL Image
        imgCanvas = PIL.Image.fromarray(imgCanvas)

        # Configures the genai Library
        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

        # Initializes a Flash Generative Model
        model = genai.GenerativeModel(model_name = 'gemini-1.5-flash')

        # Input Prompt
        prompt = "Analyze the image and provide the following:\n" \
                 "* The mathematical equation represented in the image.\n" \
                 "* The solution to the equation.\n" \
                 "* A short and sweet explanation of the steps taken to arrive at the solution."
        
        # Sends Request to Model to Generate Content using a Text Prompt and Image
        response = model.generate_content([prompt, imgCanvas])

        # Extract the Text Content of the Model’s Response.
        return response.text

    def main(self):
        col1, _, col3 = st.columns([0.8, 0.02, 0.18])

        with col1:
            # Stream the webcam video
            stframe = st.empty()
        
        with col3:
            # Placeholder for result output
            st.markdown(f'<h5 style="text-position:center;color:green;">OUTPUT:</h5>', unsafe_allow_html=True)
            result_placeholder = st.empty()

        while True:

            if not self.cap.isOpened():
                add_vertical_space(5)
                st.markdown(body=f'<h4 style="text-position:center; color:orange;">Error: Could not open webcam. \
                                    Please ensure your webcam is connected and try again</h4>', 
                            unsafe_allow_html=True)
                break

            self.process_frame()

            self.process_hands()

            self.identify_fingers()

            self.handle_drawing_mode()

            self.blend_canvas_with_feed()
            
            # Display the Output Frame in the Streamlit App
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            stframe.image(self.img, channels="RGB")

            # After Done Process with AI
            if sum(self.fingers) == 2 and self.fingers[1]==self.fingers[2]==1:
                result = self.analyze_image_with_genai()
                result_placeholder.write(f"Result: {result}")
        
        # Release the camera and close windows
        self.cap.release()
        cv2.destroyAllWindows()

try:

    # Creating an instance of the class
    calc = Calculator()

    # Streamlit Configuration Setup
    calc.streamlit_config()

    # Calling the main method
    calc.main()
except Exception as e:

    add_vertical_space(5)

    # Displays the Error Message
    st.markdown(f'<h5 style="text-position:center;color:orange;">{e}</h5>', unsafe_allow_html=True)

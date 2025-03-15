# AI-Powered Virtual Calculator

## Overview
The **AI-Powered Virtual Calculator** is an innovative tool that allows users to solve mathematical equations using hand gestures. This project integrates **MediaPipe** for hand tracking, **CNN-based feature extraction**, and **Google's Generative AI (Gemini)** for equation solving. The interface is built using **Streamlit**, making it interactive and user-friendly.

## Features
- **Hand Gesture Recognition:** Uses MediaPipe to detect hand movements and extract numerical input.
- **CNN-Based Feature Extraction:** Recognizes hand-drawn numbers and operators.
- **AI-Powered Equation Solver:** Uses Google's Gemini AI to process and solve mathematical expressions.
- **Interactive UI:** Built with Streamlit for real-time input and results display.
- **Seamless Integration:** Combines multiple AI techniques for a smooth user experience.

## Technologies Used
- **Python** (Core programming language)
- **MediaPipe** (Hand gesture tracking)
- **OpenCV** (Image processing)
- **TensorFlow/Keras** (CNN model for digit recognition)
- **Google Gemini AI** (Equation solving)
- **Streamlit** (Web interface)

## Installation
### Prerequisites
Make sure you have the following installed:
- Python 3.x
- pip (Python package manager)

### Steps
1. Clone the repository:
   ```sh
   git clone https://github.com/Kalpanil/Ai_Virtual_Calculator
   cd ai-virtual-calculator
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the application:
   ```sh
   streamlit run app.py
   ```

## Usage
1. Launch the application using Streamlit.
2. Show hand gestures to input numbers and mathematical operators.
3. The application processes the gestures and extracts the mathematical expression.
4. The Gemini AI model solves the equation and displays the result.

## Screenshot
![image](https://github.com/user-attachments/assets/d19944f3-308a-42db-bb10-aa436f703047)

## Future Improvements
- Enhance gesture recognition accuracy.
- Support additional mathematical operations and functions.
- Improve real-time processing speed.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

## License
This project is licensed under the MIT License.

## Contact
For any questions or suggestions, feel free to reach out at kalpanil22kanbarkar@gmail.com.


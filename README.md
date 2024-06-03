# Speaker Recognizer

This project is a simple speaker recognition application that uses machine learning to identify speakers from audio recordings. The application is built with Python and utilizes various libraries for audio processing and visualization.

## Features

- Record audio and identify the speaker.
- Import WAV files and identify the speaker.
- Visualize audio spectrum during recording.


## Requirements

- Python 3.7 or higher.
- Required libraries: `numpy`, `matplotlib`, `librosa`, `scikit-learn`, `sounddevice`, `pyaudio`, `tkinter`, `scipy`

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/speaker-recognizer.git
    cd speaker-recognizer
    ```

2. Install the required Python libraries:

    ```sh
    pip install numpy matplotlib librosa scikit-learn sounddevice pyaudio scipy
    ```

3. Ensure you have the following files in the project directory:

    - `microphone.ico`
    - `background.png`
    - `background2.png`
    - `img0.png`
    - `img1.png`
    - `img2.png`

4. Create a directory named `Voices` in the project directory. This folder should contain WAV files for training the model. The filenames should start with the speaker's name (e.g., `alice_01.wav`, `bob_02.wav`).

## Usage

1. **Run the Application:**

    ```sh
    python main.py
    ```

2. **Main Window:**
    - Click on the microphone icon to start recording.
    - Click on the upload icon to import a WAV file.

3. **Recording Interface:**
    - Recording starts automatically after a 1-second delay.
    - The predicted speaker is displayed after recording.
    - The audio spectrum is visualized in real-time.

## Code Overview

- `main.py`: Main application script.
- `Voices/`: Directory containing training audio files.
- `extract_features()`: Function to extract features from audio files.
- `import_wav()`: Function to import a WAV file and identify the speaker.
- `change_colors()`: Function to switch to the recording interface and start recording.

### Training the Model

The model is trained using the K-Nearest Neighbors (KNN) algorithm. Training data is loaded from the `Voices` directory. MFCC features are extracted from the audio files for training.

### Example Training Data Structure

```
Voices/
├── alice_01.wav
├── alice_02.wav
├── bob_01.wav
└── bob_02.wav
```

## Acknowledgements

This project uses the following open-source libraries:

- [librosa](https://librosa.org/)
- [scikit-learn](https://scikit-learn.org/)
- [sounddevice](https://python-sounddevice.readthedocs.io/)
- [pyaudio](http://people.csail.mit.edu/hubert/pyaudio/)
- [tkinter](https://docs.python.org/3/library/tkinter.html)
- [matplotlib](https://matplotlib.org/)

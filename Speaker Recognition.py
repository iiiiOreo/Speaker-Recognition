from tkinter import *
from tkinter import filedialog
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import style
import os
import librosa
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import sounddevice as sd
import scipy.io.wavfile as wav

style.use('dark_background')
speaker = "(Wait till record)"
Recording = "recording.wav"

def change_colors(event):
    global speaker
    window.destroy()
    window2 = Tk()
    window2.geometry("750x450")
    window2.configure(bg="#393e46")
    window2.iconbitmap("microphone.ico")
    window2.title("Speaker Recognizer")

    canvas = Canvas(window2, bg="#393e46", height=450, width=750, bd=0, highlightthickness=0, relief="ridge")
    canvas.place(x=0, y=0)

    background_img = PhotoImage(file="background2.png")
    background = canvas.create_image(375.0, 226.0, image=background_img)

    img1 = PhotoImage(file="img1.png")
    circle_button = canvas.create_image(376, 389, image=img1)

    fig, ax = plt.subplots()
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 4000)
    bar_container = ax.bar(np.arange(100), np.zeros(100), color='white', width=1.0)
    canvas_plot = FigureCanvasTkAgg(fig, master=window2)
    canvas_plot.get_tk_widget().place(x=70, y=115, width=610, height=180)
    canvas_plot.get_tk_widget().config(bg='#080616', highlightbackground='#080616')

    def start_recording():
        global speaker
        DURATION = 10
        recording = sd.rec(int(DURATION * 44100), samplerate=44100, channels=2)
        sd.wait()
        wav.write(Recording, 44100, recording)
        features = extract_features(Recording)
        features = np.array([features])
        predicted_speaker_probabilities = model.predict_proba(features)[0]
        max_probability = np.max(predicted_speaker_probabilities)
        if max_probability < 0.7:
            speaker = "Not Recognized"
        else:
            predicted_speaker = unique_labels[np.argmax(predicted_speaker_probabilities)]
            speaker = predicted_speaker
        label.config(text=speaker)
        start_stream()  # Start the stream after recording is done

    p = pyaudio.PyAudio()

    def update_plot():
        data = stream.read(CHUNK, exception_on_overflow=False)
        data_np = np.frombuffer(data, dtype=np.int16)
        spectrum = np.abs(np.fft.fft(data_np))[:CHUNK // 2]

        for rect, h in zip(bar_container.patches, spectrum[:100]):
            rect.set_height(h)

        canvas_plot.draw()
        window2.after(10, update_plot)

    def start_stream():
        global stream
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        update_plot()

    label = Label(window2, text=speaker, bd=0, anchor=CENTER, bg="#070513", font=("cambria", 22, "bold"), fg="#FF0133", padx=15, pady=15, justify=CENTER, relief=RAISED)
    label.place(x=410, y=44)

    window2.after(1000, start_recording)  # Start recording after 1 second delay
    window2.resizable(False, False)
    window2.mainloop()

def import_wav(event):
    global Recording, speaker
    Recording = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if Recording:
        features = extract_features(Recording)
        features = np.array([features])
        predicted_speaker_probabilities = model.predict_proba(features)[0]
        max_probability = np.max(predicted_speaker_probabilities)
        if max_probability < 0.7:  # You can adjust this threshold as needed
            speaker = "Not Recognized"
        else:
            predicted_speaker = unique_labels[np.argmax(predicted_speaker_probabilities)]
            speaker = predicted_speaker
        result_label.config(text=f"Speaker: {speaker}")

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except Exception as e:
        print("Error encountered while parsing file:", file_path)
        return None

def get_labels_and_features(audio_dir):
    labels = []
    features = []

    for file_name in os.listdir(audio_dir):
        try:
            label = file_name.split('_')[0]
            labels.append(label)

            file_path = os.path.join(audio_dir, file_name)
            mfccs = extract_features(file_path)
            if mfccs is not None:
                features.append(mfccs)
        except Exception as e:
            print("Error encountered while parsing file:", file_name)
            continue
    return np.array(labels), np.array(features)

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 256

AUDIO_DIR = 'Voices'
labels, features = get_labels_and_features(AUDIO_DIR)
unique_labels, label_counts = np.unique(labels, return_counts=True)
print("Training data distribution:")
for label, count in zip(unique_labels, label_counts):
    print(f"{label}: {count} samples")

print("Unique labels:", np.unique(labels))
print("Number of training features:", len(features))
labels_train, labels_test, features_train, features_test = train_test_split(labels, features, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(features_train, labels_train)
labels_pred = model.predict(features_test)

accuracy = accuracy_score(labels_test, labels_pred)
print('Accuracy: ', accuracy)
train_accuracy = model.score(features_train, labels_train)
print('Training Accuracy: ', train_accuracy)
test_accuracy = model.score(features_test, labels_test)
print('Test Accuracy: ', test_accuracy)

window = Tk()
window.geometry("750x450")
window.configure(bg="#393e46")
window.iconbitmap("microphone.ico")
window.title("Speaker Recognizer")

canvas = Canvas(window, bg="#393e46", height=450, width=750, bd=0, highlightthickness=0, relief="ridge")
canvas.place(x=0, y=0)

background_img = PhotoImage(file="background.png")
background = canvas.create_image(375.0, 225.0, image=background_img)

img0 = PhotoImage(file="img0.png")
circle_button = canvas.create_image(376, 392, image=img0)
canvas.tag_bind(circle_button, '<ButtonPress-1>', change_colors)

img2 = PhotoImage(file="img2.png")
circle_button2 = canvas.create_image(375, 230, image=img2)
canvas.tag_bind(circle_button2, '<ButtonPress-1>', import_wav)

result_label = Label(window, text="", bg="#0C0A24", font=("cambria", 18, "bold"), fg="#D9D9D9")
result_label.place(x=283, y=270)

window.resizable(False, False)
window.mainloop()
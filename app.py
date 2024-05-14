import os
import json
import numpy as np
import librosa
import crepe
import moviepy.editor as mpy
from scipy.ndimage import median_filter
from moviepy.video.VideoClip import TextClip

class MelodyExtractor:
    def __init__(self, song_path, output_path_json, output_path_video):
        self.song_path = song_path
        self.output_path_json = output_path_json
        self.output_path_video = output_path_video
        self.notes = []
        self.timings = []

    def note_to_midi(self, note_name):
        return librosa.note_to_midi(note_name) if note_name is not None else None

    def midi_to_note(self, midi_number):
        return librosa.midi_to_note(midi_number) if midi_number is not None else None

    def analyze_audio(self):
        # Load the audio file
        y, sr = librosa.load(self.song_path, sr=None)

        # Harmonic-percussive source separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        # Use CREPE to analyze the harmonic component and extract pitch
        time, frequency, confidence, _ = crepe.predict(y_harmonic, sr, viterbi=True)

        # Apply a high confidence threshold to filter out low-confidence pitches
        confidence_threshold = 0.8
        confident_indices = confidence > confidence_threshold
        frequencies = frequency[confident_indices]
        times = time[confident_indices]

        # Convert frequencies to MIDI notes
        midi_notes = librosa.hz_to_midi(frequencies)

        # Smooth the midi notes using a median filter
        midi_notes = median_filter(midi_notes, size=5)

        # Group notes with a larger time threshold to avoid frequent notes
        grouped_notes = []
        current_group = [midi_notes[0]]
        current_time = times[0]
        time_threshold = 0.5  # Increased threshold to reduce note density

        for midi_note, t in zip(midi_notes[1:], times[1:]):
            if t - current_time < time_threshold:
                current_group.append(midi_note)
            else:
                grouped_notes.append((current_group, current_time))
                current_group = [midi_note]
                current_time = t

        if current_group:
            grouped_notes.append((current_group, current_time))

        self.notes = []
        self.timings = []

        for group, t in grouped_notes:
            # Take the most frequent note in the group to represent the melody
            most_frequent_note = max(set(group), key=group.count)
            self.notes.append([self.midi_to_note(most_frequent_note)])
            self.timings.append(t)

    def generate_json(self):
        notes_data = []
        for i, note_group in enumerate(self.notes[:-1]):
            note_info = {
                "notes": note_group,
                "start_time": self.timings[i],
                "duration": self.timings[i+1] - self.timings[i]
            }
            notes_data.append(note_info)

        # Handle the last note group separately
        last_note_info = {
            "notes": self.notes[-1],
            "start_time": self.timings[-1],
            "duration": 0.5  # Default duration for the last note
        }
        notes_data.append(last_note_info)

        with open(self.output_path_json, 'w') as outfile:
            json.dump(notes_data, outfile, indent=4)

    def generate_video(self, use_original_audio=True):
        # Downsample the number of frames to reduce video length
        frame_interval = max(1, len(self.timings) // 300)

        # Function to create a text clip for the notes
        def make_frame(t):
            frame_idx = min(int(t * len(self.notes) / self.timings[-1]), len(self.notes) - 1)
            note_text = " + ".join(self.notes[frame_idx]) if self.notes[frame_idx] else ""
            text_clip = TextClip(note_text, fontsize=70, color='white', bg_color='black', size=(1920, 1080))
            return text_clip.get_frame(t)

        # Create the video clip using the make_frame function
        duration = self.timings[-1] if self.timings else 0  # Handle empty timings case
        animation = mpy.VideoClip(make_frame, duration=duration)

        if use_original_audio and duration > 0:
            # Use the original audio
            audio = mpy.AudioFileClip(self.song_path)
            duration = min(duration, audio.duration)
            audio = audio.subclip(0, duration)
        else:
            # Generate silence (dummy audio clip if needed)
            audio = mpy.AudioClip(lambda t: np.sin(440 * 2 * np.pi * t), duration=duration)

        # Combine the video and audio
        final_clip = animation.set_audio(audio)

        # Write the final video
        final_clip.write_videofile(self.output_path_video, fps=24)

    def generate_tutorial(self):
        self.analyze_audio()
        self.generate_json()
        self.generate_video(use_original_audio=True)

# Example usage
song_path = '24.mp3'
output_path_json = 'notes.json'
output_path_video = 'notes_video.mp4'

extractor = MelodyExtractor(song_path, output_path_json, output_path_video)
extractor.generate_tutorial()

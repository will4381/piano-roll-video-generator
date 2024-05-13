import os
import librosa
import numpy as np
import moviepy.editor as mpy
from music21 import note, stream, duration
from moviepy.video.io.bindings import mplfig_to_npimage
import matplotlib.pyplot as plt
from mido import Message, MidiFile, MidiTrack

class PianoRollGenerator:
    def __init__(self, song_path, output_path):
        self.song_path = song_path
        self.output_path = output_path
        self.notes = None
        self.timings = None
        self.stream = None
        self.piano_roll = None
        self.piano_keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    def analyze_audio(self):
        # Load the audio file
        y, sr = librosa.load(self.song_path)

        # Perform pitch detection
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

        # Define a threshold for the minimum magnitude
        threshold = np.median(magnitudes)  # Adjust this value based on your audio

        # Filter out the positions where the magnitude is too low
        max_magnitude_indices = magnitudes.argmax(axis=0)
        confident_indices = (magnitudes.max(axis=0) > threshold)

        # Initialize notes array
        self.notes = []

        # Iterate over confident indices to get the pitch values
        for j, (mag, idx) in enumerate(zip(magnitudes.max(axis=0), max_magnitude_indices)):
            if confident_indices[j]:  # Only consider frames with magnitudes above the threshold
                pitch = pitches[idx, j]
                if np.isfinite(pitch) and pitch > 0:
                    note_name = librosa.hz_to_note(pitch)
                    self.notes.append(note_name)
                else:
                    self.notes.append(None)  # Placeholder for non-notes or rest
            else:
                self.notes.append(None)  # Placeholder for non-notes or rest

        # Get the timing of each note
        self.timings = librosa.frames_to_time(np.arange(len(self.notes)), sr=sr)

    def transcribe_notes(self):
        # Create a music21 stream
        self.stream = stream.Stream()
        self.stream.quarterLength = 0.25  # Ensure it's never zero, adjust as needed

        # Iterate over the notes and timings
        for note_name, start_time, end_time in zip(self.notes[:-1], self.timings[:-1], self.timings[1:]):
            if note_name:  # This is a note
                # Correct accidental symbols for compatibility with music21
                note_name = note_name.replace('♯', '#').replace('♭', '-')

                # Parse the note name into a music21 Note
                current_note = note.Note(note_name)
            else:  # This is a rest
                current_note = note.Rest()

            # Calculate the note duration based on the timings
            duration_quarter_length = (end_time - start_time) / self.stream.quarterLength
            current_note.duration = duration.Duration(duration_quarter_length)

            # Add the note or rest to the stream
            self.stream.append(current_note)

    def generate_piano_roll(self):
        # Create a piano roll matrix
        piano_roll = np.zeros((len(self.piano_keys), len(self.notes)))

        # Fill the piano roll matrix with note velocities
        for i, note_name in enumerate(self.notes):
            if note_name:
                # Get the index of the piano key for the current note
                key_index = self.piano_keys.index(note_name.split('-')[0].split('/')[0])
                piano_roll[key_index, i] = 1

        self.piano_roll = piano_roll

    def generate_video(self, use_original_audio=True):
        # Create a figure and axis for the piano roll plot
        fig, ax = plt.subplots(figsize=(8, 4))

        # Plot the piano roll
        piano_roll_plot = ax.imshow(self.piano_roll, cmap='binary', aspect='auto', origin='lower')

        # Set the y-ticks to the piano keys
        ax.set_yticks(range(len(self.piano_keys)))
        ax.set_yticklabels(self.piano_keys)

        # Hide the x-axis
        ax.get_xaxis().set_visible(False)

        # Function to update the plot for each frame
        def make_frame(t):
            # Calculate the frame index based on time
            frame_idx = int(t * len(self.notes) / self.timings[-1])

            # Update the piano roll plot
            piano_roll_plot.set_data(self.piano_roll[:, :frame_idx])

            # Convert the plot to a numpy array
            return mplfig_to_npimage(fig)

        # Create the video clip using the make_frame function
        animation = mpy.VideoClip(make_frame, duration=self.timings[-1])

        if use_original_audio:
            # Use the original audio
            audio = mpy.AudioFileClip(self.song_path)
        else:
            # Generate MIDI audio
            midi_file = MidiFile()
            track = MidiTrack()
            midi_file.tracks.append(track)
            
            elapsed_time = 0
            for note_name, start_time, end_time in zip(self.notes[:-1], self.timings[:-1], self.timings[1:]):
                if note_name:
                    # Get the MIDI note number
                    note_number = librosa.note_to_midi(note_name)

                    # Calculate delta times in ticks
                    delta_start = mido.second2tick(start_time - elapsed_time, midi_file.ticks_per_beat, tempo)
                    delta_end = mido.second2tick(end_time - start_time, midi_file.ticks_per_beat, tempo)

                    # Create MIDI messages with delta times
                    track.append(Message('note_on', note=note_number, velocity=100, time=int(delta_start)))
                    track.append(Message('note_off', note=note_number, velocity=0, time=int(delta_end)))

                    elapsed_time = end_time

            # Save the MIDI file
            midi_file.save('temp_midi.mid')

            # Generate audio from the MIDI file
            audio = mpy.AudioFileClip('temp_midi.mid')

        # Combine the video and audio
        final_clip = animation.set_audio(audio)

        # Write the final video
        final_clip.write_videofile(self.output_path, fps=24)

        # Clean up temporary files
        os.remove('temp_midi.mid')

    def generate_tutorial(self, use_original_audio=True):
        self.analyze_audio()
        self.transcribe_notes()
        self.generate_piano_roll()
        self.generate_video(use_original_audio)

# Example usage
song_path = '24.mp3'
output_path = 'video.mp4'

generator = PianoRollGenerator(song_path, output_path)
generator.generate_tutorial(use_original_audio=True)
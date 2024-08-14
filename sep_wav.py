import os
from typing import List
from tqdm import tqdm
from glob import glob
import subprocess
import numpy as np
import librosa
import soundfile
from pydub import AudioSegment, effects
import torch
import torchaudio
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.transforms import Fade

temp_log_path = "temp_ffmpeg_log.txt"  # Temporary path for silence detection logs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def extract_voice(
        model,
        mix,
        segment=16.,
        overlap=0.1,
        device=None,
        sample_rate=None
):
    """ Apply model to a given mixture.
    Args:
        segment (int): Segment length in seconds
        device (torch.device, str, or None): If provided, device on which to execute the computation, otherwise `mix.device` is assumed.
            When `device` is different from `mix.device`, only local computations will be on `device`, while the entire tracks will be stored on `mix.device`.
    """

    if device is None:
        device = mix.device
    else:
        device = torch.device(device)

    if sample_rate is None:
        raise "Demucs model loading error"
    mix = mix.to(device)
    batch, channels, length = mix.shape

    chunk_len = int(sample_rate * segment * (1 + overlap))
    start = 0
    end = chunk_len
    overlap_frames = overlap * sample_rate
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape='linear')

    final = torch.zeros(batch, len(model.sources), channels, length, device=device)

    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        with torch.no_grad():
            out = model.forward(chunk)
        out = fade(out)
        final[:, :, :, start:end] += out
        if start == 0:
            fade.fade_in_len = int(overlap_frames)
            start += int(chunk_len - overlap_frames)
        else:
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0
    return final


def mp4_to_wav(input_dir: str, input_file: str):
    # Convert mp4 files to wav format

    ext = os.path.splitext(input_file)[1][1:]

    if ext != "mp4":
        return 
    else:
        track = AudioSegment.from_file(os.path.join(input_dir, input_file),  format='mp4')
        track.export(os.path.join(input_dir, os.path.splitext(input_file)[0]+".wav"), format='wav')


def audio_norm(input_filepath: str, output_filepath: str, sample_rate = 44100, use_preprocessing = True):
    # Apply normalization to audio files

    ext = os.path.splitext(input_filepath)[1][1:]

    assert ext in ["wav", "flac"], "This format is not supported."

    rawsound = AudioSegment.from_file(input_filepath, format=ext)

    # Change sample rate
    rawsound = rawsound.set_frame_rate(sample_rate)

    # Change channels
    if rawsound.channels != 1 :
        rawsound = rawsound.set_channels(1)

    normalizedsound = effects.normalize(rawsound)
    normalizedsound.export(output_filepath, format="flac")


def get_ffmpeg_args(filepath: str) -> str:
    """ Generate a command line for ffmpeg.
    Args:
        filepath (str)
    Returns:
        str: Command line with ffmpeg argument values
    """

    global temp_log_path

    return f'ffmpeg -i "{filepath}" -af "silencedetect=n=-50dB:d=1.5,ametadata=print:file={temp_log_path}" -f null -'


def get_audiofiles(path: str) -> List[str]:
    """ All audio files inside that folder will be imported. (Only wav and flac formats are supported)
    Args:
        path (str): Path to folder
    Returns:
        List[str]: Path to audio file
    """

    filepaths = glob(os.path.join(path, "**", "*.flac"), recursive=True)
    filepaths += glob(os.path.join(path, "*.flac"), recursive=True)
    filepaths += glob(os.path.join(path, "**", "*.wav"), recursive=True)
    filepaths += glob(os.path.join(path, "*.wav"), recursive=True)
    filepaths = list(set(filepaths))
    filepaths.sort()

    return filepaths


def main(input_dir: str, output_dir: str, split_sil: bool=False, use_preprocessing: bool=True, use_norm: bool=True, use_extract: bool=True) -> None:
    """ Main
    Args:
        input_dir (str): Source path for audio files
        output_dir (str): Output path for audio files that have completed processing (The final version is saved in the "final" folder.)
        split_sil (bool, optional): Cut out partial silence from audio files. Default is False
        use_norm (bool, optional): Apply audio normalization. Default is True
        use_extract (bool, optional): Extract only voices from audio mixed with singing. Default is True
    """

    for filename in tqdm(os.listdir(input_dir), desc="Converting mp4 to wav... "):
        mp4_to_wav(input_dir, filename)

    filepaths = get_audiofiles(input_dir)

    output_final_dir = os.path.join(output_dir, "final")
    os.makedirs(output_final_dir, exist_ok=True)

    if use_norm:
        output_norm_dir = os.path.join(output_dir, "norm")
        os.makedirs(output_norm_dir, exist_ok=True)

        for filepath in tqdm(filepaths, desc="Normalizing... "):
            filename = os.path.splitext(os.path.basename(filepath))[0]
            out_filepath = os.path.join(output_norm_dir, filename) + ".wav"
            audio_norm(filepath, out_filepath, use_preprocessing)

        filepaths = get_audiofiles(output_norm_dir)

    for filepath in tqdm(filepaths, desc="Cutting... "):
        duration = librosa.get_duration(filename=filepath)
        max_last_seg_duration = 0
        sep_duration_final = 15
        sep_duration = 15

        while sep_duration > 4:
            last_seg_duration = duration % sep_duration
            if max_last_seg_duration < last_seg_duration:
                max_last_seg_duration = last_seg_duration
                sep_duration_final = sep_duration
            sep_duration -= 1

        filename = os.path.splitext(os.path.basename(filepath))[0]
        out_filepath = os.path.join(output_final_dir, f"{filename}-%03d.wav")
        subprocess.run(f'ffmpeg -i "{filepath}" -f segment -segment_time {sep_duration_final} "{out_filepath}" -y', capture_output=True, shell=True)

    filepaths = get_audiofiles(output_final_dir)

    if use_extract:
        output_voice_dir = os.path.join(output_dir, "voice")
        os.makedirs(output_voice_dir, exist_ok=True)
        
        bundle = HDEMUCS_HIGH_MUSDB_PLUS
        model = bundle.get_model()
        model.to(device)
        sample_rate = bundle.sample_rate
        print(f"Sample rate: {sample_rate}")

        for filepath in tqdm(filepaths, desc="Extracting vocals... "):
            if os.path.exists(temp_log_path):
                os.remove(temp_log_path)

            waveform, sample_rate = torchaudio.load(filepath)  # Replace SAMPLE_SONG with desired path for different song
            waveform.to(device)

            # parameters
            segment: int = 15
            overlap = 0.1

            sources = extract_voice(
                model,
                waveform[None],
                device=device,
                segment=segment,
                overlap=overlap,
                sample_rate=sample_rate
            )[0]

            sources_list = model.sources
            sources = list(sources)

            audios = dict(zip(sources_list, sources))

            filename = os.path.splitext(os.path.basename(filepath))[0]
            out_filepath = os.path.join(output_voice_dir, f"{filename}.wav")

            torchaudio.save(out_filepath, audios["vocals"].cpu(), sample_rate)  # The audio has drums, bass, vocals, etc., but we only need the vocals.

            if use_preprocessing:
                rawsound = AudioSegment.from_file(out_filepath, format='wav')
                rawsound = rawsound.set_channels(1)
                rawsound.export(out_filepath, format="wav")

        filepaths = get_audiofiles(output_voice_dir)

    for filepath in tqdm(filepaths, desc="Removing silence... "):
        if os.path.exists(temp_log_path):
            os.remove(temp_log_path)

        ffmpeg_arg = get_ffmpeg_args(filepath)
        subprocess.run(ffmpeg_arg, capture_output=True, shell=True)

        start = None
        end = None

        with open(temp_log_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if "lavfi.silence_start" in line:
                    start = float(line.split("=")[1])
                if "lavfi.silence_end" in line:
                    end = float(line.split("=")[1])

        if start != None:
            if start == 0 and end == None:
                os.remove(filepath)
            elif split_sil:
                if end == None:
                    end = len(y)
                else:
                    end = int(end)

                y, sr = librosa.load(filepath, sr=None)
                y = np.concatenate((y[: round(sr * start)], y[round(sr * end) :]), axis=None)
                soundfile.write(filepath, y, samplerate=sr)

    if os.path.exists(temp_log_path):
        os.remove(temp_log_path)


def demucs(input_path, output_path):
    bundle = HDEMUCS_HIGH_MUSDB_PLUS
    model = bundle.get_model()
    model.to(device)
    sample_rate = bundle.sample_rate
    print(f"Sample rate: {sample_rate}")

    filepaths = glob(input_path+"*.wav")

    for filepath in tqdm(filepaths, desc="Extracting vocals... "):
        if os.path.exists(temp_log_path):
            os.remove(temp_log_path)

        waveform, sample_rate = torchaudio.load(filepath)  # Replace SAMPLE_SONG with desired path for different song

        # Check number of channels
        num_channels = waveform.shape[0]

        # If audio is stereo, convert to mono
        if num_channels == 1:
            waveform = torchaudio.functional.remix_channels(waveform, [0, 0])

        waveform.to(device)

        # parameters
        segment: int = 15
        overlap = 0.1

        sources = extract_voice(
            model,
            waveform[None],
            device=device,
            segment=segment,
            overlap=overlap,
            sample_rate=sample_rate
        )[0]

        sources_list = model.sources
        sources = list(sources)

        audios = dict(zip(sources_list, sources))

        filename = os.path.splitext(os.path.basename(filepath))[0]
        out_filepath = os.path.join(output_path, f"{filename}.wav")

        torchaudio.save(out_filepath, audios["vocals"].cpu(), sample_rate)  # The audio has drums, bass, vocals, etc., but we only need the vocals.

        '''
        if use_preprocessing:
            rawsound = AudioSegment.from_file(out_filepath, format='wav')
            rawsound = rawsound.set_channels(1)
            rawsound.export(out_filepath, format="wav")
        '''


if __name__ == "__main__":
    input_dir = "preprocess"
    output_dir = "preprocess_out"
    split_sil = False
    use_preprocessing = True    # for set samplerate to 44100, channel to mono
    use_norm = True
    use_extract = True

    main(
        input_dir=input_dir,
        output_dir=output_dir,
        split_sil=split_sil,
        use_preprocessing=use_preprocessing,
        use_norm=use_norm,
        use_extract=use_extract
    )
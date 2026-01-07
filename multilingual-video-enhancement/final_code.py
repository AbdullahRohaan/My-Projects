import requests
import time
from datetime import timedelta
import moviepy.editor as mp
from deep_translator import GoogleTranslator
from gtts import gTTS
import subprocess
import chardet

# Set your AssemblyAI API key
api_key = 'f7a456901a6b4d35af26682864e6033c'

# Step 1: Upload the audio file to AssemblyAI for transcription


def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']


def upload_audio(file_path):
    headers = {'authorization': api_key}
    with open(file_path, 'rb') as f:
        response = requests.post(
            'https://api.assemblyai.com/v2/upload', headers=headers, files={'file': f})
    return response.json()['upload_url']

# Step 2: Submit transcription request with Spanish language


def submit_transcription(audio_url, language='es'):
    headers = {
        'authorization': api_key,
        'content-type': 'application/json'
    }
    data = {
        'audio_url': audio_url,
        'language_code': language  # Spanish language code
    }
    response = requests.post(
        'https://api.assemblyai.com/v2/transcript', headers=headers, json=data)
    return response.json()['id']

# Step 3: Poll for the transcription results


def get_transcription_result(transcript_id):
    headers = {'authorization': api_key}
    while True:
        response = requests.get(
            f'https://api.assemblyai.com/v2/transcript/{transcript_id}', headers=headers)
        result = response.json()
        if result['status'] == 'completed':
            return result
        elif result['status'] == 'failed':
            raise Exception('Transcription failed')
        time.sleep(5)  # Wait for 5 seconds before checking again

# Step 4: Create an .srt subtitle file


def create_srt(transcript, output_file='subtitles.srt'):
    words = transcript['words']
    with open(output_file, 'w') as f:
        for idx, word_info in enumerate(words):
            start_time = str(timedelta(seconds=word_info['start'] / 1000))
            end_time = str(timedelta(seconds=word_info['end'] / 1000))

            f.write(f"{idx + 1}\n")
            f.write(
                f"{start_time.replace('.', ',')} --> {end_time.replace('.', ',')}\n")
            f.write(f"{word_info['text']}\n\n")
    return output_file

# Step 5: Translate subtitles using deep_translator


def translate_subtitles(input_srt, output_srt,code, target_language='en'):
    translator = GoogleTranslator(source='auto', target=target_language)

    with open(input_srt, 'r', encoding=code) as f:
        lines = f.readlines()

    translated_lines = []
    for line in lines:
        if '-->' in line or line.strip().isdigit():
            translated_lines.append(line)
        else:
            translated_text = translator.translate(line.strip())
            translated_lines.append(translated_text + '\n')

    with open(output_srt, 'w', encoding='utf-8') as f:
        f.writelines(translated_lines)
    return output_srt

# Step 6: Convert translated subtitles into audio using gTTS


def subtitles_to_audio(subtitles_file, output_audio_file, language='en'):
    with open(subtitles_file, 'r', encoding='utf-8') as f:
        subtitles = f.read()

    tts = gTTS(text=subtitles, lang=language)
    tts.save(output_audio_file)
    return output_audio_file

# Step 7: Attach translated audio to the video


def attach_audio_to_video(video_file, audio_file, output_video_file):
    cmd = f'ffmpeg -i {video_file} -i {audio_file} -c:v copy -c:a aac {output_video_file}'
    subprocess.run(cmd, shell=True)
    return output_video_file

# Main function to run the entire process


def process_video(video_file, api_key, target_language='en'):
    audio_file = "extracted_audio.mp3"
    subtitles_file = "subtitles.srt"
    translated_subtitles_file = "translated_subtitles.srt"
    translated_audio_file = "translated_audio.mp3"
    output_video_file = "output_video.mp4"

    # Step 1: Extract audio from the video (using moviepy)
    video = mp.VideoFileClip(video_file)
    video.audio.write_audiofile(audio_file)

    # Step 2: Upload the audio and generate Spanish subtitles via AssemblyAI
    audio_url = upload_audio(audio_file)
    transcript_id = submit_transcription(audio_url, language='es')
    transcript = get_transcription_result(transcript_id)
    subtitles_file = create_srt(transcript)
    code=detect_encoding(subtitles_file)
    # Step 3: Translate the Spanish subtitles to the target language
    translate_subtitles(subtitles_file, translated_subtitles_file, code, target_language)

    # Step 4: Convert the translated subtitles into speech/audio
    subtitles_to_audio(translated_subtitles_file,
                       translated_audio_file, language=target_language)

    # Step 5: Attach the translated audio to the original video
    attach_audio_to_video(video_file, translated_audio_file, output_video_file)

    print(f"Process complete! The final video is saved as {output_video_file}")


# Example usage
if __name__ == "__main__":
    video_path = 'input.mp4'  # Path to your input video
    # Target language for translation (English in this case)
    target_language = 'en'

    process_video(video_path, api_key, target_language)


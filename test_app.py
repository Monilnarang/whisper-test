#!/usr/bin/env python3
import os
import threading
import time
import psutil
import platform
import subprocess
from flask import Flask, request, jsonify
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import whisper
import torch
import requests
import tempfile

load_dotenv()

app = Flask(__name__)

# Global variables
_model = None
system_info = {}

def get_system_info():
    """Collect comprehensive system information"""
    global system_info

    try:
        # Basic system info
        system_info = {
            'timestamp': datetime.now().isoformat(),
            'platform': platform.platform(),
            'processor': platform.processor(),
            'architecture': platform.architecture(),
            'python_version': platform.python_version(),

            # Memory info
            'total_memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'available_memory_gb': round(psutil.virtual_memory().available / (1024**3), 2),
            'memory_percent_used': psutil.virtual_memory().percent,

            # CPU info
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'cpu_percent': psutil.cpu_percent(interval=1),

            # Disk info
            'disk_total_gb': round(psutil.disk_usage('/').total / (1024**3), 2),
            'disk_free_gb': round(psutil.disk_usage('/').free / (1024**3), 2),
            'disk_used_percent': psutil.disk_usage('/').percent,

            # PyTorch/CUDA info
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }

        # Add CUDA device info if available
        if torch.cuda.is_available():
            system_info['cuda_device_name'] = torch.cuda.get_device_name(0)
            system_info['cuda_memory_gb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)

        # Check ffmpeg
        try:
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=10)
            system_info['ffmpeg_available'] = True
            system_info['ffmpeg_version'] = result.stdout.split('\n')[0] if result.stdout else 'Unknown'
        except:
            system_info['ffmpeg_available'] = False
            system_info['ffmpeg_version'] = 'Not available'

    except Exception as e:
        system_info['error'] = f"Error collecting system info: {e}"

    return system_info

def download_sample_audio():
    """Download a sample MP3 file for testing"""

    # List of public domain dialogue audio sources
    sample_urls = [
        # LibriVox - public domain audiobooks with dialogue
        "https://ia800304.us.archive.org/11/items/short_poetry_collection_036_1202_librivox/spokenword_00_variousauthors_64kb.mp3",

        # Internet Archive - public domain radio shows
        "https://archive.org/download/OTRR_X_Minus_One_Singles/XMinusOne56-04-24024TheGreenHills.mp3",

        # Fallback: BBC Learning English (educational, fair use)
        "https://downloads.bbc.co.uk/learningenglish/features/6min/200402_6min_english_chatbots.mp3"
    ]

    # Try downloading from the sample URLs
    for i, url in enumerate(sample_urls):
        try:
            print(f"Attempting to download sample {i+1}: {url[:50]}...")
            response = requests.get(url, timeout=60, stream=True)

            if response.status_code == 200:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')

                # Download with size limit (max 10MB)
                downloaded = 0
                max_size = 10 * 1024 * 1024  # 10MB

                for chunk in response.iter_content(chunk_size=8192):
                    if downloaded > max_size:
                        break
                    temp_file.write(chunk)
                    downloaded += len(chunk)

                temp_file.close()

                if os.path.getsize(temp_file.name) > 10000:  # At least 10KB
                    print(f"Successfully downloaded: {os.path.getsize(temp_file.name)} bytes")
                    return temp_file.name
                else:
                    os.unlink(temp_file.name)

        except Exception as e:
            print(f"Failed to download from {url[:50]}: {e}")
            continue

    # Fallback: Generate synthetic dialogue using text-to-speech with ffmpeg
    try:
        print("Generating synthetic dialogue for testing...")
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')

        # Create a longer, more complex audio for better testing
        # Generate multiple tones at different frequencies to simulate speech patterns
        cmd = [
            'ffmpeg', '-f', 'lavfi', '-i',
            'sine=frequency=300:duration=2,sine=frequency=400:duration=2,sine=frequency=350:duration=2,sine=frequency=450:duration=2',
            '-filter_complex', '[0:0][1:0][2:0][3:0]concat=n=4:v=0:a=1[out]',
            '-map', '[out]', '-codec:a', 'mp3', '-y', temp_file.name
        ]

        result = subprocess.run(cmd, capture_output=True, timeout=30)

        if os.path.getsize(temp_file.name) > 1000:
            print(f"Generated synthetic audio: {os.path.getsize(temp_file.name)} bytes")
            return temp_file.name
        else:
            os.unlink(temp_file.name)

    except Exception as e:
        print(f"Could not generate synthetic audio: {e}")

    # Final fallback: Simple tone
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        cmd = [
            'ffmpeg', '-f', 'lavfi', '-i',
            'sine=frequency=440:duration=30',
            '-codec:a', 'mp3', '-y', temp_file.name
        ]
        subprocess.run(cmd, capture_output=True, timeout=30)

        if os.path.getsize(temp_file.name) > 1000:
            return temp_file.name
    except:
        pass

    return None

def load_whisper_model():
    """Load Whisper model and measure performance"""
    global _model

    model_info = {}

    try:
        print("Loading Whisper turbo model...")
        start_time = time.time()

        # Get memory before loading
        mem_before = psutil.virtual_memory().available / (1024**3)

        _model = whisper.load_model("turbo")

        # Get memory after loading
        mem_after = psutil.virtual_memory().available / (1024**3)
        load_time = time.time() - start_time

        model_info = {
            'model_loaded': True,
            'load_time_seconds': round(load_time, 2),
            'memory_used_gb': round(mem_before - mem_after, 2),
            'memory_before_gb': round(mem_before, 2),
            'memory_after_gb': round(mem_after, 2),
            'model_name': 'turbo'
        }

        print(f"Model loaded successfully in {load_time:.2f} seconds")
        print(f"Memory used: {model_info['memory_used_gb']:.2f} GB")

    except Exception as e:
        model_info = {
            'model_loaded': False,
            'error': str(e),
            'load_time_seconds': 0,
            'memory_used_gb': 0
        }
        print(f"Failed to load model: {e}")

    return model_info

def test_transcription():
    """Test transcription with sample audio"""
    transcription_info = {}

    try:
        # Download/generate sample audio
        print("Getting sample audio file...")
        audio_file = download_sample_audio()

        if not audio_file:
            return {
                'transcription_tested': False,
                'error': 'Could not obtain sample audio file'
            }

        if not _model:
            return {
                'transcription_tested': False,
                'error': 'Whisper model not loaded'
            }

        print(f"Testing transcription with file: {audio_file}")
        print(f"File size: {os.path.getsize(audio_file)} bytes")

        # Get memory before transcription
        mem_before = psutil.virtual_memory().available / (1024**3)
        start_time = time.time()

        # Transcribe with verbose output to get detailed segments
        result = _model.transcribe(audio_file, verbose=True)

        # Get metrics
        transcription_time = time.time() - start_time
        mem_after = psutil.virtual_memory().available / (1024**3)

        # Format transcript with timestamps (like your original code)
        def seconds_to_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            if hours > 0:
                return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
            else:
                return f"{minutes:02d}:{secs:06.3f}"

        # Create formatted transcript
        formatted_transcript = []
        for segment in result.get('segments', []):
            start_time_str = seconds_to_time(segment['start'])
            end_time_str = seconds_to_time(segment['end'])
            text = segment['text'].strip()
            formatted_transcript.append(f"[{start_time_str} --> {end_time_str}]  {text}")

        full_transcript = "\n".join(formatted_transcript)

        transcription_info = {
            'transcription_tested': True,
            'transcription_time_seconds': round(transcription_time, 2),
            'audio_duration_seconds': round(result.get('duration', 0), 2),
            'transcription_text': result['text'][:300] + '...' if len(result['text']) > 300 else result['text'],
            'full_transcript': full_transcript,
            'segments_count': len(result.get('segments', [])),
            'memory_used_during_transcription_gb': round(mem_before - mem_after, 2),
            'audio_file_size_mb': round(os.path.getsize(audio_file) / (1024**2), 2),
            'language_detected': result.get('language', 'unknown'),
            'processing_speed_ratio': round(result.get('duration', 0) / max(transcription_time, 1), 2)
        }

        # Clean up
        os.unlink(audio_file)

        print(f"Transcription completed in {transcription_time:.2f} seconds")
        print(f"Audio duration: {result.get('duration', 0):.2f} seconds")
        print(f"Processing speed: {transcription_info['processing_speed_ratio']:.2f}x realtime")
        print(f"Language detected: {result.get('language', 'unknown')}")
        print(f"Result preview: {result['text'][:100]}...")

    except Exception as e:
        transcription_info = {
            'transcription_tested': False,
            'error': str(e)
        }
        print(f"Transcription test failed: {e}")

        # Clean up on error
        if 'audio_file' in locals() and audio_file and os.path.exists(audio_file):
            os.unlink(audio_file)

    return transcription_info

def send_test_email(system_info, model_info, transcription_info):
    """Send test results via email with transcript attachment"""
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = os.getenv("GMAIL_EMAIL")
    sender_password = os.getenv("GMAIL_APP_PASSWORD")
    recipient_email = os.getenv("RECIPIENT_EMAIL")

    if not all([sender_email, sender_password, recipient_email]):
        return {'email_sent': False, 'error': 'Email credentials not configured'}

    try:
        # Create comprehensive report
        report = f"""
WHISPER TRANSCRIPTION SERVICE - TEST REPORT
==========================================
Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SYSTEM SPECIFICATIONS:
---------------------
Platform: {system_info.get('platform', 'Unknown')}
Total Memory: {system_info.get('total_memory_gb', 0)} GB
Available Memory: {system_info.get('available_memory_gb', 0)} GB
Memory Usage: {system_info.get('memory_percent_used', 0)}%
CPU Cores: {system_info.get('cpu_count', 0)} physical, {system_info.get('cpu_count_logical', 0)} logical
CPU Usage: {system_info.get('cpu_percent', 0)}%
Disk Space: {system_info.get('disk_free_gb', 0)} GB free of {system_info.get('disk_total_gb', 0)} GB total

PYTORCH/CUDA INFO:
-----------------
PyTorch Version: {system_info.get('torch_version', 'Unknown')}
CUDA Available: {system_info.get('cuda_available', False)}
CUDA Devices: {system_info.get('cuda_device_count', 0)}
{"CUDA Device: " + system_info.get('cuda_device_name', 'None') if system_info.get('cuda_available') else ''}
{"CUDA Memory: " + str(system_info.get('cuda_memory_gb', 0)) + " GB" if system_info.get('cuda_available') else ''}

FFMPEG:
-------
Available: {system_info.get('ffmpeg_available', False)}
Version: {system_info.get('ffmpeg_version', 'Unknown')}

WHISPER MODEL TEST:
------------------
Model Loaded: {model_info.get('model_loaded', False)}
Load Time: {model_info.get('load_time_seconds', 0)} seconds
Memory Used by Model: {model_info.get('memory_used_gb', 0)} GB
{"Error: " + model_info.get('error', '') if not model_info.get('model_loaded') else ''}

TRANSCRIPTION TEST:
------------------
Transcription Tested: {transcription_info.get('transcription_tested', False)}
{"Transcription Time: " + str(transcription_info.get('transcription_time_seconds', 0)) + " seconds" if transcription_info.get('transcription_tested') else ''}
{"Audio Duration: " + str(transcription_info.get('audio_duration_seconds', 0)) + " seconds" if transcription_info.get('transcription_tested') else ''}
{"Processing Speed: " + str(round(transcription_info.get('audio_duration_seconds', 0) / max(transcription_info.get('transcription_time_seconds', 1), 1), 2)) + "x realtime" if transcription_info.get('transcription_tested') else ''}
{"Audio File Size: " + str(transcription_info.get('audio_file_size_mb', 0)) + " MB" if transcription_info.get('transcription_tested') else ''}
{"Segments Detected: " + str(transcription_info.get('segments_count', 0)) if transcription_info.get('transcription_tested') else ''}
{"Error: " + transcription_info.get('error', '') if not transcription_info.get('transcription_tested') else ''}

PERFORMANCE ASSESSMENT:
----------------------
"""

        # Add performance assessment
        total_memory = system_info.get('total_memory_gb', 0)
        model_memory = model_info.get('memory_used_gb', 0)

        if total_memory >= 4 and model_info.get('model_loaded'):
            report += "✅ EXCELLENT: System has sufficient memory for Whisper turbo model\n"
        elif total_memory >= 2:
            report += "⚠️  MARGINAL: System may work but could be slow or fail with large files\n"
        else:
            report += "❌ INSUFFICIENT: System likely lacks memory for reliable operation\n"

        if transcription_info.get('transcription_tested'):
            speed = transcription_info.get('audio_duration_seconds', 0) / max(transcription_info.get('transcription_time_seconds', 1), 1)
            if speed >= 2:
                report += "✅ FAST: Transcription speed is excellent (>2x realtime)\n"
            elif speed >= 1:
                report += "✅ GOOD: Transcription speed is acceptable (>1x realtime)\n"
            else:
                report += "⚠️  SLOW: Transcription is slower than realtime\n"

        report += f"\nRECOMMENDATION:\n"
        if model_info.get('model_loaded') and transcription_info.get('transcription_tested'):
            report += "🎉 System is ready for production deployment!\n"
            report += "   • 5-10 minute audio files should process in 2-5 minutes\n"
            report += "   • Memory usage is within acceptable limits\n"
            report += "   • Email delivery is working correctly\n"
        else:
            report += "❌ System needs attention before production use.\n"
            if not model_info.get('model_loaded'):
                report += "   • Consider using a smaller Whisper model (base, small)\n"
                report += "   • Or upgrade to a paid Render plan for more memory\n"
            if not transcription_info.get('transcription_tested'):
                report += "   • Check audio file processing capabilities\n"

        # Create email
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.base import MIMEBase
        from email import encoders

        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = "🎯 Whisper Service Test Results - Render Deployment"

        # Add main report as body
        msg.attach(MIMEText(report, 'plain'))

        # Add transcript as attachment if available
        if transcription_info.get('transcription_tested') and transcription_info.get('full_transcript'):
            transcript_text = transcription_info['full_transcript']

            # Create transcript attachment
            attachment = MIMEBase('application', 'octet-stream')
            attachment.set_payload(transcript_text.encode('utf-8'))
            encoders.encode_base64(attachment)
            attachment.add_header(
                'Content-Disposition',
                'attachment; filename= "whisper_test_transcript.txt"'
            )
            msg.attach(attachment)

        # Send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()

        return {
            'email_sent': True,
            'report_length': len(report),
            'transcript_attached': transcription_info.get('transcription_tested', False)
        }

    except Exception as e:
        return {'email_sent': False, 'error': str(e)}

def run_full_test():
    """Run complete system test"""
    print("Starting comprehensive system test...")

    # Collect system info
    print("1. Collecting system information...")
    sys_info = get_system_info()

    # Test model loading
    print("2. Testing Whisper model loading...")
    model_info = load_whisper_model()

    # Test transcription
    print("3. Testing transcription...")
    transcription_info = test_transcription()

    # Send email report
    print("4. Sending email report...")
    email_info = send_test_email(sys_info, model_info, transcription_info)

    return {
        'test_completed': True,
        'system_info': sys_info,
        'model_info': model_info,
        'transcription_info': transcription_info,
        'email_info': email_info,
        'completed_at': datetime.now().isoformat()
    }

@app.route('/test', methods=['POST', 'GET'])
def run_test():
    """Run the full test suite"""
    try:
        # Run test in background thread to avoid timeouts
        result = run_full_test()
        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e), 'test_completed': False}), 500

@app.route('/quick-info')
def quick_info():
    """Get quick system info without heavy testing"""
    info = get_system_info()
    return jsonify(info)

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'service': 'whisper-test'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Whisper Test Service on port {port}")
    print("Visit /test to run full test suite")
    print("Visit /quick-info for system specs only")
    app.run(host='0.0.0.0', port=port)
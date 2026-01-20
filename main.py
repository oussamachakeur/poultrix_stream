from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from PIL import Image
import io
import os
import numpy as np
import cv2
import requests
from threading import Thread, Lock
from queue import Queue
import time
import paho.mqtt.client as mqtt
import ssl
import json
from datetime import datetime
import torch
import subprocess
import shutil
from pathlib import Path

import shutil

# ============================================
# AWS IoT Configuration
# ============================================

ENDPOINT = "a1s6wwnw9j50mk-ats.iot.ap-southeast-1.amazonaws.com"
CCTV_TOPIC = 'poultry/cctv/detections'

DEVICE_CONFIG = {
    "device_id": "301",
    "coop_id": 1,
    "farm_id": 1,
    "device_name": "esp32_cam_1"
}

CERT_PATH = "sensors_policy/device-certificate.pem.crt"
KEY_PATH = "sensors_policy/private.pem.key"
CA_PATH = "sensors_policy/AmazonRootCA1.pem"

INTRUDER_CLASSES = ['person', 'cat', 'dog', 'fox']
BIRD_CLASS = 'bird'
CONFIDENCE_THRESHOLD = 0.60
DETECTION_COOLDOWN = 10
CHICKEN_ALERT_COOLDOWN = 30

# ============================================
# PERFORMANCE SETTINGS
# ============================================

INFERENCE_FPS = 3
STREAM_FPS = 10
YOLO_INPUT_SIZE = 320

# ============================================
# HLS CONFIGURATION
# ============================================

import shutil

# Detect if running on Windows or Linux (Docker)
if os.name == 'nt':
    FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"
else:
    # On Linux/Railway, we just use the command "ffmpeg" (installed via Dockerfile)
    FFMPEG_PATH = shutil.which("ffmpeg") or "ffmpeg"

HLS_OUTPUT_DIR = "hls_streams"
os.makedirs(HLS_OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{HLS_OUTPUT_DIR}/raw", exist_ok=True)
os.makedirs(f"{HLS_OUTPUT_DIR}/annotated", exist_ok=True)

ffmpeg_processes = {
    'raw': None,
    'annotated': None
}

# ============================================
# Load Model
# ============================================

current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, 'intruder.pt')
CHICKEN_DETECTOR_PATH = os.path.join(current_dir, 'dead.pt')
HEALTH_CLASSIFIER_PATH = os.path.join(current_dir, 'normal_abnormal.pt')
DEAD_ALIVE_PATH = os.path.join(current_dir, 'dead_alive.pt')

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Intruder model not found: {MODEL_PATH}")

print("Loading YOLOv8 models...")
model_intruder = YOLO(MODEL_PATH)

model_chicken_detector = None
model_health_classifier = None
model_dead_alive = None

if os.path.exists(CHICKEN_DETECTOR_PATH):
    model_chicken_detector = YOLO(CHICKEN_DETECTOR_PATH)
    print("✅ Chicken detector loaded (dead.pt)")

if os.path.exists(HEALTH_CLASSIFIER_PATH):
    model_health_classifier = YOLO(HEALTH_CLASSIFIER_PATH)
    print("✅ Health classifier loaded (normal_abnormal.pt)")

if os.path.exists(DEAD_ALIVE_PATH):
    model_dead_alive = YOLO(DEAD_ALIVE_PATH)
    print("✅ Dead/Alive detector loaded (dead_alive.pt)")

# GPU setup
if torch.cuda.is_available():
    model_intruder.to("cuda")
    if model_chicken_detector:
        model_chicken_detector.to("cuda")
    if model_health_classifier:
        model_health_classifier.to("cuda")
    if model_dead_alive:
        model_dead_alive.to("cuda")
    print(f"✅ GPU ENABLED: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ WARNING: No GPU detected, using CPU (will be slow)")

print(f"Primary model device: {model_intruder.device}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# SHARED STATE
# ============================================

# Updated to use the public Ngrok tunnel - Can be updated dynamically via /update-camera-url
ESP32_CAM_URL = os.environ.get("CAMERA_URL", "rtsp://poultrix:poultrix123@0.tcp.ap.ngrok.io:15655/stream2")

class SharedState:
    def __init__(self):
        self.lock = Lock()
        self.raw_frame = None
        self.detection_results = None
        self.annotated_frame = None
        self.detection_active = False
        self.total_detections = 0
        self.fps_inference = 0.0
        self.fps_stream = 0.0
        
        self.metrics = {
            'intruder_detections': 0,
            'chicken_detections': 0,
            'healthy_chickens': 0,
            'sick_chickens': 0,
            'dead_chickens': 0,
            'alerts_sent': 0,
            'alerts_blocked_cooldown': 0,
            'frames_processed': 0,
            'inference_errors': 0,
            'last_detection_time': None,
            'uptime_start': time.time(),
            'detections_by_class': {},
            'cascade_triggers': {
                'bird_detected': 0,
                'chicken_confirmed': 0,
                'health_checked': 0,
                'dead_alive_checked': 0
            }
        }

state = SharedState()

mqtt_client = None
mqtt_connected = False
last_detection_time = {}
last_chicken_alert_time = {}

alert_queue = Queue(maxsize=100)

# ============================================
# MQTT Functions
# ============================================

def on_connect(client, userdata, flags, rc):
    global mqtt_connected
    mqtt_connected = (rc == 0)
    print(f"{'✅' if rc == 0 else '❌'} AWS IoT: {'Connected' if rc == 0 else f'Failed ({rc})'}")

def on_publish(client, userdata, mid):
    print(f"📤 Alert sent (ID: {mid})")

def setup_mqtt():
    global mqtt_client
    try:
        # Compatible with both old and new paho-mqtt versions
        try:
            mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
        except AttributeError:
            # Fallback for older paho-mqtt versions
            mqtt_client = mqtt.Client()
        
        mqtt_client.on_connect = on_connect
        mqtt_client.on_publish = on_publish
        
        mqtt_client.tls_set(
            ca_certs=CA_PATH,
            certfile=CERT_PATH,
            keyfile=KEY_PATH,
            tls_version=ssl.PROTOCOL_TLSv1_2
        )
        
        mqtt_client.connect(ENDPOINT, 8883, 60)
        mqtt_client.loop_start()
        return True
    except Exception as e:
        print(f"❌ MQTT failed: {e}")
        return False

def send_detection_to_aws(class_name, confidence, model_type='intruder', chicken_id=None, health_status=None):
    """Send detection alert to AWS IoT"""
    global last_detection_time, last_chicken_alert_time
    
    if not mqtt_connected:
        print("⚠️ AWS not connected, skipping alert")
        return
    
    current_time = time.time()
    
    if model_type == 'chicken_health':
        cooldown_key = f"chicken_{chicken_id}"
        cooldown_dict = last_chicken_alert_time
        cooldown_period = CHICKEN_ALERT_COOLDOWN
    else:
        cooldown_key = class_name
        cooldown_dict = last_detection_time
        cooldown_period = DETECTION_COOLDOWN
    
    if cooldown_key in cooldown_dict:
        time_since_last = current_time - cooldown_dict[cooldown_key]
        if time_since_last < cooldown_period:
            with state.lock:
                state.metrics['alerts_blocked_cooldown'] += 1
            print(f"⏰ Cooldown active for {cooldown_key} ({time_since_last:.1f}s < {cooldown_period}s)")
            return
    
    cooldown_dict[cooldown_key] = current_time
    
    payload = {
        **DEVICE_CONFIG,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "detection_type": model_type,
        "class": class_name,
        "confidence": round(confidence, 2)
    }
    
    if model_type == 'chicken_health':
        payload['chicken_id'] = chicken_id
        payload['health_status'] = health_status
    
    try:
        result = mqtt_client.publish(CCTV_TOPIC, json.dumps(payload), qos=1)
        
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            with state.lock:
                state.metrics['alerts_sent'] += 1
                state.metrics['last_detection_time'] = datetime.now().isoformat()
            print(f"✅ Alert sent: {class_name} ({confidence:.2f})")
        else:
            print(f"❌ Publish failed: {result.rc}")
    except Exception as e:
        print(f"❌ Alert error: {e}")

# ============================================
# THREAD 1: Frame Capture
# ============================================

def thread_capture_frames():
    print("🎥 [THREAD 1] Frame capture started")
    consecutive_errors = 0
    max_errors = 5
    
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000|analyzeduration;500000|probesize;500000"
    
    while True:
        cap = None
        try:
            # Check if URL is valid before attempting connection
            if not ESP32_CAM_URL or not ESP32_CAM_URL.startswith("rtsp://"):
                 print(f"⚠️ Invalid ESP32_CAM_URL (must start with rtsp://): {ESP32_CAM_URL}. Waiting...")
                 time.sleep(10)
                 continue

            print(f"🔌 Connecting to stream: {ESP32_CAM_URL}")
            cap = cv2.VideoCapture(ESP32_CAM_URL, cv2.CAP_FFMPEG)
            
            # Reduce buffer size to minimize lag
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not cap.isOpened():
                raise Exception("Could not open RTSP stream - Check Ngrok URL or Camera Power")
            
            print("✓ ESP32 connected")
            consecutive_errors = 0
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("⚠️ Stream read returned False (Stream ended/lagged)")
                    break
                
                if frame is not None and frame.size > 0:
                    with state.lock:
                        state.raw_frame = frame
                    
                    frame_count += 1
                    if frame_count % 100 == 0:
                        print(f"📊 Captured {frame_count} frames")
                else:
                    print("⚠️ Empty frame received")
                    break
                
                # Small sleep to prevent CPU spiking
                time.sleep(0.01)
            
        except Exception as e:
            consecutive_errors += 1
            print(f"⚠️ Capture error ({consecutive_errors}/{max_errors}): {e}")
            
        finally:
            # Ensure resource release
            if cap:
                cap.release()
            
            if consecutive_errors >= max_errors:
                print(f"❌ Connection unstable. Waiting 30s cooldown...")
                time.sleep(30)
                consecutive_errors = 0
            else:
                print("🔄 Reconnecting in 2s...")
                time.sleep(2)

# ============================================
# THREAD 2: GPU Inference
# ============================================


def thread_gpu_inference():
    print(f"🚀 [THREAD 2] Smart Cascading Pipeline @ {INFERENCE_FPS} FPS")
    print(f"   Waiting for first frame...")
    
    frame_count = 0
    fps_start = time.time()
    last_inference = 0
    inference_interval = 1.0 / INFERENCE_FPS
    chicken_id_counter = 0
    first_frame_received = False
    
    while True:
        current_time = time.time()
        
        if current_time - last_inference < inference_interval:
            time.sleep(0.01)
            continue
        
        last_inference = current_time
        
        with state.lock:
            if state.raw_frame is None:
                continue
            frame = state.raw_frame.copy()
            
        if not first_frame_received:
            first_frame_received = True
            print(f"✅ [THREAD 2] First frame received! Starting inference...")
        
        orig_h, orig_w = frame.shape[:2]
        small = cv2.resize(frame, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))
        
        try:
            t0 = time.time()
            
            with torch.no_grad():
                results_intruder = model_intruder(
                    small, 
                    imgsz=YOLO_INPUT_SIZE, 
                    device=0 if torch.cuda.is_available() else 'cpu',
                    half=torch.cuda.is_available(),
                    verbose=False
                )
            
            scale_x = orig_w / YOLO_INPUT_SIZE
            scale_y = orig_h / YOLO_INPUT_SIZE
            
            detections = []
            detection_found = False
            bird_detected = False
            bird_boxes = []
            
            for result in results_intruder:
                boxes = result.boxes
                for box in boxes:
                    x1_small, y1_small, x2_small, y2_small = box.xyxy[0].tolist()
                    
                    x1 = int(x1_small * scale_x)
                    y1 = int(y1_small * scale_y)
                    x2 = int(x2_small * scale_x)
                    y2 = int(y2_small * scale_y)
                    
                    x1 = max(0, min(orig_w - 1, x1))
                    y1 = max(0, min(orig_h - 1, y1))
                    x2 = max(0, min(orig_w - 1, x2))
                    y2 = max(0, min(orig_h - 1, y2))
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = model_intruder.names[class_id]
                    
                    if confidence > 0.5:
                        detection_found = True
                        
                        with state.lock:
                            state.metrics['detections_by_class'][class_name] = \
                                state.metrics['detections_by_class'].get(class_name, 0) + 1
                            if class_name in INTRUDER_CLASSES:
                                state.metrics['intruder_detections'] += 1
                        
                        if class_name == BIRD_CLASS:
                            bird_detected = True
                            bird_boxes.append({'bbox': (x1, y1, x2, y2)})
                            with state.lock:
                                state.metrics['cascade_triggers']['bird_detected'] += 1
                        
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'class': class_name,
                            'confidence': confidence,
                            'model': 'intruder'
                        })
                        
                        if class_name in INTRUDER_CLASSES and confidence > CONFIDENCE_THRESHOLD:
                            try:
                                alert_queue.put_nowait((class_name, confidence, 'intruder', None, None))
                            except:
                                pass
            
            # Chicken health cascade logic
            if bird_detected and model_chicken_detector:
                for bird_info in bird_boxes:
                    x1, y1, x2, y2 = bird_info['bbox']
                    crop = frame[y1:y2, x1:x2]
                    
                    if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                        continue
                    
                    crop_resized = cv2.resize(crop, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))
                    
                    with torch.no_grad():
                        chicken_results = model_chicken_detector(
                            crop_resized,
                            imgsz=YOLO_INPUT_SIZE,
                            device=0 if torch.cuda.is_available() else 'cpu',
                            half=torch.cuda.is_available(),
                            verbose=False
                        )
                    
                    chicken_confirmed = False
                    for result in chicken_results:
                        if result.boxes is not None and len(result.boxes) > 0:
                            chicken_confirmed = True
                            chicken_id_counter += 1
                            with state.lock:
                                state.metrics['chicken_detections'] += 1
                                state.metrics['cascade_triggers']['chicken_confirmed'] += 1
                            break
                    
                    if not chicken_confirmed:
                        continue
                    
                    if model_health_classifier:
                        with state.lock:
                            state.metrics['cascade_triggers']['health_checked'] += 1
                        
                        with torch.no_grad():
                            health_results = model_health_classifier(
                                crop_resized,
                                imgsz=YOLO_INPUT_SIZE,
                                device=0 if torch.cuda.is_available() else 'cpu',
                                half=torch.cuda.is_available(),
                                verbose=False
                            )
                        
                        health_status = None
                        health_conf = 0.0
                        
                        for result in health_results:
                            if result.boxes is not None and len(result.boxes) > 0:
                                box = result.boxes[0]
                                class_id = int(box.cls[0])
                                health_status = model_health_classifier.names[class_id]
                                health_conf = float(box.conf[0])
                                break
                        
                        if health_status and 'normal' in health_status.lower():
                            with state.lock:
                                state.metrics['healthy_chickens'] += 1
                            
                            detections.append({
                                'bbox': bird_info['bbox'],
                                'class': 'chicken_healthy',
                                'confidence': health_conf,
                                'model': 'health'
                            })
                            continue
                        
                        if health_status and 'abnormal' in health_status.lower() and model_dead_alive:
                            with state.lock:
                                state.metrics['cascade_triggers']['dead_alive_checked'] += 1
                            
                            with torch.no_grad():
                                dead_alive_results = model_dead_alive(
                                    crop_resized,
                                    imgsz=YOLO_INPUT_SIZE,
                                    device=0 if torch.cuda.is_available() else 'cpu',
                                    half=torch.cuda.is_available(),
                                    verbose=False
                                )
                            
                            for result in dead_alive_results:
                                if result.boxes is not None and len(result.boxes) > 0:
                                    box = result.boxes[0]
                                    class_id = int(box.cls[0])
                                    status = model_dead_alive.names[class_id]
                                    conf = float(box.conf[0])
                                    
                                    with state.lock:
                                        if status == 'dead':
                                            state.metrics['dead_chickens'] += 1
                                        else:
                                            state.metrics['sick_chickens'] += 1
                                    
                                    detections.append({
                                        'bbox': bird_info['bbox'],
                                        'class': f'chicken_{status}',
                                        'confidence': conf,
                                        'model': 'dead_alive',
                                        'health': health_status
                                    })
                                    
                                    if conf > CONFIDENCE_THRESHOLD:
                                        try:
                                            alert_queue.put_nowait((
                                                status,
                                                conf,
                                                'chicken_health',
                                                chicken_id_counter,
                                                health_status
                                            ))
                                        except:
                                            pass
                                    break
            
            with state.lock:
                state.detection_results = detections
                state.detection_active = detection_found
                if detection_found:
                    state.total_detections += 1
                state.metrics['frames_processed'] += 1
            
            inference_time = time.time() - t0
            
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_start
                with state.lock:
                    state.fps_inference = frame_count / elapsed
                print(f"🔥 Inference: {state.fps_inference:.1f} FPS | {inference_time*1000:.1f}ms")
                frame_count = 0
                fps_start = time.time()
                
        except Exception as e:
            print(f"⚠️ Inference error: {e}")
            with state.lock:
                state.metrics['inference_errors'] += 1
            time.sleep(0.1)

# ============================================
# THREAD 3: Alert Sender
# ============================================

def thread_send_alerts():
    print("📡 [THREAD 3] Alert sender started")
    
    while True:
        try:
            alert_data = alert_queue.get(timeout=1.0)
            
            if len(alert_data) == 5:
                class_name, confidence, model_type, chicken_id, health_status = alert_data
            else:
                class_name, confidence = alert_data[:2]
                model_type = 'intruder'
                chicken_id = None
                health_status = None
            
            if not mqtt_connected:
                continue
            
            send_detection_to_aws(class_name, confidence, model_type, chicken_id, health_status)
            
        except:
            pass

# ============================================
# THREAD 4 & 5: Streaming Functions (MJPEG)
# ============================================

def generate_stream():
    """AI Annotated Stream"""
    
    color_map = {
        "person": (0, 0, 255),
        "cat": (255, 0, 0),
        "dog": (0, 255, 255),
        "fox": (255, 0, 255),
        "bird": (0, 255, 0),
        "chicken_healthy": (0, 255, 0),
        "chicken_alive": (0, 200, 200),
        "chicken_dead": (0, 0, 128)
    }
    
    frame_count = 0
    fps_start = time.time()
    stream_interval = 1.0 / STREAM_FPS
    last_stream = 0
    
    while True:
        current_time = time.time()
        
        if current_time - last_stream < stream_interval:
            time.sleep(0.005)
            continue
        
        last_stream = current_time
        
        with state.lock:
            if state.raw_frame is None:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Connecting to ESP32...", (80, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                frame = placeholder
                detections = []
                detection_active = False
                fps_inf = 0.0
                fps_str = 0.0
            else:
                frame = state.raw_frame.copy()
                detections = state.detection_results or []
                detection_active = state.detection_active
                fps_inf = state.fps_inference
                fps_str = state.fps_stream
        
        annotated = frame.copy()
        detection_count = len(detections)
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            confidence = det['confidence']
            
            color = color_map.get(class_name, (0, 255, 0))
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            if 'health' in det:
                label = f"{class_name} {confidence:.2f} ({det['health']})"
            else:
                label = f"{class_name} {confidence:.2f}"
            
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        status = f"Det:{detection_count} | Inf:{fps_inf:.1f} Str:{fps_str:.1f} FPS"
        if mqtt_connected:
            status += " | AWS✓"
        
        cv2.putText(annotated, status, (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        ret, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - fps_start
            with state.lock:
                state.fps_stream = frame_count / elapsed
            frame_count = 0
            fps_start = time.time()

def generate_raw_stream():
    """Raw Stream without AI annotations"""
    
    frame_count = 0
    fps_start = time.time()
    stream_interval = 1.0 / STREAM_FPS
    last_stream = 0
    
    while True:
        current_time = time.time()
        
        if current_time - last_stream < stream_interval:
            time.sleep(0.005)
            continue
        
        last_stream = current_time
        
        with state.lock:
            if state.raw_frame is None:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Connecting to ESP32...", (80, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                frame = placeholder
            else:
                frame = state.raw_frame.copy()
        
        fps = frame_count / (time.time() - fps_start) if (time.time() - fps_start) > 0 else 0
        cv2.putText(frame, f"RAW {fps:.1f} FPS", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        
        frame_count += 1
        if frame_count % 30 == 0:
            frame_count = 0
            fps_start = time.time()

# ============================================
# THREAD 6 & 7: HLS CONVERTERS (NEW!)
# ============================================

def thread_hls_converter(stream_type='raw'):
    """Convert MJPEG stream to HLS for ngrok/phone compatibility"""
    print(f"🎬 [HLS-{stream_type.upper()}] Converter started")
    
    output_dir = f"{HLS_OUTPUT_DIR}/{stream_type}"
    playlist_file = f"{output_dir}/stream.m3u8"
    
    # Verify FFmpeg exists
    if not os.path.exists(FFMPEG_PATH):
        print(f"❌ FFmpeg not found at: {FFMPEG_PATH}")
        return
    
    print(f"   Using FFmpeg: {FFMPEG_PATH}")
    
    # FFmpeg command for HLS conversion - SIMPLIFIED FOR DEBUGGING
    cmd = [
        FFMPEG_PATH,
        '-f', 'image2pipe',
        '-vcodec', 'mjpeg',
        '-r', '10',  # Input framerate
        '-i', 'pipe:0',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-g', '30',
        '-sc_threshold', '0',
        '-b:v', '2M',  # Bitrate
        '-f', 'hls',
        '-hls_time', '2',
        '-hls_list_size', '3',
        '-hls_flags', 'delete_segments+append_list',
        '-hls_segment_filename', f'{output_dir}/segment_%03d.ts',
        '-loglevel', 'info',  # Show FFmpeg errors
        playlist_file,
        '-y'
    ]
    
    while True:
        try:
            print(f"🔄 Starting HLS conversion for {stream_type}...")
            print(f"   Output: {playlist_file}")
            
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=10**8,
                universal_newlines=False
            )
            
            ffmpeg_processes[stream_type] = process
            
            # Monitor FFmpeg output in separate thread
            def monitor_ffmpeg():
                for line in process.stdout:
                    try:
                        msg = line.decode('utf-8', errors='ignore').strip()
                        if msg and ('error' in msg.lower() or 'warning' in msg.lower()):
                            print(f"   [FFmpeg {stream_type}] {msg}")
                    except:
                        pass
            
            import threading
            monitor_thread = threading.Thread(target=monitor_ffmpeg, daemon=True)
            monitor_thread.start()
            
            # Feed frames to FFmpeg
            if stream_type == 'raw':
                generator = generate_raw_stream()
            else:
                generator = generate_stream()
            
            frame_count = 0
            for frame_data in generator:
                if process.poll() is not None:
                    print(f"⚠️ FFmpeg {stream_type} process died (exit code: {process.returncode})")
                    break
                
                # Extract JPEG data
                if b'\r\n\r\n' in frame_data:
                    parts = frame_data.split(b'\r\n\r\n')
                    if len(parts) > 1:
                        # Get the actual image data
                        jpeg_payload = parts[1].split(b'\r\n')[0]
                        
                        # Only write if we have valid data (checks for JPEG Start/End bytes)
                        if len(jpeg_payload) > 0 and jpeg_payload.startswith(b'\xff\xd8') and jpeg_payload.endswith(b'\xff\xd9'):
                            try:
                                process.stdin.write(jpeg_payload)
                                process.stdin.flush()
                                frame_count += 1
                                
                                if frame_count == 1:
                                    print(f"   ✅ First frame sent to FFmpeg ({stream_type})")
                                elif frame_count == 10:
                                    print(f"   📊 {stream_type}: 10 frames sent, checking output...")
                                    if os.path.exists(playlist_file):
                                        print(f"   ✅ {stream_type}: HLS playlist created!")
                                    else:
                                        print(f"   ⚠️ {stream_type}: No playlist yet, still buffering...")
                                elif frame_count % 100 == 0:
                                    print(f"   📊 HLS {stream_type}: {frame_count} frames encoded")
                            except Exception as e:
                                print(f"   ⚠️ Write error {stream_type}: {e}")
                                break
                    
        except Exception as e:
            print(f"⚠️ HLS {stream_type} error: {e}")
            if ffmpeg_processes[stream_type]:
                try:
                    ffmpeg_processes[stream_type].terminate()
                except:
                    pass
            time.sleep(5)

# ============================================
# FastAPI Startup
# ============================================

@app.on_event("startup")
async def startup():
    print("\n" + "="*70)
    print("🚀 STARTING DETECTION PIPELINE WITH HLS STREAMING")
    print("="*70)
    
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ NO GPU - Using CPU (performance will be limited)")
    
    print("\n🔧 Initializing MQTT connection...")
    setup_mqtt()
    time.sleep(2)
    
    print("🔧 Starting background threads...")
    
    t1 = Thread(target=thread_capture_frames, daemon=True, name="CaptureThread")
    t2 = Thread(target=thread_gpu_inference, daemon=True, name="InferenceThread")
    t3 = Thread(target=thread_send_alerts, daemon=True, name="AlertThread")
    t4 = Thread(target=thread_hls_converter, args=('raw',), daemon=True, name="HLS-Raw")
    t5 = Thread(target=thread_hls_converter, args=('annotated',), daemon=True, name="HLS-Annotated")
    
    t1.start()
    print("   ✅ Thread 1 (Frame Capture) started")
    time.sleep(0.5)
    
    t2.start()
    print("   ✅ Thread 2 (GPU Inference) started")
    time.sleep(0.5)
    
    t3.start()
    print("   ✅ Thread 3 (Alert Sender) started")
    time.sleep(0.5)
    
    t4.start()
    print("   ✅ Thread 4 (HLS Raw) started")
    time.sleep(0.5)
    
    t5.start()
    print("   ✅ Thread 5 (HLS Annotated) started")
    
    time.sleep(2)
    
    # Mount HLS directory
    app.mount("/hls", StaticFiles(directory=HLS_OUTPUT_DIR), name="hls")
    
    print(f"\n📊 Configuration:")
    print(f"   Inference: {INFERENCE_FPS} FPS @ {YOLO_INPUT_SIZE}px")
    print(f"   Streaming: {STREAM_FPS} FPS")
    print(f"   Device: {DEVICE_CONFIG['device_name']}")
    print(f"   Stream URL: {ESP32_CAM_URL[:60]}...")
    print("="*70)
    print("✅ System ready!")
    print("\n🌐 Access streams at:")
    print("   MJPEG (localhost only): http://127.0.0.1:8000/stream")
    print("   HLS Raw (works on ngrok): http://127.0.0.1:8000/hls/player/raw")
    print("   HLS AI (works on ngrok): http://127.0.0.1:8000/hls/player/annotated")
    print("="*70 + "\n")

@app.on_event("shutdown")
async def shutdown():
    """Stop FFmpeg processes"""
    print("🛑 Stopping HLS converters...")
    for name, process in ffmpeg_processes.items():
        if process:
            try:
                process.terminate()
                print(f"   ✅ {name} stopped")
            except:
                pass

# ============================================
# API Endpoints
# ============================================

@app.get("/")
def root():
    return {
        "message": "Smart Cascading Detection Pipeline with HLS",
        "status": "running",
        "gpu_available": torch.cuda.is_available(),
        "endpoints": {
            "/stream": "AI Annotated MJPEG (localhost only)",
            "/stream/raw": "Raw MJPEG (localhost only)",
            "/hls/player/raw": "Raw HLS player (works on ngrok/phone)",
            "/hls/player/annotated": "AI HLS player (works on ngrok/phone)",
            "/hls/raw/stream.m3u8": "Raw HLS playlist",
            "/hls/annotated/stream.m3u8": "AI HLS playlist",
            "/status": "System status",
            "/metrics": "Detailed metrics"
        }
    }

@app.get("/stream")
async def stream():
    """AI Annotated Stream (MJPEG - localhost only)"""
    return StreamingResponse(
        generate_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }
    )

@app.get("/stream/raw")
async def stream_raw():
    """Raw video stream (MJPEG - localhost only)"""
    return StreamingResponse(
        generate_raw_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }
    )

# ============================================
# HLS PLAYER ENDPOINTS (NEW!)
# ============================================

@app.get("/hls/player/raw")
async def hls_player_raw():
    """HTML player for raw HLS stream - WORKS ON NGROK"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Raw CCTV Stream</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
        <meta http-equiv="Pragma" content="no-cache">
        <meta http-equiv="Expires" content="0">
        <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
        <style>
            body { 
                margin: 0; 
                background: #000; 
                display: flex; 
                justify-content: center; 
                align-items: center; 
                height: 100vh; 
                flex-direction: column;
                font-family: Arial, sans-serif;
            }
            video { 
                max-width: 95vw; 
                max-height: 85vh;
                border: 2px solid #fff;
            }
            h2 { 
                color: white; 
                margin: 10px;
            }
            .status {
                color: #0f0;
                font-size: 14px;
                margin: 5px;
            }
            .error {
                color: #f00;
            }
        </style>
    </head>
    <body>
        <h2>📹 Raw CCTV Stream</h2>
        <div class="status" id="status">Loading...</div>
        <video id="video" controls autoplay muted playsinline></video>
        <script>
            console.log('Player page loaded');
            
            var video = document.getElementById('video');
            var status = document.getElementById('status');
            
            // Get the current URL and construct the stream URL
            var baseUrl = window.location.protocol + '//' + window.location.host;
            var videoSrc = baseUrl + '/hls/raw/stream.m3u8';
            
            console.log('Stream URL:', videoSrc);
            status.textContent = 'Connecting to: ' + videoSrc;
            
            setTimeout(function() {
                if (Hls.isSupported()) {
                    console.log('HLS.js is supported');
                    var hls = new Hls({
                        liveSyncDurationCount: 3,
                        liveMaxLatencyDurationCount: 5,
                        maxBufferLength: 10,
                        debug: true
                    });
                    
                    hls.on(Hls.Events.MANIFEST_PARSED, function() {
                        console.log('Manifest parsed');
                        status.textContent = '✅ Stream connected';
                        video.play().catch(function(e) {
                            console.error('Play error:', e);
                            status.textContent = '⚠️ Click video to play';
                        });
                    });
                    
                    hls.on(Hls.Events.ERROR, function(event, data) {
                        console.error('HLS Error:', data);
                        status.textContent = '⚠️ Error: ' + data.details;
                        status.className = 'status error';
                    });
                    
                    console.log('Loading source...');
                    hls.loadSource(videoSrc);
                    hls.attachMedia(video);
                } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
                    console.log('Native HLS support');
                    video.src = videoSrc;
                    video.addEventListener('loadedmetadata', function() {
                        status.textContent = '✅ Stream connected (native HLS)';
                    });
                } else {
                    status.textContent = '❌ HLS not supported in this browser';
                    status.className = 'status error';
                }
            }, 1000);
        </script>
    </body>
    </html>
    """
    return Response(content=html, media_type="text/html")

@app.get("/hls/player/annotated")
async def hls_player_annotated():
    """HTML player for AI annotated HLS stream - WORKS ON NGROK"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Detection Stream</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
        <meta http-equiv="Pragma" content="no-cache">
        <meta http-equiv="Expires" content="0">
        <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
        <style>
            body { 
                margin: 0; 
                background: #000; 
                display: flex; 
                justify-content: center; 
                align-items: center; 
                height: 100vh; 
                flex-direction: column;
                font-family: Arial, sans-serif;
            }
            video { 
                max-width: 95vw; 
                max-height: 85vh;
                border: 2px solid #0f0;
            }
            h2 { 
                color: #0f0; 
                margin: 10px;
            }
            .status {
                color: #0f0;
                font-size: 14px;
                margin: 5px;
            }
            .error {
                color: #f00;
            }
        </style>
    </head>
    <body>
        <h2>🤖 AI Detection Stream</h2>
        <div class="status" id="status">Loading...</div>
        <video id="video" controls autoplay muted playsinline></video>
        <script>
            console.log('AI Player page loaded');
            
            var video = document.getElementById('video');
            var status = document.getElementById('status');
            
            // Get the current URL and construct the stream URL
            var baseUrl = window.location.protocol + '//' + window.location.host;
            var videoSrc = baseUrl + '/hls/annotated/stream.m3u8';
            
            console.log('Stream URL:', videoSrc);
            status.textContent = 'Connecting to: ' + videoSrc;
            
            setTimeout(function() {
                if (Hls.isSupported()) {
                    console.log('HLS.js is supported');
                    var hls = new Hls({
                        liveSyncDurationCount: 3,
                        liveMaxLatencyDurationCount: 5,
                        maxBufferLength: 10,
                        debug: true
                    });
                    
                    hls.on(Hls.Events.MANIFEST_PARSED, function() {
                        console.log('Manifest parsed');
                        status.textContent = '✅ AI Stream connected';
                        video.play().catch(function(e) {
                            console.error('Play error:', e);
                            status.textContent = '⚠️ Click video to play';
                        });
                    });
                    
                    hls.on(Hls.Events.ERROR, function(event, data) {
                        console.error('HLS Error:', data);
                        status.textContent = '⚠️ Error: ' + data.details;
                        status.className = 'status error';
                    });
                    
                    console.log('Loading source...');
                    hls.loadSource(videoSrc);
                    hls.attachMedia(video);
                } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
                    console.log('Native HLS support');
                    video.src = videoSrc;
                    video.addEventListener('loadedmetadata', function() {
                        status.textContent = '✅ AI Stream connected (native HLS)';
                    });
                } else {
                    status.textContent = '❌ HLS not supported in this browser';
                    status.className = 'status error';
                }
            }, 1000);
        </script>
    </body>
    </html>
    """
    return Response(content=html, media_type="text/html")

# NEW MINIMAL PLAYERS (video only, no decoration)
@app.get("/embed/raw")
async def embed_raw():
    """Minimal embedded player - RAW stream (just video, no UI)"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
        <style>
            * { margin: 0; padding: 0; }
            body { background: #000; overflow: hidden; }
            video { width: 100vw; height: 100vh; object-fit: contain; }
        </style>
    </head>
    <body>
        <video id="video" controls autoplay muted playsinline></video>
        <script>
            const video = document.getElementById('video');
            const streamUrl = window.location.protocol + '//' + window.location.host + '/hls/raw/stream.m3u8';
            
            if (Hls.isSupported()) {
                const hls = new Hls({ liveSyncDurationCount: 3, liveMaxLatencyDurationCount: 5 });
                hls.loadSource(streamUrl);
                hls.attachMedia(video);
                hls.on(Hls.Events.MANIFEST_PARSED, () => video.play());
            } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
                video.src = streamUrl;
            }
        </script>
    </body>
    </html>
    """
    return Response(content=html, media_type="text/html")

@app.get("/embed/annotated")
async def embed_annotated():
    """Minimal embedded player - AI stream (just video, no UI)"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
        <style>
            * { margin: 0; padding: 0; }
            body { background: #000; overflow: hidden; }
            video { width: 100vw; height: 100vh; object-fit: contain; }
        </style>
    </head>
    <body>
        <video id="video" controls autoplay muted playsinline></video>
        <script>
            const video = document.getElementById('video');
            const streamUrl = window.location.protocol + '//' + window.location.host + '/hls/annotated/stream.m3u8';
            
            if (Hls.isSupported()) {
                const hls = new Hls({ liveSyncDurationCount: 3, liveMaxLatencyDurationCount: 5 });
                hls.loadSource(streamUrl);
                hls.attachMedia(video);
                hls.on(Hls.Events.MANIFEST_PARSED, () => video.play());
            } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
                video.src = streamUrl;
            }
        </script>
    </body>
    </html>
    """
    return Response(content=html, media_type="text/html")

@app.get("/status")
def status():
    with state.lock:
        return {
            "detection_active": state.detection_active,
            "total_detections": state.total_detections,
            "fps_inference": round(state.fps_inference, 2),
            "fps_stream": round(state.fps_stream, 2),
            "aws_connected": mqtt_connected,
            "stream_connected": state.raw_frame is not None,
            "hls_raw_active": ffmpeg_processes['raw'] is not None and ffmpeg_processes['raw'].poll() is None,
            "hls_annotated_active": ffmpeg_processes['annotated'] is not None and ffmpeg_processes['annotated'].poll() is None
        }

@app.get("/metrics")
def get_metrics():
    with state.lock:
        uptime = time.time() - state.metrics['uptime_start']
        metrics_copy = state.metrics.copy()
    
    uptime_hours = uptime / 3600 if uptime > 0 else 0.001
    
    return {
        "system_health": {
            "uptime_hours": round(uptime_hours, 2),
            "fps_inference": round(state.fps_inference, 2),
            "fps_stream": round(state.fps_stream, 2),
            "frames_processed": metrics_copy['frames_processed'],
            "aws_connected": mqtt_connected,
            "hls_status": {
                "raw_active": ffmpeg_processes['raw'] is not None,
                "annotated_active": ffmpeg_processes['annotated'] is not None
            }
        },
        "detection_summary": {
            "total_detections": state.total_detections,
            "intruder_detections": metrics_copy['intruder_detections'],
            "chicken_detections": metrics_copy['chicken_detections'],
            "healthy_chickens": metrics_copy['healthy_chickens'],
            "sick_chickens": metrics_copy['sick_chickens'],
            "dead_chickens": metrics_copy['dead_chickens']
        },
        "alerts": {
            "sent": metrics_copy['alerts_sent'],
            "blocked": metrics_copy['alerts_blocked_cooldown']
        }
    }

# ============================================
# DYNAMIC CAMERA URL MANAGEMENT
# ============================================

@app.get("/camera-url")
def get_camera_url():
    """Get the current camera stream URL"""
    global ESP32_CAM_URL
    return {
        "current_url": ESP32_CAM_URL,
        "status": "connected" if state.raw_frame is not None else "disconnected"
    }

@app.post("/update-camera-url")
async def update_camera_url(new_url: str = None):
    """
    Update camera URL dynamically without redeployment.
    
    Usage: POST /update-camera-url?new_url=rtsp://user:pass@host:port/stream
    
    Example for ngrok:
    POST /update-camera-url?new_url=rtsp://poultrix:poultrix123@0.tcp.ap.ngrok.io:NEW_PORT/stream2
    """
    global ESP32_CAM_URL
    
    if not new_url:
        return JSONResponse(
            status_code=400,
            content={"error": "new_url parameter is required"}
        )
    
    old_url = ESP32_CAM_URL
    ESP32_CAM_URL = new_url
    
    print(f"📹 Camera URL updated:")
    print(f"   Old: {old_url}")
    print(f"   New: {new_url}")
    
    return {
        "success": True,
        "old_url": old_url,
        "new_url": new_url,
        "message": "Camera URL updated. Stream capture thread will reconnect automatically."
    }

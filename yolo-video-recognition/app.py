from flask import Flask, render_template, request, Response, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import queue
import time

app = Flask(__name__)

# 加载YOLO模型 (使用预训练的yolov8n模型)
model = YOLO('yolov8n.pt')

# 全局变量用于摄像头线程
camera_thread = None
camera_running = False
camera_lock = threading.Lock()
frame_queue = queue.Queue(maxsize=2)

def process_frame_for_web(frame):
    """处理帧并进行YOLO识别"""
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()
    return annotated_frame

def camera_capture_thread():
    """后台摄像头捕获线程"""
    global camera_running, camera_lock
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return
    
    while True:
        with camera_lock:
            if not camera_running:
                break
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # 处理帧
        processed_frame = process_frame_for_web(frame)
        
        # 转换为JPEG
        ret, jpeg = cv2.imencode('.jpg', processed_frame)
        if ret:
            try:
                frame_queue.put(jpeg.tobytes(), block=False)
            except queue.Full:
                pass
    
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """视频流路由"""
    def generate():
        while True:
            try:
                frame_bytes = frame_queue.get(timeout=1)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except queue.Empty:
                continue
            except GeneratorExit:
                break
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera():
    """启动摄像头识别"""
    global camera_thread, camera_running, camera_lock
    
    with camera_lock:
        if camera_running:
            return jsonify({'status': 'already_running'})
        
        camera_running = True
        camera_thread = threading.Thread(target=camera_capture_thread)
        camera_thread.daemon = True
        camera_thread.start()
    
    return jsonify({'status': 'started'})

@app.route('/stop_camera')
def stop_camera():
    """停止摄像头识别"""
    global camera_running, camera_lock
    
    with camera_lock:
        camera_running = False
    
    if camera_thread:
        camera_thread.join(timeout=2)
    
    # 清空队列
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            break
    
    return jsonify({'status': 'stopped'})

@app.route('/process_video', methods=['POST'])
def process_video():
    """处理视频链接并返回识别结果"""
    video_url = request.json.get('video_url')
    
    if not video_url:
        return jsonify({'error': 'No video URL provided'}), 400
    
    try:
        # 打开视频流
        cap = cv2.VideoCapture(video_url)
        
        if not cap.isOpened():
            return jsonify({'error': 'Cannot open video stream'}), 400
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 处理前30帧作为预览
        processed_frames = []
        frame_count = 0
        max_frames = 30
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 每隔几帧处理一次
            if frame_count % 3 == 0:
                processed_frame = process_frame_for_web(frame)
                ret, jpeg = cv2.imencode('.jpg', processed_frame)
                if ret:
                    processed_frames.append(jpeg.tobytes())
            
            frame_count += 1
        
        cap.release()
        
        return jsonify({
            'status': 'success',
            'fps': fps,
            'width': width,
            'height': height,
            'total_frames': total_frames,
            'processed_frames_count': len(processed_frames),
            'frames': [frame.hex() for frame in processed_frames] if processed_frames else []
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """处理上传的视频文件"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    
    if video_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # 保存上传的文件
        import os
        upload_dir = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        video_path = os.path.join(upload_dir, video_file.filename)
        video_file.save(video_path)
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            os.remove(video_path)
            return jsonify({'error': 'Cannot open video file'}), 400
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 处理前30帧
        processed_frames = []
        frame_count = 0
        max_frames = 30
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % 3 == 0:
                processed_frame = process_frame_for_web(frame)
                ret, jpeg = cv2.imencode('.jpg', processed_frame)
                if ret:
                    processed_frames.append(jpeg.tobytes())
            
            frame_count += 1
        
        cap.release()
        
        # 删除临时文件
        os.remove(video_path)
        
        return jsonify({
            'status': 'success',
            'fps': fps,
            'width': width,
            'height': height,
            'total_frames': total_frames,
            'processed_frames_count': len(processed_frames),
            'frames': [frame.hex() for frame in processed_frames] if processed_frames else []
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

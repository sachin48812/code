#!/usr/bin/env python3
"""
Simplified Backend API Server for Smart Traffic Management System
This server provides all the API endpoints needed by the frontend
"""

from flask import Flask, request, jsonify, Response, send_from_directory, send_file
from flask_cors import CORS
import time
import json
import random
import threading
import os
import cv2
import numpy as np
import requests
from datetime import datetime
from urllib.parse import urlparse
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# Frontend directory path
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend')

# Global state
system_running = True  # Set to True by default to ensure the web interface displays 'Running' on startup
traffic_data = {
    'north': {'car': 0, 'bus': 0, 'motorcycle': 0, 'truck': 0, 'emergency': 0, 'total': 0},
    'east': {'car': 0, 'bus': 0, 'motorcycle': 0, 'truck': 0, 'emergency': 0, 'total': 0},
    'south': {'car': 0, 'bus': 0, 'motorcycle': 0, 'truck': 0, 'emergency': 0, 'total': 0},
    'west': {'car': 0, 'bus': 0, 'motorcycle': 0, 'truck': 0, 'emergency': 0, 'total': 0}
}

signal_states = {
    'main_intersection': {
        'current_green_direction': 'NORTH',
        'remaining_time': 45,
        'signal_states': {
            'NORTH': 'GREEN',
            'EAST': 'RED', 
            'SOUTH': 'RED',
            'WEST': 'RED'
        },
        'is_manual_override': False,
        'phase_duration': 60,
        'timestamp': time.time(),
        'last_change_time': time.time(),
        'yellow_start_time': None,
        'yellow_duration': 4
    }
}

# Queue detection for adaptive timing
queue_data = {
    'north': {'stopped_vehicles': 0, 'moving_vehicles': 0, 'total_wait_time': 0},
    'east': {'stopped_vehicles': 0, 'moving_vehicles': 0, 'total_wait_time': 0},
    'south': {'stopped_vehicles': 0, 'moving_vehicles': 0, 'total_wait_time': 0},
    'west': {'stopped_vehicles': 0, 'moving_vehicles': 0, 'total_wait_time': 0}
}

# Signal timing parameters (following specification: 30-90 seconds)
SIGNAL_TIMING = {
    'MIN_GREEN_TIME': 30,  # Minimum green light duration (per user specification)
    'MAX_GREEN_TIME': 90,  # Maximum green light duration  
    'TIME_PER_VEHICLE': 2,  # Seconds needed per vehicle to clear (as per user formula)
    'BASE_TIME': 30,       # Base green time even if no vehicles (minimum)
    'EMERGENCY_TIME': 20,  # Time for emergency vehicles
    'BUS_BONUS_TIME': 5,   # Extra time per bus
    'YELLOW_TIME': 4,      # Yellow phase duration
    'MIN_PHASE_DURATION': 30,  # Minimum time before signal can change (matches user requirement)
    'SIGNAL_CHECK_INTERVAL': 5  # Check signal changes every 5 seconds for stability
}

# Sequential direction order (North → East → South → West)
DIRECTION_SEQUENCE = ['NORTH', 'EAST', 'SOUTH', 'WEST']

# Signal control timing
last_signal_check_time = 0

config = {
    'camera_urls': {
        'north': '',
        'east': '',
        'south': '',
        'west': ''
    },
    'detection_threshold': 0.5,
    'emergency_threshold': 0.8,
    'yellow_time': 4.0,
    'camera_fps': 30,
    'device': 'cuda',  # Use GPU by default
    'model_size': 'yolov8n',
    'fast_inference': True,
    'max_resolution': 640,  # Good quality
    'confidence': 0.25,  # Better detection sensitivity
    'jpeg_quality': 80  # Good video quality
}

# Active camera streams
active_cameras = {}
camera_detection_enabled = True

# YOLO model for vehicle detection
yolo_model = None
model_lock = threading.Lock()

# Vehicle class IDs from COCO dataset
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle', 
    5: 'bus',
    7: 'truck'
}

# Emergency vehicle detection (simplified - in real implementation would use specialized model)
EMERGENCY_KEYWORDS = ['ambulance', 'police', 'fire', 'emergency']

def load_yolo_model():
    """Load YOLO model for vehicle detection with GPU support"""
    global yolo_model
    try:
        if yolo_model is None:
            print("Loading YOLO model...")
            
            # Check GPU availability
            try:
                import torch
                if torch.cuda.is_available():
                    device = 'cuda'
                    gpu_name = torch.cuda.get_device_name(0)
                    print(f"🚀 GPU Available: {gpu_name}")
                    print("🔥 Loading YOLO on GPU for maximum performance")
                else:
                    device = 'cpu'
                    print("⚠️ GPU not available, using CPU")
            except ImportError:
                device = 'cpu'
                print("⚠️ PyTorch not available, using CPU")
            
            # Load model with proper error handling for PyTorch weights_only issue
            model_size = config.get('model_size', 'yolov8n')
            
            # Try different approaches to load the model
            yolo_model = None
            load_errors = []
            
            # Approach 1: Standard loading (might fail with weights_only issue)
            try:
                yolo_model = YOLO(f'{model_size}.pt')
                print("✅ Model loaded with standard method")
            except Exception as e:
                load_errors.append(f"Standard loading failed: {str(e)[:100]}")
                print(f"⚠️ {load_errors[-1]}")
            
            # Approach 2: If standard fails, try with torch.load workaround
            if yolo_model is None:
                try:
                    import torch
                    # For PyTorch 2.6+, we need to add safe globals
                    if hasattr(torch.serialization, 'add_safe_globals'):
                        try:
                            from ultralytics.nn.tasks import DetectionModel
                            torch.serialization.add_safe_globals([DetectionModel])
                            yolo_model = YOLO(f'{model_size}.pt')
                            print("✅ Model loaded with add_safe_globals workaround")
                        except Exception as e:
                            load_errors.append(f"add_safe_globals failed: {str(e)[:100]}")
                            print(f"⚠️ {load_errors[-1]}")
                    
                    # If that doesn't work, the standard method should work
                    if yolo_model is None:
                        try:
                            # Try to load with specific parameters
                            yolo_model = YOLO(f'{model_size}.pt')
                            print("✅ Model loaded with direct method")
                        except Exception as e:
                            load_errors.append(f"Direct method failed: {str(e)[:100]}")
                            print(f"⚠️ {load_errors[-1]}")
                except Exception as e:
                    load_errors.append(f"Torch workaround failed: {str(e)[:100]}")
                    print(f"⚠️ {load_errors[-1]}")
            
            # If all approaches fail, log the errors and return None
            if yolo_model is None:
                print("❌ Failed to load YOLO model with all approaches:")
                for i, error in enumerate(load_errors, 1):
                    print(f"   {i}. {error}")
                print("💡 Try downloading the model again or check PyTorch/ultralytics compatibility")
                return None
            
            # Force GPU usage if available
            if device == 'cuda':
                yolo_model.to('cuda')
                config['device'] = 'cuda'
                print(f"✅ YOLO model loaded on GPU: {model_size}")
                
                # Test GPU performance
                import numpy as np
                test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                start_time = time.time()
                _ = yolo_model(test_frame, conf=0.5, verbose=False, device='cuda')
                inference_time = (time.time() - start_time) * 1000
                print(f"🚀 GPU Inference speed: {inference_time:.1f}ms")
            else:
                yolo_model.to('cpu')
                config['device'] = 'cpu'
                print(f"✅ YOLO model loaded on CPU: {model_size}")
            
        return yolo_model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        import traceback
        traceback.print_exc()
        return None

def detect_vehicles_in_frame(frame):
    """Detect vehicles in frame using YOLO with GPU acceleration and return counts and annotated frame"""
    print("🔍 detect_vehicles_in_frame called")
    
    if frame is None:
        print("⚠️ Frame is None, returning empty results")
        return None, {}
    
    try:
        print("🔄 Attempting to load YOLO model...")
        model = load_yolo_model()
        print(f"📊 Model load result: {model is not None}")
        
        if model is None:
            print("❌ Model failed to load, returning frame without detection")
            return frame, {}
        
        print("✅ Model loaded successfully, proceeding with detection")
        
        # Optimize frame size for better performance while maintaining quality
        original_height, original_width = frame.shape[:2]
        max_width = config.get('max_resolution', 640)
        
        if original_width > max_width:
            scale = max_width / original_width
            new_width = max_width
            new_height = int(original_height * scale)
            inference_frame = cv2.resize(frame, (new_width, new_height))
        else:
            inference_frame = frame
        
        print(f"📐 Frame resized from {original_width}x{original_height} to {inference_frame.shape[1]}x{inference_frame.shape[0]}")
        
        # Performance monitoring
        start_time = time.time()
        
        # Run inference with GPU if available
        device = config.get('device', 'cpu')
        conf_threshold = config.get('confidence', 0.25)
        
        print(f"🚀 Running inference on device: {device}, confidence threshold: {conf_threshold}")
        
        if device == 'cuda':
            try:
                import torch
                if torch.cuda.is_available():
                    print("🔥 Using CUDA for inference")
                    results = model(inference_frame, conf=conf_threshold, verbose=False, device='cuda')
                else:
                    print("⚠️ CUDA not available, using CPU")
                    results = model(inference_frame, conf=conf_threshold, verbose=False)
            except ImportError:
                print("⚠️ PyTorch import error, using CPU")
                results = model(inference_frame, conf=conf_threshold, verbose=False)
        else:
            print("💻 Using CPU for inference")
            results = model(inference_frame, conf=conf_threshold, verbose=False)
        
        inference_time = (time.time() - start_time) * 1000
        print(f"⏱️ Inference completed in {inference_time:.1f}ms")
        
        # Initialize counts
        vehicle_counts = {
            'car': 0,
            'bus': 0, 
            'motorcycle': 0,
            'truck': 0,
            'emergency': 0
        }
        
        # Scale coordinates back if frame was resized
        scale_x = original_width / inference_frame.shape[1]
        scale_y = original_height / inference_frame.shape[0]
        
        # Process detections
        detection_count = 0
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Only process vehicle classes
                    if cls in VEHICLE_CLASSES and conf > conf_threshold:
                        vehicle_type = VEHICLE_CLASSES[cls]
                        vehicle_counts[vehicle_type] += 1
                        detection_count += 1
                        
                        # Scale coordinates back to original frame
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
                        
                        # Color coding for different vehicles
                        colors = {
                            'car': (0, 255, 0),      # Green
                            'bus': (255, 165, 0),    # Orange  
                            'motorcycle': (255, 255, 0), # Yellow
                            'truck': (0, 0, 255),    # Red
                            'emergency': (255, 0, 255) # Magenta
                        }
                        
                        color = colors.get(vehicle_type, (0, 255, 0))
                        
                        # Draw rectangle and label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{vehicle_type} {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Calculate total
        vehicle_counts['total'] = sum(vehicle_counts[k] for k in ['car', 'bus', 'motorcycle', 'truck', 'emergency'])
        
        print(f"🚗 Detected {detection_count} vehicles: {vehicle_counts}")
        
        # Add performance and device info overlay
        timestamp = datetime.now().strftime("%H:%M:%S")
        device_status = device.upper()
        
        # Color code by performance
        if inference_time < 50:
            perf_color = (0, 255, 0)  # Green
            perf_status = "FAST"
        elif inference_time < 100:
            perf_color = (0, 255, 255)  # Yellow
            perf_status = "OK"
        else:
            perf_color = (0, 0, 255)  # Red
            perf_status = "SLOW"
        
        cv2.putText(frame, f"Live Detection - {timestamp} | {device_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Vehicles: {vehicle_counts['total']} | {inference_time:.0f}ms ({perf_status})", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, perf_color, 2)
        
        # Log performance occasionally
        if inference_time > 100:
            print(f"⚠️ Slow detection: {inference_time:.1f}ms")
        elif inference_time < 50:
            print(f"✅ Fast detection: {inference_time:.1f}ms")
        
        return frame, vehicle_counts
        
    except Exception as e:
        print(f"❌ Error in vehicle detection: {e}")
        import traceback
        traceback.print_exc()
        return frame, {}

events_log = []
performance_metrics = {
    'average_wait_time': 0,
    'throughput': 0,
    'congestion_level': 0,
    'efficiency_score': 85,
    'timestamp': time.time()
}

def update_traffic_data_from_detection(direction, vehicle_counts):
    """Update traffic data with real detection results and queue management"""
    global traffic_data, signal_states
    
    if direction in traffic_data:
        traffic_data[direction].update(vehicle_counts)
        
        # Determine if this direction has green light
        current_green = signal_states['main_intersection']['current_green_direction'].lower()
        is_green_light = (direction == current_green)
        
        # Update queue data for adaptive timing
        update_queue_data(direction, vehicle_counts, is_green_light)
        
        # Log significant changes
        total = vehicle_counts.get('total', 0)
        if total > 5:  # High traffic
            events_log.append({
                'type': 'high_traffic_detected',
                'timestamp': time.time(),
                'data': {'direction': direction, 'count': total, 'vehicles': vehicle_counts, 'queue_data': queue_data[direction]}
            })
        
def calculate_optimal_green_time(direction, vehicle_counts, queue_info):
    """Calculate optimal green light duration using user formula: Green Light Duration = Start-up Lost Time + (∑Time per Vehicle type)"""
    total_vehicles = vehicle_counts.get('total', 0)
    stopped_vehicles = queue_info.get('stopped_vehicles', 0)
    emergency_vehicles = vehicle_counts.get('emergency', 0)
    buses = vehicle_counts.get('bus', 0)
    cars = vehicle_counts.get('car', 0)
    motorcycles = vehicle_counts.get('motorcycle', 0)
    trucks = vehicle_counts.get('truck', 0)
    
    # Only log when there are significant changes
    if total_vehicles > 0 or emergency_vehicles > 0:
        print(f"\n=== Green Light Duration Calculation for {direction.upper()} ===")
        print(f"Total vehicles detected: {total_vehicles}")
        print(f"Stopped vehicles in queue: {stopped_vehicles}")
        print(f"Vehicle breakdown - Cars: {cars}, Buses: {buses}, Motorcycles: {motorcycles}, Trucks: {trucks}, Emergency: {emergency_vehicles}")
    
    # START-UP LOST TIME (constant for all directions)
    START_UP_LOST_TIME = 4  # 4 seconds for signal transition and initial acceleration
    
    # TIME PER VEHICLE TYPE (in seconds) - different vehicles need different clearance times
    VEHICLE_TYPE_TIMES = {
        'car': 2.5,        # Standard cars need 2.5 seconds
        'motorcycle': 2.0,  # Motorcycles are faster
        'bus': 4.0,        # Buses need more time due to size
        'truck': 4.5,      # Trucks need the most time
        'emergency': 3.0    # Emergency vehicles
    }
    
    # Use actual vehicle counts by type for calculation
    cars_waiting = cars if stopped_vehicles > 0 else 0
    motorcycles_waiting = motorcycles if stopped_vehicles > 0 else 0
    buses_waiting = buses if stopped_vehicles > 0 else 0
    trucks_waiting = trucks if stopped_vehicles > 0 else 0
    emergency_waiting = emergency_vehicles if stopped_vehicles > 0 else 0
    
    # Calculate ∑Time per Vehicle type
    sum_vehicle_times = (
        cars_waiting * VEHICLE_TYPE_TIMES['car'] +
        motorcycles_waiting * VEHICLE_TYPE_TIMES['motorcycle'] +
        buses_waiting * VEHICLE_TYPE_TIMES['bus'] +
        trucks_waiting * VEHICLE_TYPE_TIMES['truck'] +
        emergency_waiting * VEHICLE_TYPE_TIMES['emergency']
    )
    
    # Apply the user's formula: Green Light Duration = Start-up Lost Time + (∑Time per Vehicle type)
    calculated_time = START_UP_LOST_TIME + sum_vehicle_times
    
    # Emergency vehicles get absolute priority - extend time if needed
    if emergency_vehicles > 0:
        emergency_bonus = emergency_vehicles * SIGNAL_TIMING['EMERGENCY_TIME']
        calculated_time += emergency_bonus
        print(f"  Emergency priority bonus: +{emergency_bonus}s")
    
    # Bus priority (3+ buses get extra time as per specification)
    if buses >= 3:
        bus_bonus = buses * SIGNAL_TIMING['BUS_BONUS_TIME']
        calculated_time += bus_bonus
        print(f"  Bus priority bonus (3+ buses): +{bus_bonus}s")
    
    # Apply min/max limits: max(30, min(90, calculated_time))
    optimal_time = max(SIGNAL_TIMING['MIN_GREEN_TIME'], 
                      min(SIGNAL_TIMING['MAX_GREEN_TIME'], calculated_time))
    
    # Only log when there are significant changes  
    if total_vehicles > 0 or emergency_vehicles > 0:
        print(f"Formula: Green Light Duration = Start-up Lost Time + (∑Time per Vehicle type)")
        print(f"  Start-up Lost Time: {START_UP_LOST_TIME}s")
        print(f"  ∑Time per Vehicle type: {sum_vehicle_times}s")
        print(f"    Cars: {cars_waiting} × {VEHICLE_TYPE_TIMES['car']}s = {cars_waiting * VEHICLE_TYPE_TIMES['car']}s")
        print(f"    Motorcycles: {motorcycles_waiting} × {VEHICLE_TYPE_TIMES['motorcycle']}s = {motorcycles_waiting * VEHICLE_TYPE_TIMES['motorcycle']}s")
        print(f"    Buses: {buses_waiting} × {VEHICLE_TYPE_TIMES['bus']}s = {buses_waiting * VEHICLE_TYPE_TIMES['bus']}s")
        print(f"    Trucks: {trucks_waiting} × {VEHICLE_TYPE_TIMES['truck']}s = {trucks_waiting * VEHICLE_TYPE_TIMES['truck']}s")
        print(f"    Emergency: {emergency_waiting} × {VEHICLE_TYPE_TIMES['emergency']}s = {emergency_waiting * VEHICLE_TYPE_TIMES['emergency']}s")
        print(f"  Total: {START_UP_LOST_TIME} + {sum_vehicle_times} = {calculated_time}s")
        print(f"  Final (with 30-90s limits): {optimal_time}s")
        print(f"===================\n")
    
    return optimal_time

def update_queue_data(direction, vehicle_counts, is_green_light):
    """Update queue data based on current signal state"""
    global queue_data
    
    total_vehicles = vehicle_counts.get('total', 0)
    
    # Initialize queue data for direction if not exists
    if direction not in queue_data:
        queue_data[direction] = {
            'stopped_vehicles': 0,
            'moving_vehicles': 0,
            'total_wait_time': 0
        }
    
    if is_green_light:
        # Green light - vehicles are moving, reduce queue
        queue_data[direction]['moving_vehicles'] = total_vehicles
        # Clear stopped vehicles gradually as they move
        if queue_data[direction]['stopped_vehicles'] > 0:
            # Vehicles are leaving the queue
            vehicles_moved = min(total_vehicles, queue_data[direction]['stopped_vehicles'])
            queue_data[direction]['stopped_vehicles'] = max(0, queue_data[direction]['stopped_vehicles'] - vehicles_moved)
            print(f"{direction.title()}: {vehicles_moved} vehicles moved, {queue_data[direction]['stopped_vehicles']} still waiting")
    else:
        # Red light - vehicles are stopped, add to queue
        queue_data[direction]['stopped_vehicles'] = total_vehicles
        queue_data[direction]['moving_vehicles'] = 0
        
        # Increase wait time for stopped vehicles
        if total_vehicles > 0:
            queue_data[direction]['total_wait_time'] += 3  # 3 seconds per update cycle
            if total_vehicles > 5:  # Log significant queues
                print(f"{direction.title()} RED: {total_vehicles} vehicles waiting (total wait: {queue_data[direction]['total_wait_time']}s)")
    
    return queue_data[direction]

def simulate_traffic_data():
    """Only simulate traffic data when explicitly in simulation mode - no random data without cameras"""
    global traffic_data, signal_states
    
    # Check if any real cameras are configured
    has_configured_cameras = any(url.strip() for url in config['camera_urls'].values())
    
    # Only simulate if no cameras are configured AND camera detection is disabled
    if not has_configured_cameras and not camera_detection_enabled:
        # Show minimal static data to indicate simulation mode
        for direction in ['north', 'east', 'south', 'west']:
            traffic_data[direction] = {
                'car': 0,
                'bus': 0,
                'motorcycle': 0,
                'truck': 0,
                'emergency': 0,
                'total': 0
            }
            # Reset queue data as well
            queue_data[direction] = {
                'stopped_vehicles': 0,
                'moving_vehicles': 0,
                'total_wait_time': 0
            }
    # If cameras are configured but detection is failing, keep zero counts
    elif has_configured_cameras:
        # Keep existing real detection data, don't override with simulation
        pass

def get_next_direction_in_sequence(current_direction):
    """Get the next direction in strict sequential order: North → East → South → West → repeat"""
    try:
        current_index = DIRECTION_SEQUENCE.index(current_direction.upper())
        next_index = (current_index + 1) % len(DIRECTION_SEQUENCE)
        return DIRECTION_SEQUENCE[next_index]
    except (ValueError, AttributeError):
        # Fallback to North if current direction is invalid
        return 'NORTH'

def simulate_signal_changes():
    """Sequential signal control with strict North→East→South→West flow and proper timing"""
    global signal_states, events_log, last_signal_check_time
    
    current_time = time.time()
    intersection = signal_states['main_intersection']
    current_green = intersection['current_green_direction']
    time_since_change = current_time - intersection['last_change_time']
    
    # Check if we're in yellow phase
    if intersection.get('yellow_start_time'):
        yellow_elapsed = current_time - intersection['yellow_start_time']
        if yellow_elapsed >= intersection['yellow_duration']:
            # Yellow phase complete, switch to NEXT direction in sequence
            next_direction = get_next_direction_in_sequence(current_green)
            complete_signal_change(intersection, next_direction)
            return
        else:
            # Still in yellow phase
            intersection['remaining_time'] = intersection['yellow_duration'] - yellow_elapsed
            return
    
    # Only check for signal changes every 5 seconds to prevent rapid changes
    if current_time - last_signal_check_time < SIGNAL_TIMING['SIGNAL_CHECK_INTERVAL']:
        # Just update remaining time without decision logic
        current_green_lower = current_green.lower()
        vehicle_data = traffic_data.get(current_green_lower, {})
        
        # Ensure queue_data entry exists
        if current_green_lower not in queue_data:
            queue_data[current_green_lower] = {
                'stopped_vehicles': 0,
                'moving_vehicles': 0,
                'total_wait_time': 0
            }
        
        queue_info = queue_data.get(current_green_lower, {})
        optimal_time = calculate_optimal_green_time(current_green_lower, vehicle_data, queue_info)
        elapsed_time = time_since_change
        intersection['remaining_time'] = max(0, optimal_time - elapsed_time)
        return
    
    last_signal_check_time = current_time
    
    # Calculate current green time requirements
    current_green_lower = current_green.lower()
    vehicle_data = traffic_data.get(current_green_lower, {})
    
    # Ensure queue_data entry exists
    if current_green_lower not in queue_data:
        queue_data[current_green_lower] = {
            'stopped_vehicles': 0,
            'moving_vehicles': 0,
            'total_wait_time': 0
        }
    
    queue_info = queue_data.get(current_green_lower, {})
    optimal_time = calculate_optimal_green_time(current_green_lower, vehicle_data, queue_info)
    elapsed_time = time_since_change
    
    # Decision logic for sequential signal change
    should_change = False
    reason = ""
    next_direction = get_next_direction_in_sequence(current_green)
    
    # Check for emergency vehicles (absolute priority)
    emergency_direction = None
    for direction in ['north', 'east', 'south', 'west']:
        # Ensure traffic_data entry exists
        if direction not in traffic_data:
            traffic_data[direction] = {
                'car': 0,
                'bus': 0,
                'motorcycle': 0,
                'truck': 0,
                'emergency': 0,
                'total': 0
            }
            
        emergency_count = traffic_data.get(direction, {}).get('emergency', 0)
        if emergency_count > 0:
            emergency_direction = direction.upper()
            break
    
    # Check if we should start yellow phase (3 seconds before change)
    time_remaining = optimal_time - elapsed_time
    min_time_for_yellow = max(SIGNAL_TIMING['MIN_PHASE_DURATION'] - 3, 1)
    
    # Only trigger yellow phase if we're past the minimum time and have 3 or fewer seconds remaining
    should_start_yellow = (
        elapsed_time >= min_time_for_yellow and
        time_remaining <= 3 and 
        not intersection.get('yellow_start_time')
    )
    
    print(f"DEBUG: {current_green} - Elapsed: {elapsed_time:.1f}s, Optimal: {optimal_time:.1f}s, Remaining: {time_remaining:.1f}s")
    print(f"DEBUG: Min time for yellow: {min_time_for_yellow}s, Should start yellow: {should_start_yellow}")
    
    if emergency_direction and emergency_direction != current_green:
        should_change = True
        next_direction = emergency_direction
        reason = f"emergency_override_{emergency_direction}"
        print(f"🚨 EMERGENCY: Switching to {emergency_direction} for emergency vehicle")
    elif should_start_yellow:
        should_change = True
        reason = "yellow_phase_trigger"
        print(f"⚠️ YELLOW PHASE TRIGGER: {current_green} ({elapsed_time:.0f}s) → {next_direction} in {time_remaining:.1f} seconds")
    # Normal sequential flow - only change after minimum time has passed
    elif elapsed_time >= optimal_time and elapsed_time >= SIGNAL_TIMING['MIN_PHASE_DURATION']:
        should_change = True
        reason = "sequential_timing_complete"
        print(f"⏰ Sequential timing complete: {current_green} ({elapsed_time:.0f}s) → {next_direction}")
    # Safety check - never exceed maximum time
    elif elapsed_time >= SIGNAL_TIMING['MAX_GREEN_TIME']:
        should_change = True
        reason = "max_time_safety"
        print(f"⚠️ Max time safety: Forcing change from {current_green} after {elapsed_time:.0f}s")
    
    if should_change and reason == "yellow_phase_trigger":
        # Start yellow phase before changing to next direction
        print(f"🚦 Starting signal change: {current_green} → {next_direction} (reason: {reason})")
        start_yellow_phase(intersection, next_direction, reason)
    elif should_change and reason != "yellow_phase_trigger":
        # Complete signal change immediately (for emergency or max time cases)
        complete_signal_change(intersection, next_direction)
    else:
        # Update remaining time for current green
        intersection['remaining_time'] = max(0, optimal_time - elapsed_time)
        
        # Ensure minimum time is respected
        if intersection['remaining_time'] < 5 and elapsed_time < SIGNAL_TIMING['MIN_PHASE_DURATION']:
            intersection['remaining_time'] = SIGNAL_TIMING['MIN_PHASE_DURATION'] - elapsed_time

def start_yellow_phase(intersection, next_direction, reason):
    """Start yellow phase before signal change"""
    intersection['yellow_start_time'] = time.time()
    intersection['remaining_time'] = intersection['yellow_duration']
    intersection['next_green_direction'] = next_direction.upper()
    
    print(f"\n🟡 STARTING YELLOW PHASE: {intersection['current_green_direction']} → {next_direction.upper()}")
    
    # Update all signals to show yellow for current green
    current_green = intersection['current_green_direction']
    for direction in ['NORTH', 'EAST', 'SOUTH', 'WEST']:
        if direction == current_green:
            intersection['signal_states'][direction] = 'YELLOW'
        else:
            intersection['signal_states'][direction] = 'RED'
    
    print(f"   Yellow phase signal states: {intersection['signal_states']}")  # Debug log
    
    events_log.append({
        'type': 'yellow_phase_started',
        'timestamp': time.time(),
        'data': {
            'current': current_green,
            'next': next_direction.upper(),
            'reason': reason
        }
    })
    
    print(f"Yellow phase started: {current_green} -> {next_direction.upper()} (reason: {reason})")

def complete_signal_change(intersection, next_direction=None):
    """Complete the signal change after yellow phase with proper timing"""
    if next_direction is None:
        next_direction = intersection.get('next_green_direction', 'NORTH')
    else:
        next_direction = next_direction.upper()
    
    intersection['current_green_direction'] = next_direction
    intersection['last_change_time'] = time.time()
    intersection['yellow_start_time'] = None
    
    # Calculate optimal green time for new direction
    direction_lower = next_direction.lower()
    vehicle_data = traffic_data.get(direction_lower, {})
    queue_info = queue_data.get(direction_lower, {})
    optimal_time = calculate_optimal_green_time(direction_lower, vehicle_data, queue_info)
    
    # Ensure minimum timing is respected
    final_duration = max(optimal_time, SIGNAL_TIMING['MIN_PHASE_DURATION'])
    
    intersection['remaining_time'] = final_duration
    intersection['phase_duration'] = final_duration
    
    # Update signal states
    for direction in ['NORTH', 'EAST', 'SOUTH', 'WEST']:
        new_state = 'GREEN' if direction == next_direction else 'RED'
        intersection['signal_states'][direction] = new_state
        print(f"   Signal {direction}: {new_state}")  # Debug log
    
    # Reset queue wait time for the new green direction
    if direction_lower in queue_data:
        queue_data[direction_lower]['total_wait_time'] = 0
    
    events_log.append({
        'type': 'sequential_signal_change',
        'timestamp': time.time(),
        'data': {
            'direction': next_direction,
            'duration': final_duration,
            'vehicle_count': vehicle_data.get('total', 0),
            'stopped_vehicles': queue_info.get('stopped_vehicles', 0),
            'sequence_order': DIRECTION_SEQUENCE.index(next_direction) + 1
        }
    })
    
    print(f"\n🚦 SIGNAL CHANGED: {next_direction} for {final_duration} seconds (Sequential: {DIRECTION_SEQUENCE.index(next_direction) + 1}/4)")
    print(f"   Vehicle count: {vehicle_data.get('total', 0)}, Queue: {queue_info.get('stopped_vehicles', 0)}")
    print(f"   Next change in: {final_duration} seconds")
    print(f"   Signal states: {intersection['signal_states']}")  # Debug log
    print("")
    
    intersection['timestamp'] = time.time()

def simulation_loop():
    """Main simulation loop - handles real data and proper signal timing"""
    while True:
        if system_running:
            # Only simulate traffic if no real camera data is being provided
            simulate_traffic_data()
            
            # Always update signal logic based on current traffic data (real or simulated)
            simulate_signal_changes()
        time.sleep(3)  # Slower updates for more stable timing (3 seconds)

# Start simulation thread
simulation_thread = threading.Thread(target=simulation_loop, daemon=True)
simulation_thread.start()

# API Routes
@app.route('/api/status')
def get_status():
    active_camera_count = len([url for url in config['camera_urls'].values() if url.strip()])
    return jsonify({
        'status': 'running' if system_running else 'stopped',
        'mode': 'live' if camera_detection_enabled else 'simulation',
        'intersection_id': 'main_intersection',
        'cameras': config.get('camera_urls', {}),
        'active_cameras': active_camera_count,
        'camera_detection_enabled': camera_detection_enabled,
        'signal_states': signal_states,
        'timestamp': time.time(),
        'system_healthy': True  # Add health indicator
    })

@app.route('/api/start', methods=['POST'])
def start_system():
    global system_running
    system_running = True
    events_log.append({
        'type': 'system_started',
        'timestamp': time.time(),
        'data': {'message': 'Traffic management system started'}
    })
    return jsonify({'message': 'System started', 'status': 'running'})

@app.route('/api/stop', methods=['POST'])
def stop_system():
    global system_running
    system_running = False
    events_log.append({
        'type': 'system_stopped',
        'timestamp': time.time(),
        'data': {'message': 'Traffic management system stopped'}
    })
    return jsonify({'message': 'System stopped', 'status': 'stopped'})

@app.route('/api/vehicle-counts')
def get_vehicle_counts():
    return jsonify({
        **traffic_data,
        'queue_data': queue_data,
        'signal_timing': {
            'current_green': signal_states['main_intersection']['current_green_direction'],
            'remaining_time': signal_states['main_intersection']['remaining_time'],
            'is_yellow': signal_states['main_intersection'].get('yellow_start_time') is not None
        },
        'timestamp': time.time()
    })

@app.route('/api/signals')
def get_signal_states():
    return jsonify(signal_states)

@app.route('/api/signals/<intersection_id>/manual-override', methods=['POST'])
def manual_override(intersection_id):
    data = request.json or {}
    direction = data.get('direction', '').upper()
    
    if direction in ['NORTH', 'EAST', 'SOUTH', 'WEST']:
        signal_states[intersection_id]['current_green_direction'] = direction
        signal_states[intersection_id]['is_manual_override'] = True
        
        # Calculate optimal green time for the selected direction
        direction_lower = direction.lower()
        vehicle_data = traffic_data.get(direction_lower, {})
        queue_info = queue_data.get(direction_lower, {})
        optimal_time = calculate_optimal_green_time(direction_lower, vehicle_data, queue_info)
        
        # Ensure minimum timing is respected
        final_duration = max(optimal_time, SIGNAL_TIMING['MIN_PHASE_DURATION'])
        
        signal_states[intersection_id]['remaining_time'] = final_duration  # Use calculated time instead of fixed 30
        signal_states[intersection_id]['phase_duration'] = final_duration
        
        # Update signal states
        for dir in ['NORTH', 'EAST', 'SOUTH', 'WEST']:
            signal_states[intersection_id]['signal_states'][dir] = 'GREEN' if dir == direction else 'RED'
        
        events_log.append({
            'type': 'manual_override',
            'timestamp': time.time(),
            'data': {'direction': direction, 'intersection': intersection_id}
        })
        
        return jsonify({
            'message': f'Manual override set for {direction}',
            'intersection': intersection_id,
            'direction': direction,
            'remaining_time': final_duration
        })
    
    return jsonify({'error': 'Invalid direction'}), 400

@app.route('/api/signals/<intersection_id>/resume-auto', methods=['POST'])
def resume_auto(intersection_id):
    signal_states[intersection_id]['is_manual_override'] = False
    events_log.append({
        'type': 'auto_resumed',
        'timestamp': time.time(),
        'data': {'intersection': intersection_id}
    })
    return jsonify({'message': 'Resumed automatic control', 'intersection': intersection_id})

@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    global config, camera_detection_enabled
    
    if request.method == 'POST':
        data = request.json or {}
        print(f"\n🔧 Configuration update received:")
        print(f"   Data: {data}")
        
        # Update configuration
        for key, value in data.items():
            if key in config:
                old_value = config[key]
                config[key] = value
                print(f"   Updated {key}: {old_value} → {value}")
            elif key == 'camera_url':
                # Handle single camera URL (legacy)
                config['camera_urls']['north'] = value
                print(f"   Set single camera URL (north): {value}")
            elif key == 'camera_urls' and isinstance(value, dict):
                # Update camera URLs
                for direction, url in value.items():
                    if direction in config['camera_urls']:
                        old_url = config['camera_urls'][direction]
                        config['camera_urls'][direction] = url
                        if url != old_url:
                            print(f"   Updated {direction} camera: {old_url} → {url}")
        
        # Enable camera detection if any camera URL is configured
        has_cameras = any(url.strip() for url in config['camera_urls'].values())
        old_detection_enabled = camera_detection_enabled
        camera_detection_enabled = has_cameras
        
        if camera_detection_enabled != old_detection_enabled:
            print(f"   Camera detection: {old_detection_enabled} → {camera_detection_enabled}")
        
        # Log current configuration
        print(f"\n📹 Current camera configuration:")
        for direction, url in config['camera_urls'].items():
            status = "✅ CONFIGURED" if url.strip() else "❌ Not set"
            print(f"   {direction.upper()}: {status} {url[:50] + '...' if len(url) > 50 else url}")
        print(f"   Detection enabled: {camera_detection_enabled}")
        print(f"   Active cameras: {sum(1 for url in config['camera_urls'].values() if url.strip())}")
        
        events_log.append({
            'type': 'config_updated',
            'timestamp': time.time(),
            'data': {
                'updated_keys': list(data.keys()),
                'camera_detection_enabled': camera_detection_enabled,
                'active_cameras': [k for k, v in config['camera_urls'].items() if v.strip()]
            }
        })
        
        return jsonify({
            'message': 'Configuration updated successfully',
            'config': config,
            'camera_detection_enabled': camera_detection_enabled
        })
    
    return jsonify(config)

@app.route('/api/events')
def get_events():
    limit = request.args.get('limit', 50, type=int)
    return jsonify({
        'events': events_log[-limit:],
        'total_events': len(events_log)
    })

@app.route('/api/metrics')
def get_metrics():
    return jsonify(performance_metrics)

@app.route('/api/metrics/history')
def get_metrics_history():
    # Return current metrics as history for simplicity
    return jsonify({
        'metrics': [performance_metrics],
        'total_records': 1
    })

def get_camera_frame(camera_url):
    """Get a frame from camera URL with multiple fallback strategies"""
    try:
        if not camera_url:
            print("No camera URL provided")
            return None
        
        print(f"Attempting to connect to camera: {camera_url}")
        
        # Try different URL variants for IP webcam apps
        urls_to_try = [camera_url]
        
        # If it's an IP webcam /video URL, also try snapshot alternatives
        if '/video' in camera_url:
            base_url = camera_url.replace('/video', '')
            urls_to_try.extend([
                f"{base_url}/shot.jpg",
                f"{base_url}/capture",
                f"{base_url}/cam/1/frame.jpg",
                f"{base_url}/snapshot.jpg",
                f"{base_url}/image.jpg"
            ])
        
        for attempt_url in urls_to_try:
            try:
                print(f"Trying URL: {attempt_url}")
                
                if attempt_url.startswith('http'):
                    # HTTP/HTTPS camera (snapshot or MJPEG stream)
                    response = requests.get(attempt_url, timeout=8, stream=True)
                    print(f"HTTP response status: {response.status_code}")
                    
                    if response.status_code == 200:
                        content_type = response.headers.get('Content-Type', '').lower()
                        print(f"Content-Type: {content_type}")
                        
                        if 'image' in content_type or 'jpeg' in content_type:
                            # Direct image/JPEG response
                            img_data = response.content
                            if len(img_data) > 1000:  # Valid image should be > 1KB
                                img_array = np.frombuffer(img_data, np.uint8)
                                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                                if frame is not None and frame.size > 0:
                                    print(f"Successfully decoded image from: {attempt_url}")
                                    print(f"Frame shape: {frame.shape}")
                                    return frame
                        
                        elif 'multipart' in content_type:
                            # MJPEG stream - read first frame
                            print("Processing MJPEG stream")
                            chunk_size = 0
                            buffer = b''
                            for chunk in response.iter_content(chunk_size=1024):
                                buffer += chunk
                                chunk_size += len(chunk)
                                
                                # Look for JPEG markers
                                start = buffer.find(b'\xff\xd8')  # JPEG start
                                end = buffer.find(b'\xff\xd9')    # JPEG end
                                
                                if start != -1 and end != -1 and end > start:
                                    jpeg_data = buffer[start:end+2]
                                    img_array = np.frombuffer(jpeg_data, np.uint8)
                                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                                    if frame is not None and frame.size > 0:
                                        print(f"Successfully extracted frame from MJPEG stream")
                                        print(f"Frame shape: {frame.shape}")
                                        return frame
                                
                                # Limit buffer size to prevent memory issues
                                if chunk_size > 500000:  # 500KB
                                    break
                        
                        else:
                            # Try to decode raw content as image
                            try:
                                img_data = response.content[:100000]  # First 100KB
                                if len(img_data) > 1000:
                                    img_array = np.frombuffer(img_data, np.uint8)
                                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                                    if frame is not None and frame.size > 0:
                                        print(f"Successfully decoded raw content as image")
                                        print(f"Frame shape: {frame.shape}")
                                        return frame
                            except Exception as decode_error:
                                print(f"Failed to decode as image: {decode_error}")
                
                elif attempt_url.startswith('rtsp') or attempt_url.isdigit():
                    # RTSP stream or webcam index
                    print(f"Trying OpenCV VideoCapture for: {attempt_url}")
                    cap = cv2.VideoCapture(attempt_url)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        cap.release()
                        if ret and frame is not None and frame.size > 0:
                            print(f"Successfully captured frame via OpenCV: {frame.shape}")
                            return frame
                        else:
                            print("Failed to read frame from VideoCapture")
                    else:
                        print("Failed to open VideoCapture")
                        
            except requests.exceptions.Timeout:
                print(f"Timeout connecting to: {attempt_url}")
                continue
            except requests.exceptions.ConnectionError as e:
                print(f"Connection error to: {attempt_url} - {str(e)[:100]}")
                continue
            except Exception as e:
                print(f"Error with {attempt_url}: {str(e)[:100]}")
                continue
        
        print(f"All connection attempts failed for base URL: {camera_url}")
        return None
        
    except Exception as e:
        print(f"Critical error in camera connection: {e}")
        return None

def generate_test_frame_with_detection():
    """Generate a clear 'No Camera Connected' frame when no real camera is available"""
    try:
        # Create a clean test image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Dark background
        img[:] = [25, 25, 25]
        
        # Add border
        cv2.rectangle(img, (10, 10), (630, 470), (100, 100, 100), 2)
        
        # Add main message
        cv2.putText(img, "NO CAMERA CONNECTED", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(img, "Configure camera IP in Settings", (140, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(img, "to enable live vehicle detection", (150, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(img, f"System Time: {timestamp}", (200, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
        
        # Add status
        cv2.putText(img, "Status: Waiting for camera configuration", (120, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 150, 255), 2)
        
        # Encode as JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        _, buffer = cv2.imencode('.jpg', img, encode_param)
        
        if buffer is None or len(buffer) == 0:
            print("Error: Failed to encode 'no camera' frame")
            return None
            
        return buffer.tobytes()
        
    except Exception as e:
        print(f"Error generating 'no camera' frame: {e}")
        return None

def simulate_vehicle_detection_on_frame(frame):
    """Add real vehicle detection to camera frame using YOLO"""
    if frame is None:
        return None
        
    try:
        # Use real YOLO detection
        annotated_frame, vehicle_counts = detect_vehicles_in_frame(frame)
        
        if annotated_frame is not None:
            # Encode as JPEG with good quality but optimized for speed
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]  # Reduced from 85 for better FPS
            _, buffer = cv2.imencode('.jpg', annotated_frame, encode_param)
            return buffer.tobytes()
        
        return None
        
    except Exception as e:
        print(f"Error processing frame with YOLO: {e}")
        return None

@app.route('/snapshot')
def get_snapshot():
    """Get a single snapshot frame for testing"""
    camera_id = request.args.get('camera', 'north')
    print(f"Snapshot requested for camera: {camera_id}")
    
    try:
        # Always generate a test frame for now
        frame_bytes = generate_test_frame_with_detection()
        
        if frame_bytes is not None and len(frame_bytes) > 0:
            return Response(frame_bytes, mimetype='image/jpeg')
        else:
            # Fallback error image
            error_img = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(error_img, "No Feed Available", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            _, error_buffer = cv2.imencode('.jpg', error_img)
            return Response(error_buffer.tobytes(), mimetype='image/jpeg')
    except Exception as e:
        print(f"Error generating snapshot: {e}")
        # Simple error response
        error_img = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(error_img, "Error", (100, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        _, error_buffer = cv2.imencode('.jpg', error_img)
        return Response(error_buffer.tobytes(), mimetype='image/jpeg')

@app.route('/mjpeg_feed')
def mjpeg_feed():
    """Serve optimized live camera feed with vehicle detection"""
    camera_id = request.args.get('camera', 'north')
    print(f"\n📹 === MJPEG FEED REQUEST ===\n   Camera: {camera_id}\n   Client: {request.remote_addr}\n   User-Agent: {request.headers.get('User-Agent', 'Unknown')[:50]}")
    
    def generate():
        frame_count = 0
        camera_url = config['camera_urls'].get(camera_id, '')
        skip_frames = 3  # Process every 3rd frame for smooth video
        last_detection_time = 0
        detection_interval = 3.0  # Update vehicle counts every 3 seconds
        cached_annotated_frame = None
        
        print(f"   Camera URL for {camera_id}: {camera_url[:50] + '...' if len(camera_url) > 50 else camera_url or 'NOT CONFIGURED'}")
        print(f"   🚀 OPTIMIZATION: Skip={skip_frames}, Detection interval={detection_interval}s")
        
        while True:
            try:
                frame_bytes = None
                current_time = time.time()
                
                # Get the camera URL for the specified direction
                current_url = config['camera_urls'].get(camera_id, '')
                
                # Check if camera URL is configured
                if not current_url or not current_url.strip():
                    # No camera configured - show "no camera" message
                    if frame_count % 60 == 0:  # Log every 60 frames
                        print(f"   ⚠️ No camera URL configured for {camera_id}")
                    frame_bytes = generate_test_frame_with_detection()
                elif camera_detection_enabled:
                    if frame_count % 30 == 0:  # Log every 30 frames
                        print(f"Processing frame {frame_count} from {camera_id}")
                    
                    # Try to get real camera frame
                    frame = get_camera_frame(current_url)
                    if frame is not None:
                        # Balanced resolution for good quality
                        height, width = frame.shape[:2]
                        max_width = config.get('max_resolution', 640)
                        if width > max_width:
                            scale = max_width / width
                            frame = cv2.resize(frame, (max_width, int(height * scale)))
                        
                        # Smart detection scheduling
                        should_detect = (frame_count % skip_frames == 0) or (current_time - last_detection_time > detection_interval)
                        
                        if should_detect:
                            # Run vehicle detection and update counts
                            annotated_frame, vehicle_counts = detect_vehicles_in_frame(frame.copy())
                            
                            if vehicle_counts and annotated_frame is not None:
                                # Update traffic data with real detection results
                                update_traffic_data_from_detection(camera_id, vehicle_counts)
                                last_detection_time = current_time
                                cached_annotated_frame = annotated_frame.copy()
                                
                                if frame_count % 60 == 0:  # Log detection results
                                    print(f"Detected in {camera_id}: {vehicle_counts}")
                            
                            frame_to_encode = annotated_frame if annotated_frame is not None else frame
                        else:
                            # Use cached detection result for visual consistency
                            if cached_annotated_frame is not None:
                                frame_to_encode = cached_annotated_frame.copy()
                                # Update timestamp
                                timestamp = datetime.now().strftime("%H:%M:%S")
                                device_status = config.get('device', 'cpu').upper()
                                cv2.putText(frame_to_encode, f"Live Detection - {timestamp} | {device_status} (CACHED)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            else:
                                # Just add timestamp for smooth video
                                frame_to_encode = frame.copy()
                                timestamp = datetime.now().strftime("%H:%M:%S")
                                device_status = config.get('device', 'cpu').upper()
                                cv2.putText(frame_to_encode, f"Live Feed - {timestamp} | {device_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Good quality compression for smooth video
                        jpeg_quality = config.get('jpeg_quality', 80)
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                        _, buffer = cv2.imencode('.jpg', frame_to_encode, encode_param)
                        frame_bytes = buffer.tobytes()
                    else:
                        if frame_count % 60 == 0:
                            print(f"Failed to get frame from {camera_id}, using fallback")
                
                # If no real camera or failed to get frame, use simulation
                if frame_bytes is None:
                    frame_bytes = generate_test_frame_with_detection()
                
                # Ensure we always have valid frame bytes
                if frame_bytes is not None and len(frame_bytes) > 0:
                    yield b'--frame\r\n'
                    yield b'Content-Type: image/jpeg\r\n\r\n'
                    yield frame_bytes
                    yield b'\r\n'
                else:
                    # Generate minimal error frame
                    error_img = np.zeros((240, 320, 3), dtype=np.uint8)
                    cv2.putText(error_img, "Camera Error", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    _, error_buffer = cv2.imencode('.jpg', error_img)
                    error_bytes = error_buffer.tobytes()
                    yield b'--frame\r\n'
                    yield b'Content-Type: image/jpeg\r\n\r\n'
                    yield error_bytes
                    yield b'\r\n'
                    
                # Smooth frame rate for good video quality
                time.sleep(0.033)  # ~30 FPS
                frame_count += 1
                
                # Reset frame count and clear cache periodically
                if frame_count > 600:
                    frame_count = 0
                    cached_annotated_frame = None
                    
            except GeneratorExit:
                print(f"MJPEG feed generator closed for {camera_id}")
                break
            except Exception as e:
                print(f"Error in mjpeg_feed for {camera_id}: {e}")
                # Send error frame and continue
                try:
                    error_frame = generate_test_frame_with_detection()
                    if error_frame is not None and len(error_frame) > 0:
                        yield b'--frame\r\n'
                        yield b'Content-Type: image/jpeg\r\n\r\n'
                        yield error_frame
                        yield b'\r\n'
                except Exception:
                    # Ultimate fallback
                    yield b'--frame\r\n'
                    yield b'Content-Type: text/plain\r\n\r\n'
                    yield b'Error: Camera feed unavailable'
                    yield b'\r\n'
                time.sleep(0.5)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/gpu-info', methods=['GET'])
def get_gpu_info():
    """Get comprehensive GPU information and performance metrics"""
    try:
        gpu_info = {
            'cuda_available': False,
            'current_device': 'CPU',
            'model_size': config.get('model_size', 'yolov8n'),
            'gpu_name': 'Not Available',
            'gpu_memory_total': 0,
            'gpu_memory_used': 0,
            'cpu_speed_ms': 0,
            'gpu_speed_ms': 0,
            'speedup': 1.0,
            'recommendations': [],
            'status': 'unknown'
        }
        
        # Check CUDA availability
        try:
            import torch
            import numpy as np
            
            if torch.cuda.is_available():
                gpu_info['cuda_available'] = True
                gpu_info['gpu_name'] = torch.cuda.get_device_name(0)
                gpu_props = torch.cuda.get_device_properties(0)
                gpu_info['gpu_memory_total'] = round(gpu_props.total_memory / (1024**3), 1)
                
                # Check memory usage
                if hasattr(torch.cuda, 'memory_allocated'):
                    gpu_info['gpu_memory_used'] = round(torch.cuda.memory_allocated(0) / (1024**3), 1)
                
                # Determine current device
                current_device = config.get('device', 'cpu')
                gpu_info['current_device'] = current_device.upper()
                
                # Performance benchmarking
                print("Running GPU performance benchmark...")
                test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                
                # Test CPU performance
                if yolo_model is not None:
                    model_cpu = yolo_model
                    model_cpu.to('cpu')
                    start_time = time.time()
                    _ = model_cpu(test_frame, conf=0.5, verbose=False, device='cpu')
                    cpu_time = (time.time() - start_time) * 1000
                    gpu_info['cpu_speed_ms'] = round(cpu_time, 1)
                    
                    # Test GPU performance
                    model_gpu = yolo_model
                    model_gpu.to('cuda')
                    torch.cuda.synchronize()  # Ensure proper timing
                    start_time = time.time()
                    _ = model_gpu(test_frame, conf=0.5, verbose=False, device='cuda')
                    torch.cuda.synchronize()
                    gpu_time = (time.time() - start_time) * 1000
                    gpu_info['gpu_speed_ms'] = round(gpu_time, 1)
                    
                    # Calculate speedup
                    if gpu_time > 0:
                        gpu_info['speedup'] = round(cpu_time / gpu_time, 1)
                    
                    # Restore original device
                    if current_device == 'cuda':
                        model_gpu.to('cuda')
                    else:
                        model_gpu.to('cpu')
                
                # Generate recommendations
                if gpu_info['gpu_speed_ms'] > 100:
                    gpu_info['recommendations'].append("Consider using a smaller model (yolov8n)")
                if gpu_info['current_device'] == 'CPU' and gpu_info['cuda_available']:
                    gpu_info['recommendations'].append("Enable GPU acceleration for better performance")
                if gpu_info['gpu_speed_ms'] < 50:
                    gpu_info['recommendations'].append("✅ System is optimized! Using GPU acceleration")
                else:
                    gpu_info['recommendations'].append("✅ System is optimized! Using GPU acceleration")
                
                # Overall status
                if gpu_info['current_device'] == 'CUDA' and gpu_info['gpu_speed_ms'] < 100:
                    gpu_info['status'] = 'optimal'
                elif gpu_info['current_device'] == 'CUDA':
                    gpu_info['status'] = 'good'
                else:
                    gpu_info['status'] = 'cpu_only'
                    
            else:
                gpu_info['recommendations'].append("CUDA not available - install CUDA drivers")
                gpu_info['status'] = 'no_cuda'
                
        except ImportError:
            gpu_info['recommendations'].append("PyTorch not available - install PyTorch with CUDA")
            gpu_info['status'] = 'no_pytorch'
        except Exception as e:
            gpu_info['recommendations'].append(f"Error during GPU test: {str(e)}")
            gpu_info['status'] = 'error'
        
        return jsonify(gpu_info)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error',
            'cuda_available': False
        }), 500

@app.route('/api/camera/<camera_id>/test')
def test_camera(camera_id):
    """Test if a camera is accessible"""
    camera_url = config['camera_urls'].get(camera_id, '')
    if not camera_url:
        return jsonify({'status': 'error', 'message': 'No URL configured for this camera'})
    
    print(f"Testing camera {camera_id} with URL: {camera_url}")
    frame = get_camera_frame(camera_url)
    
    if frame is not None:
        print(f"Camera test successful for {camera_id}")
        return jsonify({
            'status': 'success', 
            'message': 'Camera is accessible',
            'frame_shape': frame.shape,
            'camera_url': camera_url
        })
    else:
        print(f"Camera test failed for {camera_id}")
        return jsonify({
            'status': 'error', 
            'message': 'Failed to connect to camera',
            'camera_url': camera_url
        })

@app.route('/api/camera/diagnose', methods=['POST'])
def diagnose_camera():
    """Comprehensive camera diagnosis with auto-configuration"""
    data = request.json or {}
    test_url = data.get('url', '')
    
    if not test_url:
        return jsonify({'status': 'error', 'message': 'No URL provided for testing'})
    
    print(f"\n=== Camera Diagnosis Started for: {test_url} ===")
    
    results = {
        'original_url': test_url,
        'status': 'testing',
        'attempts': [],
        'successful_urls': [],
        'recommended_url': None,
        'error_details': []
    }
    
    # Test different URL variations
    urls_to_test = [test_url]
    
    # Generate variations based on the input URL
    base_url = test_url
    if '/video' in test_url:
        base_url = test_url.replace('/video', '')
    elif test_url.endswith('/'):
        base_url = test_url.rstrip('/')
    
    # Common IP webcam URL patterns
    test_variations = [
        test_url,
        f"{base_url}/video",
        f"{base_url}/shot.jpg",
        f"{base_url}/capture",
        f"{base_url}/cam/1/frame.jpg",
        f"{base_url}/snapshot.jpg",
        f"{base_url}/image.jpg",
        f"{base_url}/mjpeg",
        f"{base_url}/cam_pic.php",
        f"{base_url}/photoaf.jpg",
        f"{base_url}/nphMotionJpeg?Resolution=640x480&Quality=Standard"
    ]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_urls = []
    for url in test_variations:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    
    for attempt_url in unique_urls:
        print(f"Testing: {attempt_url}")
        attempt_result = {
            'url': attempt_url,
            'status': 'failed',
            'response_code': None,
            'content_type': None,
            'content_length': None,
            'frame_captured': False,
            'frame_shape': None,
            'error_message': None,
            'response_time': None
        }
        
        start_time = time.time()
        
        try:
            # Test HTTP response first
            response = requests.head(attempt_url, timeout=10)
            attempt_result['response_code'] = response.status_code
            attempt_result['content_type'] = response.headers.get('Content-Type', 'unknown')
            attempt_result['content_length'] = response.headers.get('Content-Length', 'unknown')
            attempt_result['response_time'] = round((time.time() - start_time) * 1000, 2)
            
            if response.status_code == 200:
                # Try to get actual frame
                frame = get_camera_frame(attempt_url)
                if frame is not None:
                    attempt_result['status'] = 'success'
                    attempt_result['frame_captured'] = True
                    attempt_result['frame_shape'] = frame.shape
                    results['successful_urls'].append(attempt_url)
                    
                    # Set as recommended if it's the first successful one
                    if not results['recommended_url']:
                        results['recommended_url'] = attempt_url
                else:
                    attempt_result['error_message'] = 'Failed to decode frame from response'
            else:
                attempt_result['error_message'] = f'HTTP {response.status_code}'
                
        except requests.exceptions.Timeout:
            attempt_result['error_message'] = 'Connection timeout'
            attempt_result['response_time'] = round((time.time() - start_time) * 1000, 2)
        except requests.exceptions.ConnectionError as e:
            attempt_result['error_message'] = f'Connection error: {str(e)[:100]}'
            attempt_result['response_time'] = round((time.time() - start_time) * 1000, 2)
        except Exception as e:
            attempt_result['error_message'] = f'Error: {str(e)[:100]}'
            attempt_result['response_time'] = round((time.time() - start_time) * 1000, 2)
        
        results['attempts'].append(attempt_result)
        print(f"  Result: {attempt_result['status']} - {attempt_result.get('error_message', 'OK')}")
    
    # Set final status
    if results['successful_urls']:
        results['status'] = 'success'
        results['message'] = f"Found {len(results['successful_urls'])} working URL(s)"
    else:
        results['status'] = 'failed'
        results['message'] = 'No working camera URLs found'
        
        # Add diagnostic suggestions
        results['suggestions'] = [
            "Check if the camera IP address is correct and reachable",
            "Verify the camera is powered on and connected to network",
            "Ensure your device is on the same network as the camera",
            "Try accessing the camera URL directly in a web browser",
            "Check if the camera requires authentication",
            "Verify the camera app (like IP Webcam) is running on the device"
        ]
    
    print(f"=== Diagnosis Complete: {results['status']} ===")
    return jsonify(results)

@app.route('/api/camera/auto-configure', methods=['POST'])
def auto_configure_camera():
    """Auto-configure camera settings based on successful diagnosis"""
    data = request.json or {}
    camera_id = data.get('camera_id', 'north')
    recommended_url = data.get('recommended_url', '')
    
    if not recommended_url:
        return jsonify({'status': 'error', 'message': 'No recommended URL provided'})
    
    # Update configuration
    config['camera_urls'][camera_id] = recommended_url
    
    # Enable camera detection
    global camera_detection_enabled
    camera_detection_enabled = True
    
    events_log.append({
        'type': 'camera_auto_configured',
        'timestamp': time.time(),
        'data': {
            'camera_id': camera_id,
            'url': recommended_url,
            'auto_configured': True
        }
    })
    
    print(f"Auto-configured {camera_id} camera with URL: {recommended_url}")
    
    return jsonify({
        'status': 'success',
        'message': f'Camera {camera_id} auto-configured successfully',
        'camera_id': camera_id,
        'configured_url': recommended_url,
        'camera_detection_enabled': camera_detection_enabled
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'Smart Traffic Management Backend',
        'version': '1.0.0',
        'timestamp': time.time()
    })

# Frontend routes
@app.route('/')
def index():
    """Serve the main HTML file"""
    return send_file(os.path.join(FRONTEND_DIR, 'index.html'))

@app.route('/<path:filename>')
def frontend_files(filename):
    """Serve frontend static files"""
    return send_from_directory(FRONTEND_DIR, filename)

if __name__ == '__main__':
    print("=" * 60)
    print("🚦 Smart Traffic Management System - Backend API Server")
    print("=" * 60)
    print(f"🌐 API Server URL: http://localhost:5000")
    print(f"📊 API Status: http://localhost:5000/api/status")
    print(f"🔧 Health Check: http://localhost:5000/health")
    print("=" * 60)
    print("📡 Available API Endpoints:")
    print("   - GET  /api/status")
    print("   - POST /api/start")
    print("   - POST /api/stop")
    print("   - GET  /api/vehicle-counts")
    print("   - GET  /api/signals")
    print("   - GET  /api/config")
    print("   - GET  /api/events")
    print("   - GET  /api/metrics")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
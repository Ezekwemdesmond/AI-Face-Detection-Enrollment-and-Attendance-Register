import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import sqlite3
from deepface import DeepFace
import gc
from tqdm import tqdm
import tensorflow as tf

# Set memory growth for GPU if available
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

# Set environment variables for memory optimization
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_MEMORY_ALLOCATION'] = '2048'

class AttendanceSystem:
    def __init__(self, db_path='attendance.db', model_name="OpenFace"):
        self.db_path = db_path
        self.model_name = model_name
        self.detector_backend = 'opencv'
        self.setup_database()
        gc.collect()
    
    def setup_database(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''CREATE TABLE IF NOT EXISTS students
                    (id INTEGER PRIMARY KEY,
                     name TEXT NOT NULL,
                     face_embedding BLOB,
                     photo_path TEXT)''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS attendance
                    (id INTEGER PRIMARY KEY,
                     student_id INTEGER,
                     timestamp DATETIME,
                     confidence FLOAT,
                     FOREIGN KEY (student_id) REFERENCES students(id))''')
        
        conn.commit()
        conn.close()

    def optimize_image(self, image_path, max_size=224):
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image at {image_path}")
            
            height, width = img.shape[:2]
            if height > max_size or width > max_size:
                ratio = max_size / max(height, width)
                new_size = (int(width * ratio), int(height * ratio))
                img = cv2.resize(img, new_size)
            
            optimized_path = f"temp_optimized_{os.path.basename(image_path)}"
            cv2.imwrite(optimized_path, img)
            
            return optimized_path
        except Exception as e:
            print(f"Error optimizing image: {str(e)}")
            return None

    def enroll_student(self, name, image_path):
        temp_path = None
        try:
            temp_path = self.optimize_image(image_path)
            if not temp_path:
                raise ValueError("Failed to optimize image")
            
            gc.collect()
            
            # Get embedding representation
            embedding_dict = DeepFace.represent(
                img_path=temp_path,
                model_name=self.model_name,
                enforce_detection=True,
                detector_backend=self.detector_backend
            )
            
            # Extract the embedding array
            embedding = embedding_dict[0]['embedding']
            
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute("""INSERT INTO students (name, face_embedding, photo_path) 
                        VALUES (?, ?, ?)""",
                     (name, np.array(embedding).tobytes(), image_path))
            
            student_id = c.lastrowid
            conn.commit()
            conn.close()
            
            print(f"Successfully enrolled {name} with ID {student_id}")
            return student_id
            
        except Exception as e:
            print(f"Error enrolling student: {str(e)}")
            return None
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            gc.collect()

    def verify_face(self, frame, student_data):
        temp_path = None
        try:
            temp_path = "temp_frame.jpg"
            cv2.imwrite(temp_path, frame)
            
            # Get embedding representation
            embedding_dict = DeepFace.represent(
                img_path=temp_path,
                model_name=self.model_name,
                enforce_detection=True,
                detector_backend=self.detector_backend
            )
            
            # Extract the embedding array from the dictionary
            embedding = embedding_dict[0]['embedding']
            
            best_match = None
            best_distance = float('inf')
            
            for student_id, name, embedding_bytes, _ in student_data:
                stored_embedding = np.frombuffer(embedding_bytes, dtype=np.float64)
                distance = np.linalg.norm(np.array(embedding) - stored_embedding)
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = (student_id, name, distance)
            
            return best_match if best_match and best_distance < 0.8 else None
            
        except Exception as e:
            print(f"Error in verification: {str(e)}")
            return None
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            gc.collect()

    def mark_attendance(self, student_id, confidence):
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute("SELECT name FROM students WHERE id = ?", (student_id,))
            student_name = c.fetchone()[0]
            
            current_time = datetime.now()
            one_minute_ago = current_time.timestamp() - 60
            
            c.execute("""SELECT COUNT(*) FROM attendance 
                        WHERE student_id = ? AND 
                        timestamp > datetime(?,'unixepoch')""",
                     (student_id, one_minute_ago))
            
            if c.fetchone()[0] == 0:
                c.execute("""INSERT INTO attendance 
                           (student_id, timestamp, confidence) 
                           VALUES (?, ?, ?)""",
                         (student_id, current_time, confidence))
                conn.commit()
                print(f"\nSuccessfully marked attendance for {student_name}")
                print(f"Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Confidence: {(1 - confidence):.2f}")
                return True, student_name
            else:
                print(f"\nAttendance already marked for {student_name} in the last minute")
                return False, student_name
            
        except Exception as e:
            print(f"Error marking attendance: {str(e)}")
            return False, None
        finally:
            conn.close()

    def start_recognition(self):
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("SELECT id, name, face_embedding, photo_path FROM students")
            student_data = c.fetchall()
            conn.close()
            
            if not student_data:
                print("No students enrolled in the system")
                return
            
            video_capture = cv2.VideoCapture(0)
            if not video_capture.isOpened():
                print("Error: Could not open camera")
                return
            
            print("\nAttendance System Started:")
            print("- Press ENTER to mark attendance")
            print("- Press 'q' to quit")
            
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    continue
                
                display_frame = frame.copy()
                
                # Add instructions to the frame
                cv2.putText(display_frame, 
                          "Press ENTER to mark attendance", 
                          (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, 
                          (255, 255, 255), 
                          2)
                
                # Show the frame
                cv2.imshow('Attendance System', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                # Check for Enter key
                if key == 13:  # Enter key
                    print("\nProcessing attendance...")
                    try:
                        # Create a "processing" display
                        processing_frame = display_frame.copy()
                        cv2.putText(processing_frame, 
                                  "Processing... Please wait", 
                                  (10, 60),
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.7, 
                                  (0, 255, 255), 
                                  2)
                        cv2.imshow('Attendance System', processing_frame)
                        cv2.waitKey(1)
                        
                        # Verify face
                        match = self.verify_face(frame, student_data)
                        
                        if match:
                            student_id, name, confidence = match
                            success, student_name = self.mark_attendance(student_id, confidence)
                            
                            if success:
                                # Show success message
                                result_frame = display_frame.copy()
                                cv2.putText(result_frame, 
                                          f"Attendance marked for: {name}", 
                                          (10, 60),
                                          cv2.FONT_HERSHEY_SIMPLEX, 
                                          0.7, 
                                          (0, 255, 0), 
                                          2)
                                cv2.putText(result_frame,
                                          f"Confidence: {(1 - confidence):.2f}",
                                          (10, 90),
                                          cv2.FONT_HERSHEY_SIMPLEX,
                                          0.7,
                                          (0, 255, 0),
                                          2)
                            else:
                                # Show already marked message
                                result_frame = display_frame.copy()
                                cv2.putText(result_frame, 
                                          f"Attendance already marked for: {name}", 
                                          (10, 60),
                                          cv2.FONT_HERSHEY_SIMPLEX, 
                                          0.7, 
                                          (0, 255, 255), 
                                          2)
                        else:
                            # Show no match found message
                            result_frame = display_frame.copy()
                            cv2.putText(result_frame, 
                                      "No matching student found", 
                                      (10, 60),
                                      cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.7, 
                                      (0, 0, 255), 
                                      2)
                            print("No matching student found")
                        
                        # Display result for 2 seconds
                        cv2.imshow('Attendance System', result_frame)
                        cv2.waitKey(2000)
                        
                    except Exception as e:
                        print(f"Error processing attendance: {str(e)}")
                
                # Check for 'q' to quit
                elif key == ord('q'):
                    print("\nShutting down attendance system...")
                    break
            
            video_capture.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error in recognition system: {str(e)}")
        finally:
            gc.collect()

def capture_enrollment_photo(save_path):
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return False
        
        print("\nCapturing enrollment photo:")
        print("- Press SPACE to capture")
        print("- Press ESC to cancel")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            cv2.imshow('Enrollment Photo', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE key
                cv2.imwrite(save_path, frame)
                success = True
                break
            elif key == 27:  # ESC key
                success = False
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return success and os.path.exists(save_path)
        
    except Exception as e:
        print(f"Error capturing photo: {str(e)}")
        return False

def generate_report(db_path, start_date=None, end_date=None):
    try:
        conn = sqlite3.connect(db_path)
        query = """
            SELECT 
                s.name,
                a.timestamp,
                a.confidence
            FROM students s
            JOIN attendance a ON s.id = a.student_id
            WHERE 1=1
        """
        
        params = []
        if start_date:
            query += " AND a.timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND a.timestamp <= ?"
            params.append(end_date)
            
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if df.empty:
            print("No attendance records found")
            return None
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        # Create attendance pivot table
        attendance_pivot = df.pivot_table(
            index='name',
            columns='date',
            values='confidence',
            aggfunc='count',
            fill_value=0
        )
        
        return attendance_pivot
        
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        return None

if __name__ == "__main__":
    # Clear memory before starting
    gc.collect()
    
    # Initialize system
    attendance_system = AttendanceSystem(model_name="OpenFace")
    
    # Create directory for storing images
    image_dir = "student_images"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    while True:
        print("\nAttendance System Menu:")
        print("1. Enroll new student")
        print("2. Start attendance system")
        print("3. Generate attendance report")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            name = input("Enter student name: ")
            image_path = os.path.join(image_dir, f"{name.replace(' ', '_')}.jpg")
            
            if capture_enrollment_photo(image_path):
                student_id = attendance_system.enroll_student(name, image_path)
                if student_id:
                    print(f"\nSuccessfully enrolled {name}")
                else:
                    print("\nEnrollment failed")
                    if os.path.exists(image_path):
                        os.remove(image_path)
            else:
                print("\nPhoto capture cancelled or failed")
                
        elif choice == '2':
            attendance_system.start_recognition()
            
        elif choice == '3':
            print("\nGenerating attendance report...")
            report = generate_report(attendance_system.db_path)
            if report is not None:
                print("\nAttendance Report:")
                print(report)
            
        elif choice == '4':
            print("\nExiting system...")
            break
            
        else:
            print("\nInvalid choice. Please try again.")
    
    print("System shutdown complete.")
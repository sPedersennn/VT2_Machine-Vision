import cv2
import os
import time

def capture_images(folder_name, base_dir, num_images=200, duration=35):
    fps = num_images / duration
    delay = 1 / fps

    # Full path for current folder
    save_path = os.path.join(base_dir, folder_name)
    os.makedirs(save_path, exist_ok=True)

    # Open webcam 
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print(f"Starting capture for '{folder_name}'...")
    start_time = time.time()

    for i in range(num_images):
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Failed to capture image {i}")
            continue

        # Show the frame in a window
        cv2.imshow("Capturing - Press 'q' to quit", frame)

        # Save the frame as image
        filename = os.path.join(save_path, f"image_{i:03d}.jpg")
        cv2.imwrite(filename, frame)

        # Allow exiting early by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Capture interrupted by user.")
            break

        time.sleep(delay)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished capturing {i+1} images for '{folder_name}' in {time.time() - start_time:.2f} seconds.\n")

def main():
    # Get the path to the current script's directory
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(repo_dir, "Dataset")
    folders = []

    print("Images will be saved to a 'Dataset' folder inside this script's directory.")
    print("You will define 4 folders. Each will store 200 images taken over 10 seconds.\n")

    for i in range(4):
        folder_name = input(f"Enter name for folder #{i+1}: ")
        folders.append(folder_name)

    for folder in folders:
        input(f"\nReady to start capturing '{folder}'. Press Enter to continue...")
        capture_images(folder, base_dir)

    print("All captures complete. Images saved under:", base_dir)

if __name__ == "__main__":
    main()

import cv2
import numpy as np
import os
import sys
from main import Sentinel, ROIS

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_identify_legend():
    """
    Test the identify_legend function using pre-captured test images.
    """
    sentinel = Sentinel()
    
    # Test image paths
    test_images = [
        "screenshots/training/Duos World's Edge.png",
        "screenshots/training/Battle Royale 2.png"
    ]
    
    print("Testing identify_legend function...")
    print("=" * 50)
    
    for i, image_path in enumerate(test_images):
        if not os.path.exists(image_path):
            print(f"Warning: Test image '{image_path}' not found. Skipping...")
            continue
            
        # Load the full test image
        full_image = cv2.imread(image_path)
        if full_image is None:
            print(f"Error: Could not load image '{image_path}'. Skipping...")
            continue
            
        print(f"\nTest Image {i+1}: {os.path.basename(image_path)}")
        print(f"Image dimensions: {full_image.shape[1]}x{full_image.shape[0]}")
        
        # Extract legend region using the same ROI coordinates
        legend_roi = ROIS['legend']
        legend_region = full_image[
            legend_roi['top']:legend_roi['top'] + legend_roi['height'],
            legend_roi['left']:legend_roi['left'] + legend_roi['width']
        ]
        
        # Test legend identification
        detected_legend = sentinel.identify_legend(legend_region)
        
        print(f"Detected Legend: {detected_legend}")
        
        # Display the extracted region for visual inspection
        cv2.imshow(f"Legend Region - Test {i+1}", legend_region)
        cv2.imshow(f"Full Image - Test {i+1}", cv2.resize(full_image, (1280, 720)))
        
        # Wait for key press to continue to next image
        print("Press any key to continue to next test image...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("\nTesting complete!")

def test_all_regions():
    """
    Test all detection functions on the test images.
    """
    # Initialize Sentinel instance
    sentinel = Sentinel()
    
    # Load World's Edge as default map
    sentinel.load_current_map("WORLD'S EDGE")
    
    # Test image paths
    test_images = [
        "screenshots/training/Battle Royale 2.png",
        "screenshots/training/Battle Royale 3.png",
        "screenshots/training/Battle Royale 4.png",
        "screenshots/training/Battle Royale 5.png"
    ]
    
    print("Testing all detection functions...")
    print(f"Default map loaded: World's Edge")
    print("=" * 50)
    
    for i, image_path in enumerate(test_images):
        if not os.path.exists(image_path):
            print(f"Warning: Test image '{image_path}' not found. Skipping...")
            continue
            
        # Load the full test image
        full_image = cv2.imread(image_path)
        if full_image is None:
            print(f"Error: Could not load image '{image_path}'. Skipping...")
            continue
            
        print(f"\nTest Image {i+1}: {os.path.basename(image_path)}")
        print("-" * 30)
        
        # Extract all regions
        weapon_region = full_image[
            ROIS['weapon']['top']:ROIS['weapon']['top'] + ROIS['weapon']['height'],
            ROIS['weapon']['left']:ROIS['weapon']['left'] + ROIS['weapon']['width']
        ]
        
        legend_region = full_image[
            ROIS['legend']['top']:ROIS['legend']['top'] + ROIS['legend']['height'],
            ROIS['legend']['left']:ROIS['legend']['left'] + ROIS['legend']['width']
        ]
        
        minimap_region = full_image[
            ROIS['minimap']['top']:ROIS['minimap']['top'] + ROIS['minimap']['height'],
            ROIS['minimap']['left']:ROIS['minimap']['left'] + ROIS['minimap']['width']
        ]
        
        # Test detection functions
        detected_weapon = sentinel.identify_weapon(weapon_region)
        detected_legend = sentinel.identify_legend(legend_region)
        
        # Test player position detection
        print(f"Testing minimap matching against full map...")
        print(f"Minimap dimensions: {minimap_region.shape[:2]}")
        
        # The new find_minimap_location_on_map function directly returns the player's world position
        # by finding the center of the matched (and rotated) minimap region on the full map.
        world_position = sentinel.find_minimap_location_on_map(minimap_region)
        
        # Display results
        print(f"Detected Weapon: {detected_weapon}")
        print(f"Detected Legend: {detected_legend}")
        print(f"Current Map: {sentinel.current_map_name}")
            
        if world_position:
            print(f"Final Player Position (World): {world_position}")
        else:
            print("Player Position: Could not determine world coordinates")
        
        # Display extracted regions
        cv2.imshow(f"Weapon Region - Test {i+1}", weapon_region)
        cv2.imshow(f"Legend Region - Test {i+1}", legend_region)
        cv2.imshow(f"Minimap Region - Test {i+1}", minimap_region)
        
        # Display map with player position if available
        if sentinel.current_map is not None:
            map_with_position = sentinel.draw_player_position(sentinel.current_map, world_position)
            display_map = cv2.resize(map_with_position, (1200, 1200))
            cv2.imshow(f"Map with Position - Test {i+1}", display_map)
        
        print("\nPress any key to continue to next test image...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("\nTesting complete!")

if __name__ == "__main__":
    print("Sentinel Testing Suite")
    print("1. Test legend detection only")
    print("2. Test all detection functions")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        test_identify_legend()
    elif choice == "2":
        test_all_regions()
    else:
        print("Invalid choice. Running legend detection test...")
        test_identify_legend()
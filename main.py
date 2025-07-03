import cv2
import numpy as np
import mss
import pytesseract
import time
import os

# Set the path to the Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Regions of interest for game elements
ROIS = {
    'weapon': {'top': 1275, 'left': 2055, 'width': 245, 'height': 90},
}

# Screenshot folders
WEAPON_TEMPLATE_FOLDER = 'screenshots/weapon_templates'

class Sentinel:
    '''
    A class to handle the vision processing for Apex Legends.
    '''
    def __init__(self):
        self.sct = mss.mss()
        self.weapon_templates = self.load_weapon_templates()


    def load_weapon_templates(self):
        templates = {}
        if not os.path.exists(WEAPON_TEMPLATE_FOLDER):
            print(f"Warning: Template folder '{WEAPON_TEMPLATE_FOLDER}' does not exist.")
            return templates
        
        # Load all weapon templates
        for filename in os.listdir(WEAPON_TEMPLATE_FOLDER):
            if filename.endswith('.png'):
                weapon_name = os.path.splitext(filename)[0] # Remove file extension
                template_path = os.path.join(WEAPON_TEMPLATE_FOLDER, filename)
                
                template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    templates[weapon_name] = template
                else:
                    print(f"Warning: Could not load template '{template_path}'")

        print(f"Loaded {len(templates)} weapon templates.")
        return templates
    

    def capture_screenshot(self, region):
        '''
        Capture a screenshot of the specified region.
        Region should be a dictionary with keys: 'top', 'left', 'width', 'height'.
        '''
        screenshot = self.sct.grab(region)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) # Discards alpha channel for OpenCV
        return img
    

    def identify_weapon(self, screenshot):
        '''
        Identify the weapon in the screenshot using template matching.
        Returns the name of the weapon if found, otherwise None.
        '''
        if not self.weapon_templates:
            print("No weapon templates loaded.")
            return None

        gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        best_match = {'weapon': None, 'score': 0.7} # Threshold for matching

        for weapon_name, template in self.weapon_templates.items():
            result = cv2.matchTemplate(gray_screenshot, template, cv2.TM_CCOEFF_NORMED) # Comparison that returns a 2D array of scores
            _, max_val, _, _ = cv2.minMaxLoc(result) # minMaxLoc returns (min_val, max_val, min_loc, max_loc), so we only care about max_val
            if max_val > best_match['score']:
                best_match['weapon'] = weapon_name
                best_match['score'] = max_val

        return best_match['weapon'] if best_match['weapon'] else "Unknown weapon"

    
    def run(self):
        '''
        Main loop to capture screenshots and identify weapons.
        '''
        print("Starting Sentinel... Press 'q' to quit.")
        print("-" * 40)
        while True:
            # Capture the screenshot of the weapon region
            weapon_screenshot = self.capture_screenshot(ROIS['weapon'])

            # Identify the weapon in the screenshot
            current_weapon = self.identify_weapon(weapon_screenshot)

            # Display results
            print(f"Timestamp: {time.strftime('%H:%M:%S')}")
            print(f"Current Weapon: {current_weapon}")
            print("-" * 40)

            # Debugging
            cv2.imshow("Weapon Screenshot", weapon_screenshot)
            
            # Terminate the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(2)
        
        cv2.destroyAllWindows()
        print("Sentinel stopped.")
    
if __name__ == "__main__":
    sentinel = Sentinel()
    sentinel.run()
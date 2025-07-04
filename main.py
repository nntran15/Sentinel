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
    'legend': {'top': 1275, 'left': 115, 'width': 115, 'height': 90},
    'gamemode': {'top': 0, 'left': 0, 'width': 300, 'height': 150},
}

# Screenshot template folders
WEAPON_TEMPLATE_FOLDER = 'screenshots/weapon_templates'
LEGEND_TEMPLATE_FOLDER = 'screenshots/legend_templates'

class Sentinel:
    '''
    A class to handle the vision processing for Apex Legends.
    '''
    def __init__(self):
        self.sct = mss.mss()
        self.weapon_templates = self.load_weapon_templates()
        self.legend_templates = self.load_legend_templates()


    """""""""""""""""""""""""""""
        WEAPON DETECTION LOGIC
    """""""""""""""""""""""""""""

    def load_weapon_templates(self):
        '''
        Load weapon templates from the specified folder.
        Returns a dictionary with weapon names as keys and their templates as values.
        '''
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
            #print(f"Weapon: {weapon_name}, Match Score: {max_val:.4f}")  # Debugging output
            if max_val > best_match['score']:
                best_match['weapon'] = weapon_name
                best_match['score'] = max_val

        return best_match['weapon'] if best_match['weapon'] else "Unknown weapon"


    """""""""""""""""""""""""""""
        LEGEND DETECTION LOGIC
    """""""""""""""""""""""""""""

    def load_legend_templates(self):
        '''
        Load legend templates from the specified folder.
        Returns a dictionary with legend names as keys and their templates as values.
        '''
        templates = {}
        if not os.path.exists(LEGEND_TEMPLATE_FOLDER):
            print(f"Warning: Template folder '{LEGEND_TEMPLATE_FOLDER}' does not exist.")
            return templates

        # Load all legend templates
        for filename in os.listdir(LEGEND_TEMPLATE_FOLDER):
            if filename.endswith('.png'):
                legend_name = os.path.splitext(filename)[0] # Remove file extension
                template_path = os.path.join(LEGEND_TEMPLATE_FOLDER, filename)

                template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    templates[legend_name] = template
                else:
                    print(f"Warning: Could not load template '{template_path}'")

        print(f"Loaded {len(templates)} legend templates.")
        return templates
    

    def identify_legend(self, screenshot):
        '''
        Identify the legend in the screenshot using template matching.
        Returns the name of the legend if found, otherwise None.
        '''
        if not self.legend_templates:
            print("No legend templates loaded.")
            return None

        gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        best_match = {'legend': None, 'score': 0.6} # Threshold for matching

        for legend_name, template in self.legend_templates.items():
            result = cv2.matchTemplate(gray_screenshot, template, cv2.TM_CCOEFF_NORMED) # Comparison that returns a 2D array of scores
            _, max_val, _, _ = cv2.minMaxLoc(result) # minMaxLoc returns (min_val, max_val, min_loc, max_loc), so we only care about max_val
            if max_val > best_match['score']:
                best_match['legend'] = legend_name
                best_match['score'] = max_val

        return best_match['legend'] if best_match['legend'] else "Unknown legend"


    """""""""""""""""""""""""""""
       GAMEMODE DETECTION LOGIC
    """""""""""""""""""""""""""""
    def identify_gamemode(self, screenshot):
        '''
        Identify the gamemode and map in the screenshot using OCR.
        Returns a dictionary with 'gamemode' and 'map' keys if found, otherwise None.
        '''
        gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for better OCR performance
        _, threshold = cv2.threshold(gray_screenshot, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # Converts to black and white for isolation of text
                                                                                                        # cv2.threshold returns optimal threshold value aand resulting binary image
        # OCR configuration for Tesseract
        custom_config = r'--oem 3 --psm 6 tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '
        allowed_gamemodes = ['DUOS', 'TRIOS', 'GUN RUN', 'RANKED']
        allowed_maps = ['WORLD\'S EDGE', 'STORM POINT', 'OLYMPUS', 'E-DISTRICT', 'KING\'S CANYON', 'MONUMENT']
        
        try:
            text = pytesseract.image_to_string(threshold, config=custom_config)
            text = text.strip()  # Remove leading/trailing whitespace
            
            if not text:
                return None
            
            # Initialize result dictionary
            result = {'gamemode': None, 'map': None}
            
            # Extract gamemode
            for gamemode in allowed_gamemodes:
                if gamemode in text:
                    result['gamemode'] = gamemode
                    break
            
            # Extract map
            for map_name in allowed_maps:
                if map_name in text:
                    result['map'] = map_name
                    break
            
            # Return result if at least one value was found
            if result['gamemode'] or result['map']:
                return result
            else:
                return None
                
        except Exception as e:
            print(f"Error during OCR: {e}")
            return None


    """""""""""""""""""""""""""""
         APPLICATION LOGIC
    """""""""""""""""""""""""""""

    def capture_screenshot(self, region):
        '''
        Capture a screenshot of the specified region.
        Region should be a dictionary with keys: 'top', 'left', 'width', 'height'.
        '''
        screenshot = self.sct.grab(region)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) # Discards alpha channel for OpenCV
        return img


    def run(self):
        '''
        Main loop to capture screenshots and identify weapons.
        '''
        print("Starting Sentinel... Press 'q' to quit.")
        print("-" * 40)
        while True:
            # Capture desired regions
            weapon_screenshot = self.capture_screenshot(ROIS['weapon'])
            legend_screenshot = self.capture_screenshot(ROIS['legend'])
            gamemode_screenshot = self.capture_screenshot(ROIS['gamemode'])

            # Identify desired regions
            current_weapon = self.identify_weapon(weapon_screenshot)
            current_legend = self.identify_legend(legend_screenshot)
            current_gamemode = self.identify_gamemode(gamemode_screenshot)

            # Display results
            print(f"Timestamp: {time.strftime('%H:%M:%S')}")
            print(f"Current Weapon: {current_weapon}")
            print(f"Current Legend: {current_legend}")
            
            # Display gamemode and map
            if current_gamemode:
                print(f"Current Gamemode: {current_gamemode.get('gamemode', 'Unknown gamemode')}")
                print(f"Current Map: {current_gamemode.get('map', 'Unknown map')}")
            else:
                print("Current Gamemode: None detected")
                print("Current Map: None detected")
            
            print("-" * 40)

            # Debugging (MAY COMMENT OUT)
            #cv2.imshow("Weapon Screenshot", weapon_screenshot)
            #cv2.imshow("Legend Screenshot", legend_screenshot)
            cv2.imshow("Gamemode Screenshot", gamemode_screenshot)
            
            # Terminate the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(2)   # Delay between screenshots
        
        cv2.destroyAllWindows()
        print("Sentinel stopped.")
    
if __name__ == "__main__":
    sentinel = Sentinel()
    sentinel.run()
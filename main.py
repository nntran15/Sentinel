import cv2
import numpy as np
import mss
import pytesseract
import time
import os

# Set the path to the Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Screenshot template folders
WEAPON_TEMPLATE_FOLDER = 'screenshots/weapon_templates'
LEGEND_TEMPLATE_FOLDER = 'screenshots/legend_templates'
MAP_TEMPLATE_FOLDER = 'screenshots/map_templates'

# Regions of interest for game elements
ROIS = {
    'weapon': {'top': 1275, 'left': 2055, 'width': 245, 'height': 90},
    'legend': {'top': 1275, 'left': 115, 'width': 115, 'height': 90},
    'gamemode': {'top': 0, 'left': 0, 'width': 300, 'height': 150},
    'minimap': {'top': 0, 'left': 0, 'width': 500, 'height': 500},  # Approximate minimap location
}

class Sentinel:
    '''
    A class to handle the vision processing for Apex Legends.
    '''
    def __init__(self):
        self.sct = mss.mss()
        self.weapon_templates = self.load_weapon_templates()
        self.legend_templates = self.load_legend_templates()
        self.map_templates = self.load_map_templates()
        self.current_map = None
        self.current_map_name = None
        self.last_gamemode_check = None


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
         MAP DETECTION LOGIC
    """""""""""""""""""""""""""""
    
    def load_map_templates(self):
        '''
        Load map templates from the specified folder.
        Returns a dictionary with map names as keys and their templates as values.
        '''
        templates = {}
        if not os.path.exists(MAP_TEMPLATE_FOLDER):
            print(f"Warning: Template folder '{MAP_TEMPLATE_FOLDER}' does not exist.")
            return templates
        
        # Load all map templates
        for filename in os.listdir(MAP_TEMPLATE_FOLDER):
            if filename.endswith('.png'):
                map_name = os.path.splitext(filename)[0] # Remove file extension
                template_path = os.path.join(MAP_TEMPLATE_FOLDER, filename)
                
                template = cv2.imread(template_path, cv2.IMREAD_COLOR)
                if template is not None:
                    templates[map_name] = template
                else:
                    print(f"Warning: Could not load map template '{template_path}'")

        print(f"Loaded {len(templates)} map templates.")
        return templates
    
    def load_current_map(self, map_name):
        '''
        Load the current map based on the detected map name (map_name).
        Updates self.current_map and self.current_map_name.
        '''
        if not map_name:
            return False
            
        # Convert map name to match template filename
        template_name = map_name
        
        if template_name in self.map_templates:
            self.current_map = self.map_templates[template_name]
            self.current_map_name = map_name
            print(f"Loaded map: {map_name}")
            return True
        else:
            print(f"Warning: Map template for '{map_name}' not found. Available maps: {list(self.map_templates.keys())}")
            return False
    
    def detect_minimap_type(self, minimap_screenshot):
        '''
        Detect if the minimap is fixed or rotating based on visual cues.
        Returns 'fixed' or 'rotating' or 'unknown' (error?).
        '''
        # TEST: just assume rotating for now lol
        return 'rotating'
    
    def find_minimap_location_on_map(self, minimap_screenshot):
        '''
        Find where the minimap section appears on the full map using feature-based matching (ORB).
        Returns the center of the matched minimap area on the full map, which is the player's world position.
        '''
        if self.current_map is None:
            print("No map loaded for minimap matching")
            return None

        # Initialize ORB detector (2000 keypoints-- can be adjusted for speed)
        orb = cv2.ORB_create(nfeatures=2000)

        # Find keypoints and descriptors for the minimap and the full map
        '''
            EXAMPLE:
            kp1 = [
                cv2.KeyPoint(x=100, y=100, _size=20),
                cv2.KeyPoint(x=150, y=150, _size=20),
                ...
            ]
        '''
        kp1, des1 = orb.detectAndCompute(minimap_screenshot, None)      # Descriptors are vectors for each keypoint
        kp2, des2 = orb.detectAndCompute(self.current_map, None)

        if des1 is None or des2 is None:
            print("Could not compute descriptors for feature matching.")
            return None

        # Use a Brute-Force Matcher
        '''
            EXAMPLE:
            matches = [
                cv2.DMatch(queryIdx=1, trainIdx=0, distance=10.5),      # minimap_kp[1] corresponds to full_map_kp[0]
                cv2.DMatch(queryIdx=2, trainIdx=1, distance=12.3),      # minimap_kp[2] corresponds to full_map_kp[1]
                ...
            ]
        '''
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2) # List of DMatch objects

        # Sort matches by distance (best matches first)
        matches = sorted(matches, key=lambda x: x.distance)

        # Keep only the best matches
        good_matches_threshold = 15
        if len(matches) < good_matches_threshold:
            print(f"Not enough good matches found - {len(matches)}/{good_matches_threshold}")
            return None
        
        good_matches = matches[:50] # Use top 50 matches

        # Extract location of good matches
        '''
            src_pts: (x,y) coordinates of keypoints in the minimap
            dst_pts: (x,y) coordinates of keypoints in the full map
        '''
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find the transformation matrix
        # "Given two sets of points, find the best transformation matrix that maps minimap to full map using RANSAC while ignoring matches > 5 pixels"
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is None:
            print("Could not find a valid homography.")
            return None

        # Get the corners of the minimap
        h, w, _ = minimap_screenshot.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        
        # Transform the corners to find their position on the main map
        dst = cv2.perspectiveTransform(pts, M)

        # Calculate the center of the matched region on the main map
        center_x = int(np.mean([pt[0][0] for pt in dst]))
        center_y = int(np.mean([pt[0][1] for pt in dst]))

        # For a rotating minimap, the player is at the center of the minimap view.
        # Therefore, the center of the matched region on the main map is the player's location.
        print(f"Player world position found at: ({center_x}, {center_y})")
        return (center_x, center_y)

    
    def find_player_position(self, minimap_screenshot):
        '''
        Find the player's position on the minimap.
        For rotating minimaps, player is always at the center.
        Returns (x, y) coordinates of the player on the minimap, or None if not found.
        '''            
        # Convert minimap to grayscale for analysis
        gray_minimap = cv2.cvtColor(minimap_screenshot, cv2.COLOR_BGR2GRAY)
        
        # Detect minimap type
        minimap_type = self.detect_minimap_type(minimap_screenshot)
        
        if minimap_type == 'rotating':
            # For rotating minimaps, player is always at the center
            center_x = gray_minimap.shape[1] // 2  # width // 2
            center_y = gray_minimap.shape[0] // 2  # height // 2
            return (center_x, center_y)
        
        # TODO: implement fixed minimap detection logic if needed
        return None
    
    def map_minimap_to_world(self, minimap_pos, minimap_screenshot):
        '''
        Map the minimap position to world coordinates on the full map.
        Returns (x, y) coordinates on the full map image.
        '''
        if self.current_map is None or minimap_pos is None:
            return None
            
        minimap_x, minimap_y = minimap_pos
        minimap_h, minimap_w = minimap_screenshot.shape[:2]
        map_h, map_w = self.current_map.shape[:2]
        
        # Simple scaling approach - this would need refinement based on actual game mechanics
        # Assuming the minimap represents a portion of the full map
        # This is a simplified calculation and would need adjustment based on actual minimap scaling
        
        # Convert minimap coordinates to normalized coordinates (0-1)
        norm_x = minimap_x / minimap_w
        norm_y = minimap_y / minimap_h
        
        # Map to full map coordinates
        world_x = int(norm_x * map_w)
        world_y = int(norm_y * map_h)
        
        return (world_x, world_y)
    
    def draw_player_position(self, map_image, world_pos):
        '''
        Draw a green rectangle around the player's estimated position on the map.
        Returns the modified map image.
        '''
        if world_pos is None:
            return map_image
            
        world_x, world_y = world_pos
        
        # Create a copy of the map to draw on
        map_with_position = map_image.copy()
        
        # Draw a green rectangle around the estimated position
        rect_size = 30  # Size of the rectangle
        offset_horizontal = 30
        offset_vertical = -15
        top_left = (world_x - rect_size // 2 + offset_horizontal, world_y - rect_size // 2 + offset_vertical)
        bottom_right = (world_x + rect_size // 2 + offset_horizontal, world_y + rect_size // 2 + offset_vertical)

        cv2.rectangle(map_with_position, top_left, bottom_right, (0, 255, 0), 3)
        
        return map_with_position


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
        allowed_gamemodes = ['DUOS', 'TRIOS', 'GUN RUN', 'RANKED LEAGUES']
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
            minimap_screenshot = self.capture_screenshot(ROIS['minimap'])

            # Identify desired regions
            current_weapon = self.identify_weapon(weapon_screenshot)
            current_legend = self.identify_legend(legend_screenshot)
            current_gamemode = self.identify_gamemode(gamemode_screenshot)

            # Handle map loading when gamemode changes
            if current_gamemode and current_gamemode.get('map'):
                detected_map = current_gamemode.get('map')
                if detected_map != self.current_map_name:
                    print(f"New map detected: {detected_map}")
                    self.load_current_map(detected_map)
            
            # Track player position on minimap
            # First, find where the minimap section appears on the full map
            minimap_location_on_map = self.find_minimap_location_on_map(minimap_screenshot)
            
            # Then find player position within the minimap (center for rotating minimap)
            player_minimap_pos = self.find_player_position(minimap_screenshot)
            
            # Convert to world coordinates
            world_position = minimap_location_on_map
            
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
            
            # Display player position info
            if player_minimap_pos:
                print(f"Player Position (Minimap): {player_minimap_pos}")
                if world_position:
                    print(f"Player Position (World): {world_position}")
                else:
                    print("Player Position (World): Could not calculate")
            else:
                print("Player Position: Not detected")
            
            print("-" * 40)

            # Debugging (MAY COMMENT OUT)
            #cv2.imshow("Weapon Screenshot", weapon_screenshot)
            #cv2.imshow("Legend Screenshot", legend_screenshot)
            #cv2.imshow("Gamemode Screenshot", gamemode_screenshot)
            #cv2.imshow("Minimap Screenshot", minimap_screenshot)
            
            # Display map with player position if available
            if self.current_map is not None:
                map_with_position = self.draw_player_position(self.current_map, world_position)
                # Resize map for display if it's too large
                display_map = cv2.resize(map_with_position, (800, 800))
                cv2.imshow("Map with Player Position", display_map)
            
            # Terminate the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(2)   # Delay between screenshots
        
        cv2.destroyAllWindows()
        print("Sentinel stopped.")
    
if __name__ == "__main__":
    sentinel = Sentinel()
    sentinel.run()
# Sentinel
A computer vision oriented stat tracker for Apex Legends. 

### Program flow: 
1) Gradually take screenshots of interface every 20 seconds until "Legend Selection" screen appears
2) Record gamemode and map name, take screenshots every 10 seconds until player is in-game
3) Take screenshots every 1 second to record the following information:
    * Player legend (redundant: only need to do once in first snapshot)
    * Teammate legends and names (redundant: only need to do once in first snapshot)
    * Minimap
    * Kill/assist location
    * Death location
    * Held weapon
    * Damage done with held weapon (cumulative sum)
4) When eliminated, record information in database
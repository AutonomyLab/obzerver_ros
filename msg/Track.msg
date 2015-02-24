uint8 STATUS_LOST=0
uint8 STATUS_TRACKING=2

# 0 is reserved for unknown
uint32 uid

uint8 status

sensor_msgs/RegionOfInterest roi
float32 dominant_freq
float32[] spectrum
float32 displacement

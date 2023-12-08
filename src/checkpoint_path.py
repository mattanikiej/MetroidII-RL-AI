# format is (start coordinates): (target coordinates)
checkpoints = {(4,4): (10,6),
               # add pixels from 10,6 -> 10,6
               (10,6): (11,6),
               # add pixels from 11,6 -> 11,6
               (11,6): (12,6),
               # add pixels 12,6 -> 12,6
               (12,6): (1,2), # door
               (1,2): (1,4),
               (1,4): (3,0),
               (3,0): (0,0),
               (0,0): (1,5),
               (1,5): (1,8),
               (1,8): (5,1),
               (5,1): (2,1),
               (2,1): (0,1)
               }
import cv2
import numpy as np

# ============================================================================

INPUT_WIDTH = 160
INPUT_HEIGHT = 120

OUTPUT_TILE_WIDTH = 10
OUTPUT_TILE_HEIGHT = 12

TILE_COUNT = OUTPUT_TILE_WIDTH * OUTPUT_TILE_HEIGHT

# ============================================================================

def stitch_images(input_format, output_filename):
    output_shape = (INPUT_HEIGHT * OUTPUT_TILE_HEIGHT
        , INPUT_WIDTH * OUTPUT_TILE_WIDTH
        , 3)
    output = np.zeros(output_shape, np.uint8)

    for i in range(TILE_COUNT):
        img = cv2.imread(input_format % i)
        cv2.rectangle(img, (0, 0), (INPUT_WIDTH - 1, INPUT_HEIGHT - 1), (0, 0, 255), 1)
        # Draw the frame number
        cv2.putText(img, str(i), (2, 10)
            , cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 255, 255), 1)
        x = i % OUTPUT_TILE_WIDTH * INPUT_WIDTH
        y = i // OUTPUT_TILE_WIDTH * INPUT_HEIGHT
        print(img)
        output[y:y+INPUT_HEIGHT, x:x+INPUT_WIDTH,:] = img

    cv2.imwrite(output_filename, output)

# ============================================================================

stitch_images("images/frame_%04d.png", "stitched_frames.png")
stitch_images("images/mask_%04d.png", "stitched_masks.png")
stitch_images("images/processed_%04d.png", "stitched_processed.png")
# import pandas as pd
# from PIL import Image

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import os

# # Read the CSV file
# # Update with your CSV file path
# csv_file = '/home/ga67jup/tl/TEST/test_annotations.csv'
# df = pd.read_csv(csv_file)

# # Iterate through each row in the CSV
# for index, row in df.iterrows():
#     image_path = row['ImageID']
#     x_min = row['XMin']
#     y_min = row['YMin']
#     x_max = row['XMax']
#     y_max = row['YMax']

#     # Open the image
#     image = Image.open(os.path.join(
#         '/home/ga67jup/tl/TEST/test', str(image_path) + '.jpg'))
#     fig, ax = plt.subplots(1)
#     ax.imshow(image)

#     # Create a Rectangle patch
#     rect = patches.Rectangle((x_min, y_min), x_max - x_min,
#                              y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')

#     # Add the patch to the Axes
#     ax.add_patch(rect)

#     # Display the image with bounding box
#     plt.show()

import pandas as pd
import cv2
import os

# # Read the CSV file
# # Update with your CSV file path
# csv_file = '/home/ga67jup/tl/TEST/test_annotations.csv'
# df = pd.read_csv(csv_file)

# # Iterate through each row in the CSV
# for index, row in df.iterrows():
#     image_path = row['ImageID']
#     x_min = row['XMin']
#     y_min = row['YMin']
#     x_max = row['XMax']
#     y_max = row['YMax']

#     # Open the image
#     image = cv2.imread(os.path.join(
#         '/home/ga67jup/tl/TEST/test', str(image_path) + '.jpg'))
#     height, width, _ = image.shape

#     # Draw the bounding box
#     start_point = (int(x_min*width), int(y_min*height))
#     end_point = (int(x_max*width), int(y_max*height))
#     color = (0, 0, 255)  # Red color in BGR

#     thickness = 1
#     image = cv2.rectangle(image, start_point, end_point, color, thickness)

#     # Display the image with bounding box
#     cv2.imshow('Image with Bounding Box', image)
#     cv2.waitKey(0)  # Wait for a key press to close the window

# # Close all OpenCV windows
# cv2.destroyAllWindows()

# Read the CSV file
# Update with your CSV file path
csv_file = '/home/ga67jup/tl/tl_dataset/test_annotations.csv'
df = pd.read_csv(csv_file)

# Get unique image IDs
unique_image_ids = df['ImageID'].unique()

# Iterate through each unique image ID
for image_id in unique_image_ids:
    # Open the image
    image = cv2.imread(os.path.join(
        '/home/ga67jup/tl/tl_dataset/test', str(image_id) + '.jpg'))
    try:
        height, width, _ = image.shape
    except:
        continue
    # Get all bounding boxes for the current image ID
    image_rows = df[df['ImageID'] == image_id]

    # Draw all bounding boxes on the image
    for index, row in image_rows.iterrows():
        x_min = row['XMin']
        y_min = row['YMin']
        x_max = row['XMax']
        y_max = row['YMax']

        start_point = (int(x_min * width), int(y_min * height))
        end_point = (int(x_max * width), int(y_max * height))
        color = (0, 0, 255)  # Red color in BGR
        thickness = 1
        image = cv2.rectangle(image, start_point, end_point, color, thickness)

    # Display the image with bounding boxes
    cv2.imshow('Image with Bounding Boxes', image)
    cv2.waitKey(0)  # Wait for a key press to close the window

# Close all OpenCV windows
cv2.destroyAllWindows()

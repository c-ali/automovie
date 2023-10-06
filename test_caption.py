import os
from morph import add_caption
import cv2

# Add captions to the images



for part in [25]:
    foldername = f"tmp_part_0{part}"
    print(foldername)
    for dirpath, dnames, fnames in os.walk(foldername):
        for f in fnames:
            if f.endswith(".jpg"):
                filepath = os.path.join(foldername, f)
                print(filepath)
                img = cv2.imread(filepath)
                caption = "This is a default length sentence blabla that goes on and on. It has multiple long sentences. And it goes on for a long time"
                add_caption(img, caption, add_linebreaks=True)
                cv2.imwrite(filepath, img)



import os
import random
import shutil
import subprocess
import logging

# Set up logging
logging.basicConfig(filename='morphing.log', level=logging.INFO)

# Get list of all images in the custom_nir directory
image_dir = 'original_images'
images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

# Make sure there are enough images
if len(images) < 2:
    logging.error('Not enough images in directory.')
    raise SystemExit('Not enough images in directory.')

# Flag for random selection
random_selection = True  # Change this to False to select images with the same image_id
same_patient = False  # If random_selection == False, this applies: Change this to False to select images with the same image_id instead of patient_id
neutral_image = False # if random_selection == False and same_patient == True, this applies: If true, only neutral image will be picked 

# Repeat for n different pairs of images
for i in range(40):
    if random_selection:
        # Select two different random images
        img_fp1, img_fp2 = random.sample(images, 2)
    else:
        if same_patient:
            # Select images with the same patient_id
            patient_ids = list(set([img.split('-')[0] for img in images]))
            selected_patient_id = random.choice(patient_ids)
            selected_images = [img for img in images if img.split('-')[0] == selected_patient_id]
            if len(selected_images) < 2:
                logging.error(f'Not enough images with patient_id {selected_patient_id}.')
                continue
            img_fp1, img_fp2 = random.sample(selected_images, 2)
        else:
            # Select images with the same image_id
            image_ids = list(set([img.split('-')[1] for img in images]))
            selected_image_id = "0" if neutral_image else random.choice(image_ids)
            selected_images = [img for img in images if img.split('-')[1] == selected_image_id]
            if len(selected_images) < 2:
                logging.error(f'Not enough images with image_id {selected_image_id}.')
                continue
            img_fp1, img_fp2 = random.sample(selected_images, 2)

    # Create new directory for this pair
    prefix = "rand" if random_selection else ("same_patient" if same_patient else ("same_image_id-rand" if not neutral_image else "same_image_id-neutral"))
    new_dir = f'morphed_images/{prefix}-{i}'
    os.makedirs(new_dir, exist_ok=True)

    # Copy the images to the new directory
    shutil.copy2(img_fp1, new_dir)
    shutil.copy2(img_fp2, new_dir)

    # Run the shell command on the copied files
    cmd = f'./run_morphing_with_images.sh {os.path.join(new_dir, os.path.basename(img_fp1))} {os.path.join(new_dir, os.path.basename(img_fp2))} 30 2000'
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Log the work
    logging.info(f'Processed pair {i}: {img_fp1}, {img_fp2}')
    logging.info(f'Command output: {process.stdout}')
    if process.stderr:
        logging.error(f'Command error output: {process.stderr}')


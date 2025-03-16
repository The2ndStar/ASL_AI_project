import cv2
import os
import glob

# Define your input and output directories
input_folder = r"C:\Users\Mikee\OneDrive\Documents\ASL_AI_project-main\data\A"
output_folder = r"C:\Users\Mikee\Downloads\DATA_AI\A_grayscale"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get a list of all JPG images in the input folder
image_paths = glob.glob(os.path.join(input_folder, "*.jpg"))

# Process each image
for image_path in image_paths:
    # Read the image
    image = cv2.imread(image_path)
    
    # Check if the image was successfully loaded
    if image is None:
        print(f"Failed to load {image_path}")
        continue

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #crop = gray_image[15:280, 35:270]
    #res = cv2.getRectSubPix(gray_image,(130,280),(150,150))
    #res = cv2.getRectSubPix(img,(250,280),(150,150))
    #res = cv2.getRectSubPix(img,(150,280),(150,150))
    #forEres = cv2.getRectSubPix(img,(170,280),(150,150))
    #Fres = cv2.getRectSubPix(gray_image,(140,280),(150,150))
    #forGres = cv2.getRectSubPix(img,(250,250),(150,150))
    #forHres = cv2.getRectSubPix(img,(150,250),(150,150))
    #Ires = cv2.getRectSubPix(gray_image,(150,280),(150,150))
    #forJres = cv2.getRectSubPix(img,(210,280),(150,150))
    #forKres = cv2.getRectSubPix(img,(150,280),(150,150))
    #forLres = cv2.getRectSubPix(img,(240,280),(150,150))
    #forMres = cv2.getRectSubPix(img,(210,280),(150,150))
    #forNres = cv2.getRectSubPix(img,(210,280),(150,150))
    #forOres = cv2.getRectSubPix(img,(210,280),(150,150))
    #forPres = cv2.getRectSubPix(img,(170,280),(150,150))
    #forQres = cv2.getRectSubPix(img,(190,280),(150,150))
    #res = cv2.getRectSubPix(gray_image,(135,280),(150,150))
    #forSres = cv2.getRectSubPix(img,(200,280),(150,150))
    #forTres = cv2.getRectSubPix(img,(180,280),(150,150))
    #forUres = cv2.getRectSubPix(img,(120,280),(150,150))
    #forVres = cv2.getRectSubPix(img,(150,280),(150,150))
    #forWres = cv2.getRectSubPix(img,(140,280),(150,150))
    #forXres = cv2.getRectSubPix(img,(140,280),(150,150))
    #forYres = cv2.getRectSubPix(img,(260,280),(150,150))
    #res = cv2.getRectSubPix(img,(180,280),(150,150))
    # Construct the output path with the same base filename
    base_name = os.path.basename(image_path)
    output_path = os.path.join(output_folder, base_name)
    
    # Save the grayscale image
    success = cv2.imwrite(output_path, gray_image)
    if success:
        print(f"Saved grayscale image: {output_path}")
    else:
        print(f"Failed to save grayscale image for {image_path}")

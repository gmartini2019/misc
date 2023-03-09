# Libraries for the Model
from skimage.metrics import structural_similarity
import cv2
import numpy as np
# from pdf2image import convert_from_path

# Libraries for the API
from fastapi import FastAPI

# Create the app
app = FastAPI()

# Creathe the endpoint where the file is send it
@app.post("/uploadfile/")






def get_difference(path_image1: str, path_image2: str, path_final_images: str):
    
    """ 
    Function to detect and visualize differences between two images.It is recommended to use images with the same size to get better results.
        
        
        
        Parameters
        ----------
        path_image1: str
            The name with the extension or the path where the first image is located if it is not in the same location as api_image.py. It must be the same size as the second image
        path_image2: str
            The name with the extension or the path where the second image is located if it is not in the same location as api_image.py. It must be the same size as the first image
        path_final_images: str
            the final path where we want to save the results that show the differences between the two images mention before
        
        Returns
        ----------
            five png images with alternative ways to se the differences between the two images mention before
            
            following file formats are supported:
            Windows bitmaps - *.bmp, *.dib (always supported)
            JPEG files - *.jpeg, *.jpg, *.jpe
            JPEG 2000 files - *.jp2 
            Portable Network Graphics - *.png 
            WebP - *.webp 
            Portable image format - *.pbm, *.pgm, *.ppm *.pxm, *.pnm (always supported)
            Sun rasters - *.sr, *.ras (always supported)
            TIFF files - *.tiff, *.tif 
            OpenEXR Image files - *.exr 
            Radiance HDR - *.hdr, *.pic (always supported)
            Raster and Vector geospatial data supported by GDAL (see the Note section)
    """
    if True:# try:
        formats_allowed = ["bmp","dib","jpeg","jpg", "jpe","jp2","png",
                           "webp","pbm", "pgm", "ppm","pxm","pnm","sr",
                           "ras","tiff", "tif","exr","hdr","pic"] #,"pdf"
        
        format_image = [formats.split(".")[-1] for formats in [path_image1, path_image2]]
        format_out = [formats for formats in format_image if formats not in formats_allowed]
        if format_out:
            json_problem = {
                "message" : f"The format: {format_out} is not allow. Please enter a valid format: {formats_allowed}"
            }
            return json_problem
        # if "pdf" in format_image:
        #     pages = convert_from_path('pdf_file', 1)
        #     for page in pages:
        #         page.save('Out.jpg', 'JPEG')
        # read the images
        before = cv2.imread(path_image1)
        after = cv2.imread(path_image2)
        
        # resize the images if necesary
        width_before = int(before.shape[1])
        height_before = int(before.shape[0])
        width_after = int(after.shape[1])
        height_after = int(after.shape[0])
        dim = (width_before, height_before)
        if width_before != width_after or height_before != height_after:
            after = cv2.resize(after, dim, interpolation = cv2.INTER_AREA)
            
        # Convert images to grayscale
        before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

        # Compute SSIM between two images
        (score, diff) = structural_similarity(before_gray, after_gray, full=True)
        maxrange= 1
        minrange = -1
        percent_score= ((score-minrange)/(maxrange-minrange))*100
        print("Image similarity", round(percent_score, 2), "%")

        # The diff image contains the actual image differences between the two images
        # and is represented as a floating point data type in the range [0,1] 
        # so we must convert the array to 8-bit unsigned integers in the range
        # [0,255] before we can use it with OpenCV
        diff = (diff * 255).astype("uint8")

        # Threshold the difference image, followed by finding contours to
        # obtain the regions of the two input images that differ
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        mask = np.zeros(before.shape, dtype='uint8')
        filled_after = after.copy()

        for c in contours:
            area = cv2.contourArea(c)
            if area > 40:
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
                cv2.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)
                cv2.drawContours(mask, [c], 0, (0,255,0), -1)
                cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)

        cv2.imwrite(path_final_images + '/before.png', before)
        cv2.imwrite(path_final_images + '/after.png', after)
        cv2.imwrite(path_final_images + '/diff.png',diff)
        cv2.imwrite(path_final_images + '/mask.png',mask)
        cv2.imwrite(path_final_images + '/filled after.png',filled_after)
        
        return {'status': 'successful',
                'percent_score': percent_score}
    
    # except:
    #     return {'status': 'Fail, review the parameters please'}
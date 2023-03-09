# Libraries for the Model
from skimage.metrics import structural_similarity
import cv2
import numpy as np
import json
from typing import Optional

# Libraries for the API
from fastapi import FastAPI

# Create the app
app = FastAPI()

# Create the endpoint where the file is send it
@app.post("/uploadfile/")
def get_difference(path_image1: str, path_image2: str, path_final_images: str, y_coordinate: Optional[int] = None,
                   x_coordinate: Optional[int] = None, height_of_crop: Optional[int] = None,
                   width_of_crop: Optional[int] = None, percentage_area: float = 20, ):
    
    """ 
    Function to detect and visualize differences between two images.
    It is recommended to use images with the same size to get better results.
        
        Parameters
        ----------
        path_image1: str
            The name with the extension or the path where the first image is located if it is not in the same location as api_image.py. It must be the same size as the second image
        path_image2: str
            The name with the extension or the path where the second image is located if it is not in the same location as api_image.py. It must be the same size as the first image
        path_final_images: str
            the final path where we want to save the results that show the differences between the two images mention before
        y_coordinate
        x_coordinate
        height_of_crop
        width_of_crop
        area_of_difference: int
            the rectangle's area to identify differences

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
    formats_allowed = ["bmp", "dib", "jpeg", "jpg", "jpe", "jp2", "png",
                        "webp", "pbm", "pgm", "ppm", "pxm", "pnm", "sr",
                        "ras", "tiff", "tif", "exr", "hdr", "pic"]
    
    format_image = [formats.split(".")[-1] for formats in [path_image1, path_image2]]
    format_out = [formats for formats in format_image if formats not in formats_allowed]
    if format_out:
        json_problem = {
            "message": f"The format: {format_out} is not allow. Please enter a valid format: {formats_allowed}"
        }
        return json_problem
    # read the images
    before = cv2.imread(path_image1)
    after = cv2.imread(path_image2)
    if y_coordinate:
        if not x_coordinate or not height_of_crop or not width_of_crop:
            failure_message = {"status": "Failure",
                                "message": f"y_coordinate,x_coordinate, height_of_crop and weight_of_crop need to all have values for crop the image."
                                            f"Please complete all the parameters mention before "}
            return json.dumps(failure_message)

        x = x_coordinate
        y = y_coordinate
        h = height_of_crop
        w = width_of_crop
        before = before[y: y+h, x: x+w]
        after = after[y: y+h, x: x+w]
    # resize the images if necessary
    width_before = int(before.shape[1])
    height_before = int(before.shape[0])
    width_after = int(after.shape[1])
    height_after = int(after.shape[0])
    dim = (width_before, height_before)
    total_area = width_before * height_before
    if width_before != width_after or height_before != height_after:
        after = cv2.resize(after, dim, interpolation=cv2.INTER_AREA)
        
    # Convert images to grayscale
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    (score, diff) = structural_similarity(before_gray, after_gray, full=True)
    max_range = 1
    min_range = -1
    percent_score = ((score-min_range)/(max_range-min_range))*100
    dissimilarity_score_percentage = 100 - percent_score
    if dissimilarity_score_percentage == 0.0:
        final_message = {"status": "Success", "message": "The images are equal and do not have any differences"}
        return json.dumps(final_message)
    else:
        print("Image dissimilarity", round(dissimilarity_score_percentage, 2), "%")

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
        # count_contours = len(contours)
        # print("Number of differences:", count_contours)

        mask = np.zeros(before.shape, dtype='uint8')
        filled_after = after.copy()
        
        count_contours1 = 0
        num_differences = []
        coordinates_upper_left = []
        coordinates_centroids = []
        all_areas = []
        all_coordinates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > total_area * percentage_area / 100:
                count_contours1 += 1
                
                x, y, w, h = cv2.boundingRect(c) # x coordinate and y coordinate of the upper left side
                cv2.rectangle(before, (x, y), (x + w, y + h), (36, 255, 12), 2)
                cv2.rectangle(after, (x, y), (x + w, y + h), (36, 255, 12), 2)
                cv2.drawContours(mask, [c], 0, (0, 255, 0), -1)
                cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1)
                # get coordinates of centroid and upper left side of the rectangles
                coordinates_upper_left.append((x, y))
                M = cv2.moments(c)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                coordinates_centroids.append((cx, cy))
                # get the area and coordinates of each rectangle
                all_areas.append(area)
                dict_coordinates = {"top left": [x, y],
                                    "bottom left": [x, y + h],
                                    "bottom right": [x + w, y + h],
                                    "top right": [x + w, y]}
                all_coordinates.append(dict_coordinates)
                num_differences.append(("difference Nº" + str(count_contours1)))
        
        total_area = (round(sum(all_areas), 2))
        percentage = [round(area / total_area * 100, 2) for area in all_areas]
        dict_impact_coordinates = {diff: {"percentage_impact": percentage[index],
                                            "coordinates": all_coordinates[index]} for index, diff in enumerate(num_differences)}

        cv2.imwrite(path_final_images + '/before.png', before)
        cv2.imwrite(path_final_images + '/after.png', after)
        cv2.imwrite(path_final_images + '/diff.png', diff)
        cv2.imwrite(path_final_images + '/mask.png', mask)
        cv2.imwrite(path_final_images + '/filled after.png', filled_after)
    
        return {'status': 'Failed',
                'Image dissimilarity': f"{round(dissimilarity_score_percentage, 2)}%",
                'Number of differences': f"{count_contours1}",
                'Impact and coordinates': dict_impact_coordinates}

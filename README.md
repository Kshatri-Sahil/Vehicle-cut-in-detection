**Explanation:**

    Model Loading: The MobileNet-SSD model and its prototxt file are loaded using cv2.dnn.readNetFromCaffe.

    Distance Calculation: The distance is calculated using the formula distance = (object width * focal length) / apparent width.

    Object Detection: For each frame, the model detects objects, and if the detected object is a car with confidence greater than 50%, it calculates the distance and displays it along with a warning message if the car is too close.

**Note:**

    Adjust the car_width and focal_length to match the actual values for your setup.
    
    Make sure to download the correct versions of the MobileNet-SSD model and prototxt files.
    
    This code uses a webcam (index 0). If you're using a different camera, change the index accordingly.


**To integrate the MobileNet-SSD model into a Jupyter Notebook, you'll need to follow these steps:**

    1. Download the Model Files: You need the deploy.prototxt and the mobilenet_iter_73000.caffemodel files. You can download them using the provided links.

    2. Place the Files in Your Project Directory: Ensure both files are in the same directory as your Jupyter Notebook.

    3. Load and Use the Model in Jupyter Notebook: Write the code to load and use the model for object detection.

**Step-by-Step Guide:**

    1. Download the Model Files from the repository
        deploy.prototxt
        mobilenet_iter_73000.caffemodel
    
        Alternatively, use the following commands in your Jupyter Notebook to download the files:

        import urllib.request
        
        # Download the prototxt file
        prototxt_url = 'https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt'
        urllib.request.urlretrieve(prototxt_url, 'deploy.prototxt')
        
        # Download the caffemodel file
        caffemodel_url = 'https://drive.google.com/uc?id=1fEwSN6gE4ytGsWHyMYJgLb6RFBu8KDk7&export=download'
        urllib.request.urlretrieve(caffemodel_url, 'mobilenet_iter_73000.caffemodel')

    2. Place the Files in Your Project Directory
       Ensure the downloaded files (deploy.prototxt and mobilenet_iter_73000.caffemodel) are in the same directory.


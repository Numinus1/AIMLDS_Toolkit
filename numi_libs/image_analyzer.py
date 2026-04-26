import tkinter as tk
from PIL import Image, ImageTk
import os

import cv2

import pandas as pd
import numpy as np

gColorMapping = {
    "bat": "red",
    "ball": "blue",
    "stumps": "yellow",
    "empty": "lightgray"
    }

gClassificationEncoding = {
    "empty": "empty",
    "bat": "bat",
    "ball": "ball",
    "stumps": "stumps"
}

class ImageAnalyzer:

    
    def __init__(self):

        self.root = tk.Tk()
        self.root.title("Analyzer")
        self.resetInternals()

        display(pd.DataFrame([gColorMapping]))

        self.processed_image_directory_path = "processed_images"
        
        if not os.path.isdir(self.processed_image_directory_path):
            os.makedirs(self.processed_image_directory_path)
        
        
    def resetInternals(self):

        self.current_i = 0
        
        self.current_grid = dict()
        self.current_classifications = dict()

        self.truncatedPath = ""
        self.dirPath = ""
        self.tag = ""

        self.min_width = 800
        self.min_height = 600
        self.aspect_ratio = float(self.min_height)/self.min_width
        print(f"desired aspect ratio: {self.aspect_ratio}")

        self.n_rows = 8
        self.n_columns = 8
        

    def condenseClassifications(self, folder_path):

        ccs = dict()
        
        for root, dirs, files in os.walk(folder_path):
            for filename in files:

                if filename.lower().endswith("_classifications.csv") and not filename.lower().startswith("condensed_"):
                    
                    print("classifications file located: ", os.path.join(root, filename))
                    table = pd.read_csv(os.path.join(root, filename))
                    classDic = self.buildClassificationDictionary(table)

                    ccs = ccs | classDic


        #print("files in c:\n", np.unique(c_table['file']))
        self.saveClassificationsToFile(folder = folder_path, file = 'condensed_classifications.csv', ccs = ccs)
        
                    
    def loadImageFolder(self, image_folder):

        self.resetInternals()
        self.image_folder = image_folder
        self.image_files = self.load_images()
        self.loadClassificationsFromFile()
        

    def load_images(self):
        
        all_files = []

        print(self.image_folder)
        
        for root, dirs, files in os.walk(self.image_folder):
            for filename in files:
                if filename.lower().endswith("jpg") or filename.lower().endswith("webp"):
                    all_files.append(os.path.join(root, filename))
                        
        return all_files

        
    def g_next_image(self):
        self.current_i = self.current_i + 1 if (self.current_i < len(self.image_files) - 1) else 0
        self.displayImagesInFolder()

    
    def g_next_blank_image(self):
        
        i = 0 if (self.current_i + 1) == len(self.image_files) else self.current_i + 1
        blank_found = False
        
        while blank_found == False and i != self.current_i:

            _, truncatedPath = self.splitPaths(self.image_files[i])

            dotIndex = truncatedPath.rfind('.')
            if dotIndex != -1:
                truncatedPath = truncatedPath[:dotIndex]
            
            if truncatedPath not in self.current_classifications.keys():
                blank_found = True
                print("blank found: ", truncatedPath)
            elif len(self.current_classifications[truncatedPath]) == 0:
                blank_found = True
                print("blank found: ", truncatedPath)
            else:
                i = 0 if (i + 1) == len(self.image_files) else i + 1
                

        if i < len(self.image_files):
            self.current_i = i
            self.displayImagesInFolder()

    
    def g_prev_image(self):
        self.current_i = self.current_i - 1 if self.current_i > 0 else len(self.image_files) - 1
        self.displayImagesInFolder()

        
    def print_classifications(self):
        classifications = []
        for filename in self.current_classifications.keys():
            for celli, cl in self.current_classifications[filename].items():
                row, col = celli
                classifications.append({
                        "file": filename,
                        "row_n": row,
                        "col_n": col,
                        "cell_n": (row * self.n_columns) + col,
                        "classification": cl
                    })

        table = pd.DataFrame(classifications)
        display(table)

    
    def saveButtonClicked(self):
        
        self.saveClassificationsToFile(ccs = self.current_classifications, folder = None, file = None)

    def getCellNumber(self, row, col):
        return (row * self.n_columns) + col
    
    def saveClassificationsToFile(self, ccs, folder = None, file = None):
        
        classifications = []
            
        for filename in ccs.keys():
            for celli, cl in ccs[filename].items():
                row, col = celli
                classifications.append({
                        "file": filename,
                        "row_n": row,
                        "col_n": col,
                        "cell_n": self.getCellNumber(row, col),
                        "classification": cl
                    })

        table = pd.DataFrame(classifications)
        
        if folder == None or file == None:
            table.to_csv((self.dirPath + "_classifications.csv"), index = False)
        else:
            table.to_csv(os.path.join(folder, file), index = False)

    
    def loadClassificationsFromFile(self):

        _, imageDirName = self.splitPaths(self.image_folder)
        
        for root, dirs, files in os.walk(os.getcwd()):
            for filepath in files:
 
                if filepath.lower().endswith("_classifications.csv"):

                    _, filename = self.splitPaths(filepath)
                    
                    
                    if filename.lower().startswith("condensed_"):
                        print("condensed classifications file located: ", os.path.join(root, filename))
                        table = pd.read_csv(os.path.join(root, filename))
                        self.current_classifications.clear()
                        self.current_classifications = self.buildClassificationDictionary(table)
                        return

                    if filename.lower().startswith(imageDirName):
                        print("classifications file located: ", os.path.join(root, filename))
                        table = pd.read_csv(os.path.join(root, filename))
                        self.current_classifications.clear()
                        self.current_classifications = self.buildClassificationDictionary(table)
                        
                    
    def buildClassificationDictionary(self, table):
        
        ccs = dict()

        for index, row in table.iterrows():

            _, truncatedPath = self.splitPaths(row['file'])
            lastSlashIndex = truncatedPath.rfind('.')
            if lastSlashIndex != -1:
                truncatedPath = truncatedPath[:lastSlashIndex]
                
            if truncatedPath not in ccs.keys():
                ccs[truncatedPath] = dict()
                
            ccs[truncatedPath][(row['row_n'], row['col_n'])] = row['classification']

        return ccs

        
    def on_grid_click(self, row, col):
    
        #print(f"Clicked grid cell: Row {row}, Column {col}")
        currentC = self.current_grid[(row, col)].cget("bg")
        
        if currentC == gColorMapping["empty"]:
            self.current_grid[(row, col)].config(bg = gColorMapping["bat"])
            self.current_classifications[self.truncatedPath][(row, col)] = "bat"
        
        elif currentC == gColorMapping["bat"]:
            self.current_grid[(row, col)].config(bg = gColorMapping["ball"])
            self.current_classifications[self.truncatedPath][(row, col)] = "ball"
        
        elif currentC == gColorMapping["ball"]:
            self.current_grid[(row, col)].config(bg = gColorMapping["stumps"])
            self.current_classifications[self.truncatedPath][(row, col)] = "stumps"
        
        elif currentC == gColorMapping["stumps"]:
            self.current_grid[(row, col)].config(bg = gColorMapping["empty"])
            self.current_classifications[self.truncatedPath].pop((row, col), None)

        
    def displayImagesInFolder(self, image_folder = None):

        if image_folder != None:
            self.loadImageFolder(image_folder)
        if len(self.image_files) == 0:
            return

        image_selected = False
        
        while image_selected == False and self.current_i < len(self.image_files):

            path = self.image_files[self.current_i]
            tempDirPath, tempTruncatedPath = self.splitPaths(path)
            img = self.processImage(Image.open(path), tempTruncatedPath)

            if img != None:

                dotIndex = tempTruncatedPath.rfind('.')
                if dotIndex != -1:
                    tempTruncatedPath = tempTruncatedPath[:dotIndex]
                
                image_selected = True
                self.dirPath, self.truncatedPath = tempDirPath, tempTruncatedPath
        
                if self.truncatedPath not in self.current_classifications.keys():
                    self.current_classifications[self.truncatedPath] = dict()
            else:
                self.current_i = self.current_i + 1 if (self.current_i < len(self.image_files) - 1) else 0

        numpy_array = np.array(img)
        img_gray = cv2.cvtColor(numpy_array, cv2.COLOR_BGR2GRAY)
        _, img_thresh = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

        self.buildImageGrid(img)
        self.addButtons()
        self.root.mainloop()

    
    def processImage(self, img, imgName = None):

        # image should not be scaled up
        if img.width < self.min_width or img.height < self.min_height:
            print(f"Skipped {path} - too small")
            return None 

        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        # resize and crop image if the aspect ratio isn't 4:3
        
        if (float(img.height) / img.width) != self.aspect_ratio:

            #print(f"processing image {imgName} with ({img.height}, {img.width}) - {img.height / img.width}")

            # compress the image (while maintaining its aspect ratio) until one of the height
            # or width equals 800 or 600
            ch = img.height / self.min_height
            cw = img.width / self.min_width

            f = ch if cw > ch else cw

            img = img.resize((np.round(img.width / f).astype(int), np.round(img.height / f).astype(int)), Image.LANCZOS)

            # crop on whichever axis hasn't dropped to 800/600
            leftCrop = (img.width - self.min_width) / 2
            rightCrop = img.width - leftCrop
            topCrop = (img.height - self.min_height) / 2
            botCrop = img.height - topCrop

            img = img.crop((leftCrop, topCrop, rightCrop, botCrop))
            #print(f"resized and cropped {imgName} to aspect ratio {img.height / img.width}")

        # compress image if aspect raio is 4:3
        else:
            
            img = img.resize((self.min_width, self.min_height), Image.LANCZOS)

        if imgName != None:
            
            processed_filename = self.processed_image_directory_path + '/' + imgName
            
            if not os.path.exists(processed_filename + '.jpg'):
                img.save(processed_filename, format = 'jpeg')

        return img


    def buildImageGrid(self, img):
        self.current_grid = dict()

        imageArray = self.fragmentImage(img)

        for r in range(len(imageArray)):
            for c in range(len(imageArray[r])):

                tkImg = ImageTk.PhotoImage(imageArray[r][c])
                label = tk.Label(self.root, image = tkImg, borderwidth = 3)
                label.image = tkImg
                label.grid(row = r, column = c)
                label.bind("<Button-1>", lambda e, r = r, c = c: self.on_grid_click(r, c))
                
                if (r, c) in self.current_classifications[self.truncatedPath].keys():
                    label.config(bg = gColorMapping[self.current_classifications[self.truncatedPath][(r, c)]])
                else:
                    label.config(bg = gColorMapping["empty"])
                    
                self.current_grid[(r, c)] = label
    
    def fragmentImage(self, img):
        
        imgWidth, imgHeight = img.size
        cellWidth, cellHeight = imgWidth // self.n_columns, imgHeight // self.n_rows

        img_array = []
        
        for r in range(self.n_rows):

            col_imgs = []
            
            for c in range(self.n_columns):

        
                left = c * cellWidth
                right = left + cellWidth
                top = r * cellHeight 
                bot = top + cellHeight
                croppedImg = img.crop((left, top, right, bot))
        
                col_imgs.append(croppedImg)
                #narr = np.array(ImageTk.getimage(tkImg))
                #print(type(ImageTk.getimage(tkImg)))

            img_array.append(col_imgs)

        return img_array

    
    def splitPaths(self, path):

        if not isinstance(path, str):
            path = str(path)
            
        fileName = path
        directory = path
        
        lastSlashIndex = path.rfind('/')
        if lastSlashIndex != -1:
            fileName = path[lastSlashIndex + 1:]
            directory = path[:lastSlashIndex]
        else:
            directory = ""
            
        lastSlashIndex = path.rfind('\\')
        if lastSlashIndex != -1:
            fileName = path[lastSlashIndex + 1:]
            directory = path[:lastSlashIndex]
        else:
            directory = ""

        

        return directory, fileName

    
    def addButtons(self):
        
        next_button = tk.Button(self.root, text="Next", command=self.g_next_image)
        next_button.grid(row = self.n_rows + 1, columnspan = self.n_columns)

        next_blank_button = tk.Button(self.root, text="Next UnAnnotated", command=self.g_next_blank_image)
        next_blank_button.grid(row = self.n_rows + 2, columnspan = self.n_columns)

        prev_button = tk.Button(self.root, text="Previous", command=self.g_prev_image)
        prev_button.grid(row = self.n_rows + 3, columnspan = self.n_columns)

        save_button = tk.Button(self.root, text="Save Classifications", command=self.saveButtonClicked)
        save_button.grid(row = self.n_rows + 4, columnspan = self.n_columns)

        print_button = tk.Button(self.root, text="Print Classifications", command=self.print_classifications)
        print_button.grid(row = self.n_rows + 5, columnspan = self.n_columns)

    def lbp_histogram(self, lbp_image: np.ndarray, bins: int = 256, normalized: bool = True) -> np.ndarray:
        
        """Compute a (optionally normalized) histogram over LBP codes.
    
        Parameters:
        - lbp_image: 2D NumPy array (H x W, dtype uint8) of LBP codes.
        - bins: Number of histogram bins (default 256 to cover codes 0..255).
        - normalized: If True, returns probabilities that sum to 1; otherwise raw counts.
    
        Returns:
        - 1D NumPy array of length `bins` with dtype float64. If normalized, values sum to 1.
        """
        flat = lbp_image.ravel()
        hist = np.bincount(flat, minlength=bins)[:bins].astype(np.float64)
        if normalized and hist.sum() > 0:
            hist /= hist.sum()
        return hist
    
    def lbp_8_1(self, gray: np.ndarray) -> np.ndarray:
        
        """Compute Local Binary Pattern (LBP) codes with P=8, R=1.
    
        Parameters:
        - gray: 2D NumPy array (H x W) representing a grayscale image with dtype uint8.
    
        Returns:
        - NumPy array (H x W, dtype uint8) where each element is the LBP code in [0, 255]
          computed by comparing the 8 neighbors (radius 1) to the center pixel.
        """
        if gray.ndim != 2:
            raise ValueError("Input image must be a 2D grayscale array.")
        gray = gray.astype(np.uint8, copy=False)
    
        pad = np.pad(gray, 1, mode="edge")
        c = pad[1:-1, 1:-1]
    
        n0 = pad[0:-2, 0:-2]  # top-left
        n1 = pad[0:-2, 1:-1]  # top
        n2 = pad[0:-2, 2:  ]  # top-right
        n3 = pad[1:-1, 2:  ]  # right
        n4 = pad[2:  , 2:  ]  # bottom-right
        n5 = pad[2:  , 1:-1]  # bottom
        n6 = pad[2:  , 0:-2]  # bottom-left
        n7 = pad[1:-1, 0:-2]  # left
    
        code = (
            ((n0 >= c).astype(np.uint8) << 7)
            | ((n1 >= c).astype(np.uint8) << 6)
            | ((n2 >= c).astype(np.uint8) << 5)
            | ((n3 >= c).astype(np.uint8) << 4)
            | ((n4 >= c).astype(np.uint8) << 3)
            | ((n5 >= c).astype(np.uint8) << 2)
            | ((n6 >= c).astype(np.uint8) << 1)
            | ((n7 >= c).astype(np.uint8) << 0)
        )
        return code.astype(np.uint8, copy=False)

    def lbp_histogram(self, lbp_image: np.ndarray, bins: int = 256, normalized: bool = True) -> np.ndarray:
            
        """Compute a (optionally normalized) histogram over LBP codes.
    
        Parameters:
        - lbp_image: 2D NumPy array (H x W, dtype uint8) of LBP codes.
        - bins: Number of histogram bins (default 256 to cover codes 0..255).
        - normalized: If True, returns probabilities that sum to 1; otherwise raw counts.
    
        Returns:
        - 1D NumPy array of length `bins` with dtype float64. If normalized, values sum to 1.
        """
        flat = lbp_image.ravel()
        hist = np.bincount(flat, minlength=bins)[:bins].astype(np.float64)
        if normalized and hist.sum() > 0:
            hist /= hist.sum()
        return hist
    
    def extractLBPFeatures(self, img, bins: int = 32):

        if bins <= 0 or 256 % bins != 0:

            print(f"bins' value {bins} needs to be a positive divisor of 256")
            return None

        img_gs_arr = np.asarray(img.convert("L"), dtype=np.uint8)
        img_arr = self.lbp_histogram(self.lbp_8_1(img_gs_arr), 256, True)

        grouped = img_arr.reshape(bins, 256 // bins).sum(axis=1)

        s = grouped.sum()
        if s > 0:
                grouped = grouped / s
            
        return grouped.astype(np.float64, copy=False)

    def extractLBPFeaturesAgainstFolder(self,folder_path, bins: int = 32):

        featureList = []
        if bins <= 0 or 256 % bins != 0:

            print(f"bins' value {bins} needs to be a positive divisor of 256")
            return None
            
        self.resetInternals()
        self.image_folder = folder_path
        self.image_files = self.load_images()
        
        if len(self.image_files) == 0:
            return

        print("continue")

        image_selected = False

        for i in range(len(self.image_files)):

            path = self.image_files[i]
            _, truncatedName = self.splitPaths(path)
            img = self.processImage(Image.open(path))

            if img != None:

                img_arr = self.fragmentImage(img)

                for r in range(len(img_arr)):
                    for c in range(len(img_arr[r])):

                        features = self.extractLBPFeatures(img_arr[r][c])
                        featureDict = dict()

                        featureDict["file"] = truncatedName
                        featureDict["cell_n"] = self.getCellNumber(r, c)
                        
                        for i in range(bins):
                            featureDict[f"lbp_f{i}"] = float(features[i])

                        featureList.append(featureDict)

        table = pd.DataFrame(featureList)
        
        table.to_csv((folder_path + "/lbpfeatures.csv"), index = False)


    def joinClassificationAndFeatures(self, class_csv, features_csv, feature_prefix):

        cMap = dict()
        fMap = dict()

        final = []
        
        c_table = pd.read_csv(class_csv)
        f_table = pd.read_csv(features_csv)
        
        for index, row in c_table.iterrows():

            _, truncatedPath = self.splitPaths(row['file'])
            lastSlashIndex = truncatedPath.rfind('.')
            if lastSlashIndex != -1:
                truncatedPath = truncatedPath[:lastSlashIndex]

            if truncatedPath not in cMap.keys():
                
                cMap[truncatedPath] = dict()
                
            cMap[truncatedPath][row['cell_n']] = row['classification']

        for index, row in f_table.iterrows():

            _, truncatedPath = self.splitPaths(row['file'])
            dotIndex = truncatedPath.rfind('.')
            if dotIndex != -1:
                truncatedPath = truncatedPath[:dotIndex]

            if truncatedPath in cMap.keys():

                item = dict()
                item['file'] = truncatedPath
                item['cell_n'] = row['cell_n']

                if row['cell_n'] not in cMap[truncatedPath].keys():
                    item['classification'] = gClassificationEncoding["empty"]
                else:
                    item['classification'] = gClassificationEncoding[cMap[truncatedPath][row['cell_n']]]

                for col in f_table.columns:
                    if col.startswith(feature_prefix):
                        item[col] = row[col]
                        
                final.append(item)

        return pd.DataFrame(final)

    def generateClassifiedFeatureDataset(self, class_csv, features_csv, feature_prefix):

        df = self.joinClassificationAndFeatures(class_csv, features_csv, feature_prefix)

        print(f"Generated {feature_prefix} classified dataset")
        unique_filenames = np.unique(df['file'])
        print(f"{len(unique_filenames)} unique image names: {unique_filenames}")
        print(f"row count: {df.shape[0]}")
        print(f"column count: {df.shape[1]}")
        
        df.to_csv(feature_prefix + "_classifiedFeatureDataset.csv", index = False)


    def visualizePredictions(path_to_predictions, path_to_image_folder):

        # open predictions file
        p_table = pd.read_csv(path_to_predictions)
                    

            
            
        


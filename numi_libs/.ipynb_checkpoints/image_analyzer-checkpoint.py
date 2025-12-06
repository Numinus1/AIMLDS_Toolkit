import tkinter as tk
from PIL import Image, ImageTk
import os

import pandas as pd
import numpy as np

gColorMapping = {
    "bat": "magenta",
    "ball": "purple",
    "stumps": "cyan",
    "empty": "lightgray"
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

                if filename.lower().endswith("_classifications.csv"):
                    
                    print("classifications file located: ", os.path.join(root, filename))
                    table = pd.read_csv(os.path.join(root, filename))
                    classDic = self.buildClassificationDictionary(table)
                    
                    ccs = ccs | classDic
                    
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
                if filename.lower().endswith("jpg"):
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
                        "cell_n": getCellNumber(row, col),
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

            if row['file'] not in ccs.keys():
                ccs[row['file']] = dict()
                
            ccs[row['file']][(row['row_n'], row['col_n'])] = row['classification']

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
                
                image_selected = True
                self.dirPath, self.truncatedPath = tempDirPath, tempTruncatedPath
        
                if self.truncatedPath not in self.current_classifications.keys():
                    self.current_classifications[self.truncatedPath] = dict()
            else:
                self.current_i = self.current_i + 1 if (self.current_i < len(self.image_files) - 1) else 0
        
        self.fragmentImage(img)
        self.addButtons()
        self.root.mainloop()

    
    def processImage(self, img, imgName):

        # image should not be scaled up
        if img.width < self.min_width or img.height < self.min_height:
            print(f"Skipped {path} - too small")
            return None 

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

        processed_filename = self.processed_image_directory_path + '/' + imgName
        
        if not os.path.exists(processed_filename + '.jpg'):
            img.save(processed_filename, format = 'jpeg')

        return img

    
    def fragmentImage(self, img):

        self.current_grid = dict()
        
        imgWidth, imgHeight = img.size
        cellWidth, cellHeight = imgWidth // self.n_columns, imgHeight // self.n_rows
        
        for r in range(self.n_rows):
            for c in range(self.n_columns):
        
                left = c * cellWidth
                right = left + cellWidth
                top = r * cellHeight 
                bot = top + cellHeight
                croppedImg = img.crop((left, top, right, bot))
        
                tkImg = ImageTk.PhotoImage(croppedImg)
                #narr = np.array(ImageTk.getimage(tkImg))
                #print(type(ImageTk.getimage(tkImg)))
                label = tk.Label(self.root, image = tkImg, borderwidth = 3)
                label.image = tkImg
                label.grid(row = r, column = c)
                label.bind("<Button-1>", lambda e, r = r, c = c: self.on_grid_click(r, c))
                
                if (r, c) in self.current_classifications[self.truncatedPath].keys():
                    label.config(bg = gColorMapping[self.current_classifications[self.truncatedPath][(r, c)]])
                else:
                    label.config(bg = gColorMapping["empty"])
                    
                self.current_grid[(r, c)] = label

    
    def splitPaths(self, path):
        
        fileName = path
        directory = path
        
        lastSlashIndex = path.rfind('/')
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


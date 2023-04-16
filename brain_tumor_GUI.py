import numpy as np
import os
import nibabel as nib   # 의료 및 뇌 영상 파일 형식에 대한 읽기/쓰기를 위한 모듈
import vtk
import matplotlib.pyplot as plt

from scipy import ndimage
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from matplotlib.backends.backend_qt5agg import FigureCanvas as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from IPython.display import display
from fnmatch import fnmatch

image_data = nib.load("/Users/leebyeongju/Desktop/segmentation_GUI/data/BraTS20_Training_001_t1.nii").get_fdata()
mask_data = nib.load("/Users/leebyeongju/Desktop/segmentation_GUI/data/output_0.nii.gz").get_fdata()


class MyApp(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setupUI()
        
    def setupUI(self):
        self.setWindowTitle('Brain Tumor Segmentation')
        self.setGeometry(200, 100, 1000, 800)
        self.outer_layout = QVBoxLayout()
        self.button = QPushButton('Load', self)
        self.outer_layout.addWidget(self.button,alignment=Qt.AlignVCenter)
        self.button.clicked.connect(self.folder_load)
        self.show()
    
    def folder_load(self):
        folder =QFileDialog.getExistingDirectory(self)

        if folder[0]:
            print("파일 선택됨")
            print(folder)
            os.system('python3 ./segmentation/3d_segmentation_demo.py -i ' + folder +' -o ./segmentation/output -m ./segmentation/brain-tumor-segmentation-0002.onnx -nii -ms 0,3,1,2 --full_intensities_range')
            for file in os.listdir(folder):
                if fnmatch(file, '*_t1.nii'):
                    self.brain_image = nib.load(os.path.join(folder,file)).get_fdata()   #뇌이미지
            
            self.mask_image = nib.load('./segmentation/output/output_'+os.path.basename(folder) +'.nii.gz').get_fdata()   #마스크이미지
            
            self.output()
            
        else:
            print("파일 안 골랐음")
        
        
    def output(self):
        
        self.slider_z = QSlider(Qt.Horizontal, self)
        self.slider_z.setRange(0, image_data.shape[2]-1)
        self.slider_z.setSingleStep(1)
        
        self.slider_x = QSlider(Qt.Horizontal, self)
        self.slider_x.setRange(0, image_data.shape[0]-1)
        self.slider_x.setSingleStep(1)
        
        self.slider_y = QSlider(Qt.Horizontal, self)
        self.slider_y.setRange(0, image_data.shape[1]-1)
        self.slider_y.setSingleStep(1)
        
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        
        c1 = FigureCanvas(Figure())
        c2 = FigureCanvas(Figure())
        c3 = FigureCanvas(Figure())
        
        layout = QGridLayout(self.main_widget)
        layout.addWidget(self.slider_z, 0, 0)
        layout.addWidget(self.slider_x, 2, 0)
        layout.addWidget(self.slider_y, 0, 1)
        layout.addWidget(c1, 1, 0)
        layout.addWidget(c2, 3, 0)
        layout.addWidget(c3, 1, 1)
        
        self.slider_z.valueChanged.connect(self.update_z)
        self.slider_x.valueChanged.connect(self.update_x)
        self.slider_y.valueChanged.connect(self.update_y)
        
        self.ax1 = c1.figure.subplots()
        self.ax1.imshow(self.brain_image[:,:,100],cmap='gray')
        self.ax1.imshow(self.mask_image[:,:,100],alpha=0.3)
        self.ax1.axis('off')
        
        self.ax2=c2.figure.subplots()
        self.ax2.imshow(ndimage.rotate(self.brain_image[100,:,:],90),cmap='gray')
        self.ax2.imshow(ndimage.rotate(self.mask_image[100,:,:],90),alpha=0.3)
        self.ax2.axis('off')
        
        self.ax3=c3.figure.subplots()
        self.ax3.imshow(ndimage.rotate(self.brain_image[:,100,:],90),cmap='gray')
        self.ax3.imshow(ndimage.rotate(self.mask_image[:,100,:],90),alpha=0.3)
        self.ax3.axis('off')
        self.outer_layout.addLayout(layout)
      
        self.show()
        
    def update_z(self):
        self.ax1.clear()
        self.ax1.imshow(self.brain_image[:,:,self.slider_z.value()],cmap='gray')
        self.ax1.imshow(self.mask_image[:,:,self.slider_z.value()],alpha=0.3)
        self.ax1.axis('off')
        self.ax1.figure.canvas.draw()
        
    def update_x(self):
        self.ax2.clear()
        self.ax2.imshow(ndimage.rotate(self.brain_image[self.slider_x.value(),:,:],90),cmap='gray')
        self.ax2.imshow(ndimage.rotate(self.mask_image[self.slider_x.value(),:,:],90),alpha=0.3)
        self.ax2.axis('off')
        self.ax2.figure.canvas.draw()
        
    def update_y(self):
        self.ax3.clear()
        self.ax3.imshow(ndimage.rotate(self.brain_image[:,self.slider_y.value(),:],90),cmap='gray')
        self.ax3.imshow(ndimage.rotate(self.mask_image[:,self.slider_y.value(),:],90),alpha=0.3)
        self.ax3.axis('off')
        self.ax3.figure.canvas.draw()
     
if __name__ == '__main__':
  app = QApplication(sys.argv)
  ex = MyApp()
  
  sys.exit(app.exec_())
  
  
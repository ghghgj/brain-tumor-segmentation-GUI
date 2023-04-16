import numpy as np
import os
import nibabel as nib   # 의료 및 뇌 영상 파일 형식에 대한 읽기/쓰기를 위한 모듈
import vtk
import matplotlib.pyplot as plt
from scipy import ndimage
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvas as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from IPython.display import display
from fnmatch import fnmatch
from qtpy import QtWidgets
import pyvista as pv
from pyvistaqt import QtInteractor, MainWindow
from skimage import measure

#mri 파일을 obj 파일로 변환하는 함수
def save_obj(obj,path):
    verts, faces, normals, values = measure.marching_cubes(obj, 0)  #marching cube 알고리즘 사용
    faces = faces + 1
    #obj 파일 만드는 코드
    thefile = open(path, 'w')
    
    for item in verts:
        thefile.write("v {0} {1} {2}\n".format(item[0],item[1],item[2]))

    for item in normals:
        thefile.write("vn {0} {1} {2}\n".format(item[0],item[1],item[2]))

    for item in faces:
        thefile.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(item[0],item[1],item[2]))  

    thefile.close()


class MyApp(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setupUI()
        
    def setupUI(self):
        self.setWindowTitle('Brain Tumor Segmentation')
        self.setGeometry(200, 100, 1000, 800)
        
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        
        self.outer_layout = QGridLayout(self.main_widget)
        self.button = QPushButton('Load', self)
        self.outer_layout.addWidget(self.button,0,0)
        self.button.clicked.connect(self.folder_load)
        
        self.show()
    
    
    def folder_load(self):   #input 폴더를 선택하는 함수
        folder =QFileDialog.getExistingDirectory(self)

        if folder[0]:
            print("파일 선택됨")
            print(folder)
            #segmentation 실행
            os.system('python3 ./segmentation/3d_segmentation_demo.py -i ' + folder +' -o ./segmentation/output -m ./segmentation/brain-tumor-segmentation-0002.onnx -nii -ms 0,3,1,2 --full_intensities_range')
            for file in os.listdir(folder):
                if fnmatch(file, '*_t1.nii'):
                    self.brain_image = nib.load(os.path.join(folder,file)).get_fdata()   #뇌이미지
            
            self.mask_image = nib.load('./segmentation/output/output_'+os.path.basename(folder) +'.nii.gz').get_fdata()   #마스크이미지
            
            #마스크가 2, 1, 3 세가지인데 나눔
            self.t1= np.where(self.mask_image ==1, 1, 0)     
            self.t2= np.where(self.mask_image > 0, 1, 0)
            self.t3= np.where((self.mask_image == 3) | (self.mask_image == 1), 1, 0)
            
            self.t1_path = './segmentation/output/output_'+os.path.basename(folder) +'_t1.obj'
            self.t2_path = './segmentation/output/output_'+os.path.basename(folder) +'_t2.obj'
            self.t3_path = './segmentation/output/output_'+os.path.basename(folder) +'_t3.obj'
            self.brain_path = './segmentation/output/output_'+os.path.basename(folder) +'brain.obj'
        
            #tumor를 obj 파일로 저장
            save_obj(self.t1, self.t1_path)
            save_obj(self.t2, self.t2_path)
            save_obj(self.t3, self.t3_path)
            save_obj(self.brain_image, self.brain_path)
            
            self.output()
            
        else:
            print("파일 안 골랐음")
        
        
    def output(self):     #segmentation 결과를 보여주는 함수
        
        self.slider_z = QSlider(Qt.Horizontal, self)
        self.slider_z.setRange(0, self.brain_image.shape[2]-1)
        self.slider_z.setSingleStep(1)
        self.slider_z.setValue(self.brain_image.shape[2]//2)
        
        self.slider_x = QSlider(Qt.Horizontal, self)
        self.slider_x.setRange(0, self.brain_image.shape[0]-1)
        self.slider_x.setSingleStep(1)
        self.slider_x.setValue(self.brain_image.shape[0]//2)
        
        self.slider_y = QSlider(Qt.Horizontal, self)
        self.slider_y.setRange(0, self.brain_image.shape[1]-1)
        self.slider_y.setSingleStep(1)
        self.slider_y.setValue(self.brain_image.shape[1]//2)
        
        self.opacity = QSlider(Qt.Horizontal, self)
        self.opacity.setRange(1, 100)
        self.opacity.setSingleStep(1)
        self.opacity.setValue(40)
        
        
        c1 = FigureCanvas(Figure().set_facecolor("blue"))
        c2 = FigureCanvas(Figure())
        c3 = FigureCanvas(Figure())
        
        #4사분면 3D viewer
        self.plotter= QtInteractor()
    
        
        layout = QGridLayout()
        layout.addWidget(self.slider_z, 0, 0)
        layout.addWidget(self.slider_x, 2, 0)
        layout.addWidget(self.slider_y, 0, 1)
        layout.addWidget(self.opacity, 2, 1)
        layout.addWidget(c1, 1, 0)
        layout.addWidget(c2, 3, 0)
        layout.addWidget(c3, 1, 1)
        layout.addWidget(self.plotter, 3,1)
        
        reader = pv.get_reader(self.brain_path)
        self.brain_mesh = reader.read()
        self.t1_mesh = pv.read(self.t1_path)
        self.t2_mesh = pv.read(self.t2_path)
        self.t3_mesh = pv.read(self.t3_path)
    
        self.plotter.add_mesh(self.brain_mesh, opacity = self.opacity.value()/100, color='pink')
        self.plotter.add_mesh(self.t1_mesh, opacity=0.5 , color='green')
        self.plotter.add_mesh(self.t2_mesh, opacity=0.5 , color='yellow')
        self.plotter.add_mesh(self.t3_mesh, opacity=0.8 , color='blue')
                                      
        self.slider_z.valueChanged.connect(self.update_z)
        self.slider_x.valueChanged.connect(self.update_x)
        self.slider_y.valueChanged.connect(self.update_y)
        self.opacity.valueChanged.connect(self.update_opacity)
        
        self.ax1 = c1.figure.subplots()
        self.ax1.patch.set_facecolor('#a5e6b6') 
        
        self.ax1.imshow(self.brain_image[:,:,self.slider_z.value()],cmap='gray')
        self.ax1.imshow(self.mask_image[:,:,self.slider_z.value()],alpha=0.3)
        self.ax1.axis('off')
        
        self.ax2=c2.figure.subplots()
        self.ax2.imshow(ndimage.rotate(self.brain_image[self.slider_x.value(),:,:],90),cmap='gray')
        self.ax2.imshow(ndimage.rotate(self.mask_image[self.slider_x.value(),:,:],90),alpha=0.3)
        self.ax2.axis('off')
        
        self.ax3=c3.figure.subplots()
        self.ax3.imshow(ndimage.rotate(self.brain_image[:,self.slider_y.value(),:],90),cmap='gray')
        self.ax3.imshow(ndimage.rotate(self.mask_image[:,self.slider_y.value(),:],90),alpha=0.3)
        self.ax3.axis('off')
        self.outer_layout.addLayout(layout,1,0)
      
        self.show()
        
        
    #축과 brain 투명도를 조절하는 부분
        
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
    
    def update_opacity(self):
        self.plotter.clear()
        self.plotter.add_mesh(self.brain_mesh, opacity = self.opacity.value()/100, color='pink')
        self.plotter.add_mesh(self.t1_mesh, opacity=0.5 , color='green')
        self.plotter.add_mesh(self.t2_mesh, opacity=0.7 , color='yellow')
        self.plotter.add_mesh(self.t3_mesh, opacity=0.8 , color='blue')
        
        
        
if __name__ == '__main__':
  app = QApplication(sys.argv)
  ex = MyApp()
  
  sys.exit(app.exec_())
  
  
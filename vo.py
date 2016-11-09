import numpy as np
import numpy.ma as ma
import cv2 
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import genfromtxt
import ConfigParser
import threading
import sys
import time

print cv2.__version__
print (sys.version)

class showTrajectory(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.px = 0
        self.py = 0
    def setPoint(self,x,y):
        self.px = x
        self.py = y

    def run(self):
        
        plt.axis([-300,300,-300,300])
        plt.ion()
        plt.show()
        theta = 0.6
        ox = 5311.3284067104 
        oy= 1023.6008111322 

        R = np.matrix([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        with open('/home/liu/workspace/Datasets/2010_03_09_drive_0019/insdata.txt')  as f:
            content = f.read().splitlines( )
       
        i = 0 
        for c in content:
            a = c.split()
            pose =np.matrix([[ox+float(a[4])],[oy+float(a[5])]])
            pose = 1.5*R*pose
            plt.scatter(pose.item(0) ,pose.item(1),color='red')
            i=i+1
        plt.draw()

            
        while True:
            plt.scatter(self.px ,self.py)
            plt.draw()
 
class sampleVO:
    def __init__(self):
        self.tmap = showTrajectory()
        inifile = ConfigParser.SafeConfigParser()
        inifile.read('./config.ini')
        #load system setting
        self.dataSets_PATH = inifile.get('system', 'dataSets_PATH')
        #load feature detector parameters
        self.threshold = inifile.getint('FeatureDetection', 'threshold')
        self.minFeatureNum = inifile.getint('FeatureDetection', 'minFeatureNum')
        self.nonmaxSuppression = inifile.getint('FeatureDetection', 'nonmaxSuppression')
        #load camera parameters
        self.focal_x=inifile.getfloat('CameraParameters', 'focal_x')
        self.focal_y=inifile.getfloat('CameraParameters', 'focal_y')
        self.offsetx=inifile.getfloat('CameraParameters', 'offsetx')
        self.offsety=inifile.getfloat('CameraParameters', 'offsety')
        self.focal = (self.focal_x + self.focal_y)/2
        self.t =  np.matrix(np.zeros((3,1)))
        self.R = np.matrix(np.identity(3))
        self.fast = cv2.FastFeatureDetector_create(self.threshold,self.nonmaxSuppression)
        self.lk_params = dict( winSize  = (21,21), maxLevel = 3, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001))

    def run(self):
        self.tmap.start()
        time.sleep(3)
        # Load first image
        str_ptr= self.dataSets_PATH+'I1_%06d.png' % (0)
        img_ptr = cv2.imread(str_ptr,0)
        kp = self.fast.detect(img_ptr,None)
        kp_ptr = np.array([kp[idx].pt for idx in range(len(kp))],np.float32)
        # main loop
        for frame_id in range(1,371):
            str_cur = self.dataSets_PATH+'I1_%06d.png' % frame_id
            img_cur = cv2.imread(str_cur,0)
            img_with_kp = cv2.drawKeypoints(img_ptr, kp, None, None, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow('image',img_with_kp)
            kp_cur, st, err = cv2.calcOpticalFlowPyrLK(img_ptr, img_cur, kp_ptr, None, **self.lk_params)

            kp_ptr = np.delete(kp_ptr, np.where((st==0) | ((kp_cur[:,[1]]<0) | (kp_cur[:,[0]]<0)))[0],  0)   
            kp_cur = np.delete(kp_cur, np.where((st==0) | ((kp_cur[:,[1]]<0) | (kp_cur[:,[0]]<0)))[0]  ,0)     
            E, mask = cv2.findEssentialMat(kp_cur, kp_ptr, self.focal,(self.offsetx,self.offsety),cv2.RANSAC,0.999,1.0)
            res=cv2.recoverPose(E,kp_cur, kp_ptr,self.focal,(self.offsetx,self.offsety))
            dR = np.matrix(res[1])
            dt = np.matrix(res[2])
            si = dt.item(0)+dt.item(1)+dt.item(2)
            if(si<0):
                dt = -dt
            self.t = self.t + self.R*dt
            self.R = dR * self.R
            if (kp_ptr.shape[0]<self.minFeatureNum) & (frame_id != 1):
                kp = self.fast.detect(img_ptr,None)
                kp_ptr = np.array([kp[idx].pt for idx in range(len(kp))],np.float32)
                kp_cur, st, err = cv2.calcOpticalFlowPyrLK(img_ptr, img_cur, kp_ptr, None, **self.lk_params)
                kp_cur = np.delete(kp_cur,np.where((st==0) | ((kp_cur[:,[1]]<0) | (kp_cur[:,[0]]<0)))[0]  ,0) 
            img_ptr = img_cur
            kp_ptr = kp_cur
            self.tmap.setPoint(self.t[0] ,self.t[2])
            cv2.waitKey(30)
if __name__ == '__main__':
    a = sampleVO()
    a.run()
    cv2.waitKey(0)
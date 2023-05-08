import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode = False, maxHands = 2, modelCom = 1, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelCom = modelCom
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands#选择mediapipe的手部模型
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelCom, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils#利用mediapipe提供的函数，画出手上21点
        
        self.handLmsStyle = self.mpDraw.DrawingSpec(color=(0,0,255),thickness=10)
        self.handConStyle = self.mpDraw.DrawingSpec(color=(0,255,0),thickness=3)

    def findHands(self,img,imgRGB,draw = True):
        # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#将opencv2读到的BGR图片转换成RGB给mediapipe用
        self.result = self.hands.process(imgRGB)

        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:#遍历21点坐标列表
                if draw:
                    #参数一画在img上，参数二数据是handLms，参数三21点连起来，参数四点的样式，参数五线的样式
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS,self.handLmsStyle,self.handConStyle)
        return img       

    def findPosition(self, img, handNo = 0,draw = True):
        lmList = []
        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):#将数据对象handLms.landmark组合成一个索引序列，列出数据和数据下标
                imgHeight = img.shape[0]
                imgWidth = img.shape[1]
                xPos = int(lm.x * imgWidth)
                yPos = int(lm.y * imgHeight)         
                lmList.append([id, xPos, yPos])
            if draw:
                 cv2.circle(img, (xPos, yPos), 15, (255, 0, 255), cv2.FILLED)
        return lmList

# def main():
#     pTime = 0
#     cTime = 0
#     cap = cv2.VideoCapture(0)#镜头编号0，自带摄像头。编号1自定义摄像头
#     detector = handDetector()
#     while True:
#         ret,img = cap.read()#cap.read会回传两个数据
#         img = detector.findHands(img, draw = True)
#         lmList = detector.findPosition(img, draw = False)
#         if len(lmList) != 0:
#             print(lmList)
#             #print(lmList[0])只显示第0个点的位置

#         cTime = time.time()
#         fps = 1/(cTime-pTime)
#         pTime = cTime

#         cv2.putText(img, f"FPS:{int(fps)}", (30,50), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0), 3)

#         cv2.imshow("image", img)
#         cv2.waitKey(1)


# if __name__ == "__main__":
#     main()

import cv2
import mediapipe as mp
import time

#读取摄像头
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)#镜头编号0，自带摄像头。编号1自定义摄像头
mpHands = mp.solutions.hands#选择mediapipe的手部模型
hands = mpHands.Hands(False, 2, 1, 0.01, 0.5)
mpDraw = mp.solutions.drawing_utils#利用mediapipe提供的函数，画出手上21点
handLmsStyle = mpDraw.DrawingSpec(color=(0,0,255),thickness=5)
handConStyle = mpDraw.DrawingSpec(color=(0,255,0),thickness=10)
pTime = 0
cTime = 0

while True:
    ret,img = cap.read()#cap.read会回传两个数据
    if ret:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#将opencv2读到的BGR图片转换成RGB给mediapipe用
        result = hands.process(imgRGB)
        #print(result.multi_hand_landmarks)#打印侦测到的手的21点坐标列表
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:#遍历21点坐标列表
                #参数一画在img上，参数二数据是handLms，参数三21点连起来，参数四点的样式，参数五线的样式
                mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS,handLmsStyle,handConStyle)
                for i,lm in enumerate(handLms.landmark):#将数据对象handLms.landmark组合成一个索引序列，列出数据和数据下标
                    xPos = int(lm.x * imgWidth)
                    yPos = int(lm.y * imgHeight)
                    print(i,xPos,yPos)
        
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f"FPS:{int(fps)}", (30,50), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0), 3)
        cv2.imshow('img',img)

    if cv2.waitKey(20) ==ord('q'):
        break
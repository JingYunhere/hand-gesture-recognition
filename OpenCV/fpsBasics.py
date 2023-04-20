import cv2
import fpsBasicsModule


#读取摄像头
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)#镜头编号0，自带摄像头。编号1自定义摄像头
detector = fpsBasicsModule.FPS()
detector.begin()
#设置画面大小
# width = 640
# height = 480
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# #获取画面大小
# imgWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# imgHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

while True:
    ret,img = cap.read()#cap.read会回传两个数据
    for i in range(10000):
        if i == 9999:
            break
    i = 0

    detector.update()
    detector.stop()
    num = float(format(detector.fps()))
    cv2.putText(img, f"Average number:{int(num)}", (30,50), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0), 3)

    cv2.imshow("image", img)
    cv2.waitKey(1)
    # 做一些清理工作
cap.release()#释放内存
cv2.destroyAllWindows()#关闭程序创建的所有窗口
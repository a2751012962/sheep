import cv2
import numpy as np

def main():
    # 创建一个简单的图像
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    image[:] = (255, 255, 255)  # 白色背景
    
    # 在图像上写一些文字
    cv2.putText(image, "OpenCV Test Window", (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # 创建窗口并显示图像
    cv2.namedWindow("Test Window")
    cv2.imshow("Test Window", image)
    
    print("如果你能看到一个白色背景的窗口，说明OpenCV窗口正常工作")
    print("按任意键关闭窗口")
    
    # 等待按键
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
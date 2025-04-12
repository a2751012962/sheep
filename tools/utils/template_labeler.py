import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import shutil
from pathlib import Path
import json
import sys

class TemplateLabelingTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Card Template Labeler")
        
        try:
            # 获取所有类别
            self.train_dir = Path('dataset/train')
            if not self.train_dir.exists():
                raise FileNotFoundError("训练目录不存在")
            
            self.classes = sorted([d.name for d in self.train_dir.iterdir() if d.is_dir()])
            if not self.classes:
                raise ValueError("没有找到任何类别目录")
            
            # 获取所有模板
            self.templates_dir = Path('images/templates')
            if not self.templates_dir.exists():
                raise FileNotFoundError("模板目录不存在")
            
            self.template_files = [f for f in self.templates_dir.glob('template_*.png')]
            if not self.template_files:
                raise ValueError("没有找到任何模板文件")
            
            self.current_template_idx = 0
            
            # 创建界面
            self.create_widgets()
            self.load_current_template()
            
        except Exception as e:
            messagebox.showerror("错误", f"初始化失败: {str(e)}")
            self.root.quit()
    
    def create_widgets(self):
        # 图像显示
        self.image_label = ttk.Label(self.root)
        self.image_label.pack(pady=10)
        
        # 类别选择
        self.class_var = tk.StringVar()
        self.class_combo = ttk.Combobox(self.root, textvariable=self.class_var)
        self.class_combo['values'] = self.classes
        self.class_combo.pack(pady=5)
        
        # 按钮
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=5)
        
        ttk.Button(btn_frame, text="保存并下一个", command=self.save_and_next).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="跳过", command=self.next_template).pack(side=tk.LEFT, padx=5)
        
        # 进度显示
        self.progress_var = tk.StringVar()
        self.update_progress_text()
        ttk.Label(self.root, textvariable=self.progress_var).pack(pady=5)
        
        # 状态显示
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        ttk.Label(self.root, textvariable=self.status_var).pack(pady=5)
    
    def update_progress_text(self):
        self.progress_var.set(f"进度: {self.current_template_idx + 1}/{len(self.template_files)}")
    
    def load_current_template(self):
        try:
            if self.current_template_idx < len(self.template_files):
                # 加载并显示图像
                image = Image.open(self.template_files[self.current_template_idx])
                image = image.resize((200, 200), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                self.image_label.configure(image=photo)
                self.image_label.image = photo
                self.update_progress_text()
                self.status_var.set(f"正在显示: {self.template_files[self.current_template_idx].name}")
        except Exception as e:
            self.status_var.set(f"加载图像失败: {str(e)}")
    
    def save_and_next(self):
        try:
            if not self.class_var.get():
                self.status_var.set("请选择一个类别")
                return
            
            # 将模板复制到选定的类别目录
            src_path = self.template_files[self.current_template_idx]
            dst_dir = self.train_dir / self.class_var.get()
            dst_path = dst_dir / src_path.name
            
            if not dst_dir.exists():
                dst_dir.mkdir(parents=True)
            
            shutil.copy2(src_path, dst_path)
            self.status_var.set(f"已保存到: {dst_path}")
            
            self.next_template()
        except Exception as e:
            self.status_var.set(f"保存失败: {str(e)}")
    
    def next_template(self):
        self.current_template_idx += 1
        if self.current_template_idx < len(self.template_files):
            self.load_current_template()
        else:
            messagebox.showinfo("完成", "所有模板已标注完成！")
            self.root.quit()

def main():
    try:
        root = tk.Tk()
        app = TemplateLabelingTool(root)
        root.mainloop()
    except Exception as e:
        print(f"程序出错: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main() 
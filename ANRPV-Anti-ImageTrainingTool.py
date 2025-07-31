import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
import customtkinter as ctk
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image
from datetime import datetime
import uuid
import traceback
import json

# 設置 CustomTkinter 主題
ctk.set_appearance_mode("System")  # "Light", "Dark", or "System"
ctk.set_default_color_theme("blue")

# 語言字典
LANG = {
    'zh': {
        'title': '圖像保護工具',
        'input_dir': '輸入目錄：',
        'output_dir': '輸出目錄：',
        'select': '選擇',
        'select_files': '選擇檔案：',
        'select_files_btn': '選擇檔案',
        'original_preview': '原圖預覽',
        'process_btn': '開始處理',
        'progress': '處理進度：',
        'log': '日誌：',
        'no_files': '請選擇至少一個檔案！',
        'lang_label': '語言：',
        'log_title': '圖像保護日誌 - {}',
        'log_timestamp': '處理完成時間: {}',
        'log_output': '已生成受保護圖像：{}',
        'log_ssim': 'SSIM: {:.4f}',
        'log_ssim_good': '視覺品質良好（SSIM > 0.975）',
        'log_ssim_warning': '警告：圖像視覺差異可能較大，建議降低 epsilon（例如到 0.005）或隨機雜訊比例',
        'log_class_title': '--- 分類干擾測試 ---',
        'log_resnet_orig': '原圖 ResNet18 分類結果: {}',
        'log_resnet_prot': '受保護圖像 ResNet18 分類結果: {}',
        'log_resnet_good': '保護效果良好：ResNet18 分類結果不同',
        'log_resnet_warning': '警告：ResNet18 分類結果相同，保護效果可能不足',
        'log_vgg_orig': '原圖 VGG16 分類結果: {}',
        'log_vgg_prot': '受保護圖像 VGG16 分類結果: {}',
        'log_vgg_good': '保護效果良好：VGG16 分類結果不同',
        'log_vgg_warning': '警告：VGG16 分類結果相同，保護效果可能不足',
        'log_feature_title': '--- 特徵嵌入距離測試 ---',
        'log_feature_orig': '{} 特徵提取完成',
        'log_feature_prot': '{} 特徵提取完成',
        'log_cosine': '特徵餘弦距離: {:.4f}',
        'log_cosine_good': '保護效果良好：特徵嵌入差異顯著',
        'log_cosine_warning': '警告：特徵嵌入差異較小，保護效果可能不足，建議增加 epsilon 或迭代次數',
        'log_feature_error': '錯誤：無法計算特徵餘弦距離',
        'error_load': '錯誤：無法讀取 {}',
        'error_model': '載入模型失敗：{}',
        'error_preview': '預覽錯誤：{}',
        'error_process': '處理圖像失敗：{}',
        'error_permission': '錯誤：無權限寫入目錄 {}',
        'error_settings': '錯誤：無法保存/載入設定檔案 {}：{}'
    },
    'en': {
        'title': 'Image Protection Tool',
        'input_dir': 'Input Directory:',
        'output_dir': 'Output Directory:',
        'select': 'Select',
        'select_files': 'Select Files:',
        'select_files_btn': 'Select Files',
        'original_preview': 'Original Image Preview',
        'process_btn': 'Start Processing',
        'progress': 'Progress:',
        'log': 'Log:',
        'no_files': 'Please select at least one file!',
        'lang_label': 'Language:',
        'log_title': 'Image Protection Log - {}',
        'log_timestamp': 'Processing Completed Time: {}',
        'log_output': 'Generated Protected Image: {}',
        'log_ssim': 'SSIM: {:.4f}',
        'log_ssim_good': 'Visual Quality Good (SSIM > 0.975)',
        'log_ssim_warning': 'Warning: Visual difference may be noticeable, consider reducing epsilon (e.g., to 0.005) or random noise ratio',
        'log_class_title': '--- Classification Disruption Test ---',
        'log_resnet_orig': 'Original Image ResNet18 Classification: {}',
        'log_resnet_prot': 'Protected Image ResNet18 Classification: {}',
        'log_resnet_good': 'Protection Effective: ResNet18 Classification Different',
        'log_resnet_warning': 'Warning: ResNet18 Classification Same, Protection May Be Insufficient',
        'log_vgg_orig': 'Original Image VGG16 Classification: {}',
        'log_vgg_prot': 'Protected Image VGG16 Classification: {}',
        'log_vgg_good': 'Protection Effective: VGG16 Classification Different',
        'log_vgg_warning': 'Warning: VGG16 Classification Same, Protection May Be Insufficient',
        'log_feature_title': '--- Feature Embedding Distance Test ---',
        'log_feature_orig': '{} Feature Extraction Completed',
        'log_feature_prot': '{} Feature Extraction Completed',
        'log_cosine': 'Feature Cosine Distance: {:.4f}',
        'log_cosine_good': 'Protection Effective: Significant Feature Embedding Difference',
        'log_cosine_warning': 'Warning: Small Feature Embedding Difference, Protection May Be Insufficient, Consider Increasing epsilon or Iterations',
        'log_feature_error': 'Error: Unable to Compute Feature Cosine Distance',
        'error_load': 'Error: Unable to read {}',
        'error_model': 'Failed to load model: {}',
        'error_preview': 'Preview Error: {}',
        'error_process': 'Image Processing Failed: {}',
        'error_permission': 'Error: No permission to write to directory {}',
        'error_settings': 'Error: Unable to save/load settings file {}: {}'
    }
}

# 動態桌面路徑
DESKTOP_DIR = os.path.join(os.path.expanduser("~"), "Desktop")
BASE_DIR = os.path.join(DESKTOP_DIR, 'image_protection_DATA')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'Uploads')
PROTECTED_FOLDER = os.path.join(BASE_DIR, 'protected')
LOG_FOLDER = os.path.join(BASE_DIR, 'logs')
SETTINGS_FILE = os.path.join(BASE_DIR, 'settings.json')

# 確保目錄存在
for folder in [BASE_DIR, UPLOAD_FOLDER, PROTECTED_FOLDER, LOG_FOLDER]:
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except PermissionError:
            ctk.CTkMessageBox(title="Error" if LANG['en']['error_permission'] else "錯誤",
                             message=LANG['en']['error_permission'].format(folder), icon="cancel")
            exit(1)

# GUI 組件（提前聲明，供 load_settings 使用）
input_dir_entry = None
output_dir_entry = None
log_text = None
input_dir_label = None
output_dir_label = None
select_input_btn = None
select_output_btn = None
select_files_label = None
select_files_btn = None
original_label = None
process_btn = None
progress_label = None
log_label = None
lang_label = None

# 載入設定
def load_settings():
    global input_dir, output_dir, current_lang, lang_var
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                current_lang = settings.get('lang', 'en')
                lang_var.set(current_lang)
                input_dir = os.path.normpath(settings.get('input_dir', UPLOAD_FOLDER))
                output_dir = os.path.normpath(settings.get('output_dir', BASE_DIR))
                # 驗證路徑是否有效
                if not os.path.exists(input_dir):
                    input_dir = UPLOAD_FOLDER
                if not os.path.exists(output_dir):
                    output_dir = BASE_DIR
                    for folder in [os.path.join(output_dir, 'protected'), os.path.join(output_dir, 'logs')]:
                        os.makedirs(folder, exist_ok=True)
                # 更新 GUI 輸入框
                if input_dir_entry:
                    input_dir_entry.delete(0, 'end')
                    input_dir_entry.insert(0, input_dir)
                if output_dir_entry:
                    output_dir_entry.delete(0, 'end')
                    output_dir_entry.insert(0, output_dir)
                log_text.insert('end', f"Loaded settings: lang={current_lang}, input_dir={input_dir}, output_dir={output_dir}\n")
                log_text.see('end')
                # 更新語言界面
                update_language()
        else:
            current_lang = 'en'
            lang_var.set('en')
            input_dir = UPLOAD_FOLDER
            output_dir = BASE_DIR
            # 更新 GUI 輸入框
            if input_dir_entry:
                input_dir_entry.delete(0, 'end')
                input_dir_entry.insert(0, input_dir)
            if output_dir_entry:
                output_dir_entry.delete(0, 'end')
                output_dir_entry.insert(0, output_dir)
            log_text.insert('end', f"No settings file found, using defaults: lang={current_lang}, input_dir={input_dir}, output_dir={output_dir}\n")
            log_text.see('end')
            # 更新語言界面
            update_language()
    except Exception as e:
        error_msg = LANG[current_lang]['error_settings'].format(SETTINGS_FILE, str(e) + "\n" + traceback.format_exc())
        ctk.CTkMessageBox(title="Error" if current_lang == 'en' else "錯誤",
                         message=error_msg, icon="cancel")
        log_text.insert('end', error_msg + "\n")
        log_text.see('end')
        current_lang = 'en'
        lang_var.set('en')
        input_dir = UPLOAD_FOLDER
        output_dir = BASE_DIR
        if input_dir_entry:
            input_dir_entry.delete(0, 'end')
            input_dir_entry.insert(0, input_dir)
        if output_dir_entry:
            output_dir_entry.delete(0, 'end')
            output_dir_entry.insert(0, output_dir)
        update_language()

# 保存設定
def save_settings():
    settings = {
        'lang': current_lang,
        'input_dir': os.path.normpath(input_dir),
        'output_dir': os.path.normpath(output_dir)
    }
    try:
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=4)
        log_text.insert('end', f"Saved settings: lang={current_lang}, input_dir={input_dir}, output_dir={output_dir}\n")
        log_text.see('end')
    except Exception as e:
        error_msg = LANG[current_lang]['error_settings'].format(SETTINGS_FILE, str(e) + "\n" + traceback.format_exc())
        ctk.CTkMessageBox(title="Error" if current_lang == 'en' else "錯誤",
                         message=error_msg, icon="cancel")
        log_text.insert('end', error_msg + "\n")
        log_text.see('end')

# 初始化 TkinterDnD
root = TkinterDnD.Tk()
root.title(LANG['en']['title'])

# 載入預訓練模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    resnet_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval().to(device)
    vgg_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).eval().to(device)
except Exception as e:
    ctk.CTkMessageBox(title="Error" if LANG['en']['error_model'] else "錯誤",
                     message=LANG['en']['error_model'].format(e), icon="cancel")
    root.destroy()
    exit(1)

# 當前語言
current_lang = 'en'
lang_var = ctk.StringVar(value='en')

# GUI 組件
input_dir = UPLOAD_FOLDER
output_dir = BASE_DIR
selected_files = []
preview_labels = []

def select_input_dir():
    global input_dir
    new_dir = ctk.filedialog.askdirectory(initialdir=DESKTOP_DIR, title=LANG[current_lang]['input_dir'])
    if new_dir:
        input_dir = os.path.normpath(new_dir)
        input_dir_entry.delete(0, 'end')
        input_dir_entry.insert(0, input_dir)
        save_settings()

def select_output_dir():
    global output_dir
    new_dir = ctk.filedialog.askdirectory(initialdir=DESKTOP_DIR, title=LANG[current_lang]['output_dir'])
    if new_dir:
        output_dir = os.path.normpath(new_dir)
        for folder in [os.path.join(output_dir, 'protected'), os.path.join(output_dir, 'logs')]:
            try:
                os.makedirs(folder, exist_ok=True)
            except PermissionError:
                ctk.CTkMessageBox(title="Error" if current_lang == 'en' else "錯誤",
                                 message=LANG[current_lang]['error_permission'].format(folder), icon="cancel")
                return
        output_dir_entry.delete(0, 'end')
        output_dir_entry.insert(0, output_dir)
        save_settings()

def drop(event):
    global selected_files
    files = root.tk.splitlist(event.data)
    selected_files = [f for f in files if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png']]
    file_list.delete('1.0', 'end')
    for f in selected_files:
        file_list.insert('end', f"{os.path.basename(f)}\n")
    update_preview()

def select_files():
    global selected_files
    files = ctk.filedialog.askopenfiles(
        initialdir=input_dir,
        title=LANG[current_lang]['select_files'],
        filetypes=[("Image Files" if current_lang == 'en' else "圖像檔案", "*.jpg *.jpeg *.png")]
    )
    selected_files = [f.name for f in files]
    file_list.delete('1.0', 'end')
    for f in selected_files:
        file_list.insert('end', f"{os.path.basename(f)}\n")
    update_preview()

def update_preview():
    global preview_labels
    # 清空現有預覽
    for label in preview_labels:
        label.destroy()
    preview_labels = []

    if not selected_files:
        return

    # 動態添加圖片預覽
    for i, file_path in enumerate(selected_files):
        try:
            img = Image.open(file_path)
            # 按比例縮放，最大寬度 280px
            max_width = 280
            aspect_ratio = img.height / img.width
            new_width = min(img.width, max_width)
            new_height = int(new_width * aspect_ratio)
            img = img.resize((new_width, new_height), Image.LANCZOS)
            photo = ctk.CTkImage(light_image=img, dark_image=img, size=(new_width, new_height))
            label = ctk.CTkLabel(preview_frame, text="", image=photo)
            label.grid(row=i, column=0, padx=10, pady=5, sticky='n')
            preview_labels.append(label)
        except Exception as e:
            log_text.insert('end', LANG[current_lang]['error_preview'].format(e) + "\n")
            log_text.see('end')

def process_images():
    if not selected_files:
        ctk.CTkMessageBox(title="Warning" if current_lang == 'en' else "警告",
                         message=LANG[current_lang]['no_files'], icon="warning")
        return

    log_text.delete('1.0', 'end')
    progress_bar.set(0)
    root.update()

    for i, input_path in enumerate(selected_files):
        try:
            ext = os.path.splitext(input_path)[1].lower()
            output_filename = f"protected_{os.path.basename(input_path).split('.')[0]}_{uuid.uuid4().hex[:8]}{ext}"
            output_path = os.path.join(output_dir, 'protected', output_filename)
            log_filename = f"log_{os.path.basename(input_path).split('.')[0]}_{uuid.uuid4().hex[:8]}.log"
            log_path = os.path.join(output_dir, 'logs', log_filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            os.makedirs(os.path.dirname(log_path), exist_ok=True)

            success, log, orig_path, prot_path = protect_image(input_path, output_path, log_path, lang=current_lang)
            log_text.insert('end', log + "\n" + "="*50 + "\n")
            log_text.see('end')
            if not success:
                ctk.CTkMessageBox(title="Error" if current_lang == 'en' else "錯誤",
                                 message=log, icon="cancel")
            progress_bar.set((i + 1) / len(selected_files))
            root.update()
        except Exception as e:
            error_msg = LANG[current_lang]['error_process'].format(str(e) + "\n" + traceback.format_exc())
            log_text.insert('end', error_msg + "\n" + "="*50 + "\n")
            log_text.see('end')
            ctk.CTkMessageBox(title="Error" if current_lang == 'en' else "錯誤",
                             message=error_msg, icon="cancel")

def protect_image(input_path, output_path, log_path, lang='en'):
    try:
        img = cv2.imread(input_path)
        if img is None:
            return False, LANG[lang]['error_load'].format(input_path), None, None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transform = transforms.ToTensor()
        img_tensor = transform(img).unsqueeze(0).requires_grad_(True).to(device)

        # 1. 生成對抗性雜訊 (PGD)
        epsilon = 0.0175
        iterations = 20
        alpha = epsilon / iterations
        adv_img_tensor = img_tensor.clone().detach().requires_grad_(True)
        for _ in range(iterations):
            adv_img_tensor.grad = None
            output = resnet_model(adv_img_tensor)
            loss = output[0, 0]
            loss.backward()
            adv_img_tensor = adv_img_tensor + alpha * torch.sign(adv_img_tensor.grad)
            adv_img_tensor = torch.clamp(adv_img_tensor, img_tensor - epsilon, img_tensor + epsilon)
            adv_img_tensor = torch.clamp(adv_img_tensor, 0, 1)
            adv_img_tensor = adv_img_tensor.detach().requires_grad_(True)

        # 2. 添加隨機化像素值
        random_noise = np.random.normal(0, 0.1, img.shape).astype(np.float32)
        mask = np.random.choice([0, 1], size=img.shape[:2], p=[0.997, 0.003])[:, :, None]
        random_noise = random_noise * mask
        random_noise_tensor = transform(random_noise).unsqueeze(0).to(device)

        # 3. 結合兩種雜訊
        final_img_tensor = adv_img_tensor + 0.02 * random_noise_tensor
        final_img_tensor = torch.clamp(final_img_tensor, 0, 1)

        # 轉回圖像並保存
        final_img = final_img_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255
        final_img = final_img.astype(np.uint8)
        final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, final_img)

        # 檢查視覺品質 (SSIM)
        original_gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        protected_gray = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
        if original_gray is None or protected_gray is None:
            return False, LANG[lang]['error_load'].format(input_path), None, None
        ssim_score = ssim(original_gray, protected_gray, multichannel=False)

        # 4. 分類干擾測試（ResNet18 和 VGG16）
        transform_class = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        classes_resnet = []
        classes_vgg = []
        for path in [input_path, output_path]:
            img = cv2.imread(path)
            if img is None:
                return False, LANG[lang]['error_load'].format(path), None, None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = transform_class(img).unsqueeze(0).to(device)
            with torch.no_grad():
                output_resnet = resnet_model(img_tensor)
                output_vgg = vgg_model(img_tensor)
            classes_resnet.append(output_resnet.argmax(dim=1).item())
            classes_vgg.append(output_vgg.argmax(dim=1).item())

        # 5. 特徵嵌入距離測試
        model_feature = nn.Sequential(*list(resnet_model.children())[:-1]).to(device)
        embeddings = []
        for path in [input_path, output_path]:
            img = cv2.imread(path)
            if img is None:
                return False, LANG[lang]['error_load'].format(path), None, None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = transform_class(img).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model_feature(img_tensor).flatten()
            embeddings.append(embedding.cpu().numpy())

        # 生成日誌
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log = LANG[lang]['log_title'].format(os.path.basename(input_path)) + "\n"
        log += LANG[lang]['log_timestamp'].format(timestamp) + "\n\n"
        log += LANG[lang]['log_output'].format(output_path) + "\n"
        log += LANG[lang]['log_ssim'].format(ssim_score) + "\n"
        if ssim_score < 0.975:
            log += LANG[lang]['log_ssim_warning'] + "\n"
        else:
            log += LANG[lang]['log_ssim_good'] + "\n"
        log += "\n" + LANG[lang]['log_class_title'] + "\n"
        log += LANG[lang]['log_resnet_orig'].format(classes_resnet[0]) + "\n"
        log += LANG[lang]['log_resnet_prot'].format(classes_resnet[1]) + "\n"
        if classes_resnet[0] != classes_resnet[1]:
            log += LANG[lang]['log_resnet_good'] + "\n"
        else:
            log += LANG[lang]['log_resnet_warning'] + "\n"
        log += LANG[lang]['log_vgg_orig'].format(classes_vgg[0]) + "\n"
        log += LANG[lang]['log_vgg_prot'].format(classes_vgg[1]) + "\n"
        if classes_vgg[0] != classes_vgg[1]:
            log += LANG[lang]['log_vgg_good'] + "\n"
        else:
            log += LANG[lang]['log_vgg_warning'] + "\n"
        log += "\n" + LANG[lang]['log_feature_title'] + "\n"
        log += LANG[lang]['log_feature_orig'].format(input_path) + "\n"
        log += LANG[lang]['log_feature_prot'].format(output_path) + "\n"
        if len(embeddings) == 2:
            cosine_distance = 1 - np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
            log += LANG[lang]['log_cosine'].format(cosine_distance) + "\n"
            if cosine_distance > 0.1:
                log += LANG[lang]['log_cosine_good'] + "\n"
            else:
                log += LANG[lang]['log_cosine_warning'] + "\n"
        else:
            log += LANG[lang]['log_feature_error'] + "\n"

        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(log)
        except PermissionError:
            return False, LANG[lang]['error_permission'].format(log_path), None, None

        return True, log, input_path, output_path
    except Exception as e:
        error_msg = LANG[lang]['error_process'].format(str(e) + "\n" + traceback.format_exc())
        return False, error_msg, None, None

def update_language(*args):
    global current_lang
    current_lang = lang_var.get()
    root.title(LANG[current_lang]['title'])
    if input_dir_label:
        input_dir_label.configure(text=LANG[current_lang]['input_dir'])
    if output_dir_label:
        output_dir_label.configure(text=LANG[current_lang]['output_dir'])
    if select_input_btn:
        select_input_btn.configure(text=LANG[current_lang]['select'])
    if select_output_btn:
        select_output_btn.configure(text=LANG[current_lang]['select'])
    if select_files_label:
        select_files_label.configure(text=LANG[current_lang]['select_files'])
    if select_files_btn:
        select_files_btn.configure(text=LANG[current_lang]['select_files_btn'])
    if original_label:
        original_label.configure(text=LANG[current_lang]['original_preview'])
    if process_btn:
        process_btn.configure(text=LANG[current_lang]['process_btn'])
    if progress_label:
        progress_label.configure(text=LANG[current_lang]['progress'])
    if log_label:
        log_label.configure(text=LANG[current_lang]['log'])
    if lang_label:
        lang_label.configure(text=LANG[current_lang]['lang_label'])
    log_text.insert('end', f"Updated language to: {current_lang}\n")
    log_text.see('end')
    save_settings()

# UI 佈局
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=2)
root.grid_rowconfigure(10, weight=1)

# 左方：原圖預覽（可滾動）
original_label = ctk.CTkLabel(root, text=LANG[current_lang]['original_preview'])
original_label.grid(row=0, column=0, padx=10, pady=10, sticky='n')
preview_frame = ctk.CTkScrollableFrame(root, width=320, height=500)
preview_frame.grid(row=1, column=0, rowspan=9, padx=10, pady=10, sticky='nsew')

# 右方：語言設定、路徑、檔案選擇、日誌
lang_label = ctk.CTkLabel(root, text=LANG[current_lang]['lang_label'])
lang_label.grid(row=0, column=1, padx=10, pady=10, sticky='w')
lang_menu = ctk.CTkOptionMenu(root, values=['zh', 'en'], variable=lang_var, command=update_language)
lang_menu.grid(row=0, column=1, padx=10, pady=10, sticky='e')

input_dir_label = ctk.CTkLabel(root, text=LANG[current_lang]['input_dir'])
input_dir_label.grid(row=1, column=1, padx=10, pady=10, sticky='w')
input_dir_entry = ctk.CTkEntry(root, width=300)
input_dir_entry.grid(row=2, column=1, padx=10, pady=5, sticky='w')
input_dir_entry.insert(0, input_dir)
select_input_btn = ctk.CTkButton(root, text=LANG[current_lang]['select'], command=select_input_dir)
select_input_btn.grid(row=2, column=1, padx=10, pady=5, sticky='e')

output_dir_label = ctk.CTkLabel(root, text=LANG[current_lang]['output_dir'])
output_dir_label.grid(row=3, column=1, padx=10, pady=10, sticky='w')
output_dir_entry = ctk.CTkEntry(root, width=300)
output_dir_entry.grid(row=4, column=1, padx=10, pady=5, sticky='w')
output_dir_entry.insert(0, output_dir)
select_output_btn = ctk.CTkButton(root, text=LANG[current_lang]['select'], command=select_output_dir)
select_output_btn.grid(row=4, column=1, padx=10, pady=5, sticky='e')

select_files_label = ctk.CTkLabel(root, text=LANG[current_lang]['select_files'])
select_files_label.grid(row=5, column=1, padx=10, pady=10, sticky='w')
file_list = ctk.CTkTextbox(root, width=350, height=100)
file_list.grid(row=6, column=1, padx=10, pady=10, sticky='w')
select_files_btn = ctk.CTkButton(root, text=LANG[current_lang]['select_files_btn'], command=select_files)
select_files_btn.grid(row=7, column=1, padx=10, pady=10, sticky='w')

log_label = ctk.CTkLabel(root, text=LANG[current_lang]['log'])
log_label.grid(row=8, column=1, padx=10, pady=10, sticky='w')
log_text = ctk.CTkTextbox(root, width=350, height=150)
log_text.grid(row=9, column=1, padx=10, pady=10, sticky='nsew')

# 進度條：橫跨左右區域，底部
progress_label = ctk.CTkLabel(root, text=LANG[current_lang]['progress'])
progress_label.grid(row=10, column=0, padx=10, pady=10, sticky='sw')
progress_bar = ctk.CTkProgressBar(root, width=800)
progress_bar.grid(row=10, column=0, columnspan=2, padx=10, pady=10, sticky='sew')
progress_bar.set(0)

# 全視窗拖放
root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', drop)

# 開始按鈕
process_btn = ctk.CTkButton(root, text=LANG[current_lang]['process_btn'], command=process_images)
process_btn.grid(row=7, column=1, padx=10, pady=10, sticky='e')

# 載入設定（在 GUI 組件初始化後）
load_settings()

root.mainloop()
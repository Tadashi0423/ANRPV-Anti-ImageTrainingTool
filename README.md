# ANRPV Anti-ImageTrainingTool

## 專案簡介 / Project Overview
**中文**：ANRPV Anti-ImageTrainingTool 是一個圖像保護工具，旨在防止圖像被未經授權的 AI 模型用於訓練。透過結合**對抗性雜訊 (Adversarial Noise)**（使用投影梯度下降，PGD，生成）和 **隨機化像素值 (Randomized Pixel Values)**，本工具在保持圖像視覺品質的同時，干擾 AI 的訓練過程。工具提供直觀的圖形化界面（GUI），支援中英文切換、路徑記憶、圖片預覽、拖放功能、進度條及日誌記錄。

**English**：ANRPV Anti-ImageTrainingTool is an image protection tool designed to prevent unauthorized AI models from using images for training. By combining **Adversarial Noise** (generated using Projected Gradient Descent, PGD) and **Randomized Pixel Values**, it disrupts AI training while preserving visual quality. The tool offers an intuitive GUI, supporting Chinese/English switching, path memory, image preview, drag-and-drop, progress bar, and logging.

圖片為ANRPV界面 ANRPV User Interface:
![ANRPV GUI Screenshot](https://github.com/Tadashi0423/ANRPV-Anti-ImageTrainingTool/blob/main/gui_screenshot.png?raw=true)


## 功能 / Features
- **語言設定 / Language Settings**：支援英文 (en) 和中文 (zh)，重啟後記憶語言選擇 / Supports English (en) and Chinese (zh), remembers language selection after restart.
- **路徑記憶 / Path Memory**：記錄輸入和輸出資料夾 / Remembers input and output directories.
- **圖片預覽 / Image Preview**：最大寬度 280px，支援滾動 / Maximum width 280px, supports scrolling.
- **拖放支援 / Drag-and-Drop Support**：接受 .jpg/.jpeg/.png 檔案 / Accepts .jpg/.jpeg/.png files.
- **進度條 / Progress Bar**：顯示圖像處理進度 / Displays image processing progress.
- **日誌記錄 / Logging**：記錄 SSIM、分類干擾、特徵嵌入距離等 / Logs SSIM, classification interference, feature embedding distance, etc.

## 用戶手冊 / User Manual
**中文**：
1. **推薦方式**：從 GitHub Releases 下載 `ANRPV_Anti_ImageTrainingTool.exe`（見「可執行檔案」），無需安裝 Python 環境。
   - 運行：
     ```cmd
     ANRPV_Anti_ImageTrainingTool.exe
     ```
2. **次要方式**（進階用戶）：
   - 下載並解壓縮項目檔案，或使用：
     ```bash
     git clone https://github.com/Tadashi0423/ANRPV-Anti-ImageTrainingTool.git
     ```
   - 安裝依賴（見「環境要求」）。
   - 運行：
     ```bash
     python ANRPV_Anti_ImageTrainingTool.py
     ```
3. 在 GUI 中選擇輸入和輸出資料夾。
4. 拖放圖像檔案（.jpg/.jpeg/.png）或點擊「選擇檔案」按鈕。
5. 查看處理進度條，完成後檢查輸出資料夾中的受保護圖像（`output_dir/protected`）。
6. 查看日誌（`data/logs/log_*.log`）以了解 SSIM 等指標。

**English**:
1. **Recommended Method**: Download `ANRPV_Anti_ImageTrainingTool.exe` from GitHub Releases (see "Executable"), no Python installation required.
   - Run:
     ```cmd
     ANRPV_Anti_ImageTrainingTool.exe
     ```
2. **Secondary Method** (Advanced Users):
   - Download and extract project files, or use:
     ```bash
     git clone https://github.com/Tadashi0423/ANRPV-Anti-ImageTrainingTool.git
     ```
   - Install dependencies (see "Environment Requirements").
   - Run:
     ```bash
     python ANRPV_Anti_ImageTrainingTool.py
     ```
3. Select input and output directories in the GUI.
4. Drag and drop image files (.jpg/.jpeg/.png) or click the "Select Files" button.
5. Monitor the progress bar; check protected images in `output_dir/protected` upon completion.
6. Review logs (`data/logs/log_*.log`) for metrics like SSIM.

## 使用須知 / Usage Notes
- **運行環境 / Environment**：
  - Python 3.13
  - PyTorch 2.7.1+cpu
  - torchvision 0.22.1+cpu
  - customtkinter 5.2.2
  - opencv-python 4.12.0.88
  - numpy 2.1.2
  - scikit-image 0.25.2
  - scipy 1.16.1
  - pillow 11.0.0
  - tk 0.1.0
  - tkinterdnd2 0.4.3
  - **安裝命令 / Installation Commands**（僅適用於運行 `.py` 檔案）：
    ```bash
    pip install torch==2.7.1+cpu torchvision==0.22.1+cpu --index-url https://download.pytorch.org/whl/cpu
    pip install opencv-python==4.12.0.88 numpy==2.1.2 scikit-image==0.25.2 scipy==1.16.1 pillow==11.0.0 tk==0.1.0 tkinterdnd2==0.4.3 customtkinter
    ```

- **CPU 用量注意 / CPU Usage Notes**：
  - 本工具針對 CPU 優化，無需 GPU，適合無 GPU 的使用者。同時，若系統有 GPU，程式會自動檢測並利用 GPU (CUDA) 來加速運算，特別是在圖像處理和對抗性雜訊生成等計算密集型任務中。
  - 處理大圖像或多圖像時，CPU 使用率可能較高，建議關閉其他高負載程式。
  - Tested on CPU-only systems and RTX 5070 (CPU mode, with optional GPU acceleration).

- **生成檔案或資料夾 / Generated Files or Folders**：
  - `data/settings.json`：儲存語言和路徑設定。
  - `data/logs/log_*.log`：記錄處理日誌（SSIM、時間等）。
  - `output_dir/protected/`：儲存受保護圖像。
  - **注意 / Note**：這些檔案不應上傳至 GitHub（已在 `.gitignore` 中排除）。

- **CPU/無 GPU 使用者友善 / CPU/No-GPU User-Friendly**：
  - 使用 PyTorch CPU 版本，無需 GPU 硬體。
  - GUI 設計簡單，支援拖放，降低操作門檻。
  - 日誌和進度條提供清晰的處理反饋。

## 開發目的 / Development Purpose
**中文**：本工具旨在保護個人圖像隱私，防止未經授權的 AI 模型（如圖像識別或生成模型）使用圖像進行訓練。透過對抗性雜訊和隨機化像素值，確保圖像在視覺上無明顯變化，但對 AI 訓練無效，保護用戶數據安全。

**English**：This tool aims to protect personal image privacy by preventing unauthorized AI models (e.g., image recognition or generative models) from using images for training. By applying adversarial noise and randomized pixel values, it ensures images remain visually intact but ineffective for AI training, safeguarding user data.

## 原理解釋 / How It Works
**中文**：
- **對抗性雜訊 (Adversarial Noise)**：使用投影梯度下降（PGD）生成精心設計的微小擾動，通過迭代優化使圖像在 VGG16 等模型的特徵空間中偏離原始嵌入，干擾 AI 模型的特徵提取，使圖像難以被用於訓練，但對人眼幾乎無影響。
- **隨機化像素值 (Randomized Pixel Values)**：隨機調整部分像素值，進一步增加 AI 訓練的難度，降低模型準確性。
- **SSIM 評估**：使用結構相似性 (Structural Similarity Index) 確保處理後圖像與原始圖像視覺上高度相似。SSIM 值範圍為 0 到 1，值越接近 1 表示視覺保真度越高（>0.97 為佳），代表處理後的圖像與原始圖像在人眼看來幾乎無差異。
- **VGG16 整合**：利用預訓練的 VGG16 模型計算特徵嵌入距離，確保對抗性效果。
- **分類干擾測試**：通過比較原始圖像和處理後圖像在 AI 分類模型（如 VGG16）上的輸出結果，若結果數字（例如類別機率或標籤）顯著不同，則表示成功誤導 AI 分類演算法，達到保護效果。

**English**：
- **Adversarial Noise**: Uses Projected Gradient Descent (PGD) to generate carefully crafted subtle perturbations, iteratively optimizing to shift the image’s feature embeddings in models like VGG16, disrupting AI model feature extraction while remaining nearly imperceptible to the human eye.
- **Randomized Pixel Values**: Randomly adjusts pixel values to further increase the difficulty of AI training, reducing model accuracy.
- **SSIM Evaluation**: Uses Structural Similarity Index to ensure processed images remain visually similar to originals. SSIM ranges from 0 to 1, with values closer to 1 indicating higher visual fidelity (>0.97 is ideal), meaning the processed image is nearly indistinguishable to the human eye.
- **VGG16 Integration**: Leverages a pre-trained VGG16 model to compute feature embedding distances, ensuring adversarial effectiveness.
- **Classification Interference Test**: Compares the output of original and processed images on AI classification models (e.g., VGG16). If the results (e.g., class probabilities or labels) differ significantly, it indicates successful misleading of the AI classification algorithm, achieving the protection goal.

## 與 Fawkes 的比較 / Comparison with Fawkes
**中文**：
- **相似之處**：ANRPV 和 Fawkes 均旨在保護圖像免於未經授權的 AI 訓練，使用對抗性技術（如雜訊）干擾模型。
- **不同之處**：
  - **技術方法**：ANRPV 使用投影梯度下降（PGD）結合對抗性雜訊和隨機化像素值，Fawkes 主要使用「隱形斗篷」(cloaking) 技術改變圖像特徵。
  - **使用**：ANRPV 設計上並無特定針對對象，適用於通用圖像保護；Fawkes 主要針對臉部圖像進行干擾。
  - **硬體需求**：ANRPV 針對 CPU 優化，無需 GPU；Fawkes 在 GPU 上效率更高。

**English**：
- **Similarities**: Both ANRPV and Fawkes aim to protect images from unauthorized AI training, using adversarial techniques (e.g., noise) to disrupt models.
- **Differences**:
  - **Technical Approach**: ANRPV uses Projected Gradient Descent (PGD) combined with adversarial noise and randomized pixel values; Fawkes primarily uses "cloaking" to alter image features.
  - **Use**: ANRPV is designed for general image protection without specific targets; Fawkes primarily targets facial images.
  - **Hardware Requirements**: ANRPV is optimized for CPU, requiring no GPU; Fawkes performs better with GPU.


## 使用的技術 / Technologies Used
- **Python 3.13**：核心程式語言 / Core programming language.
- **PyTorch 2.7.1+cpu**：用於對抗性雜訊生成和 VGG16 模型 / For adversarial noise generation and VGG16 model.
- **torchvision 0.22.1+cpu**：提供預訓練 VGG16 模型 / Provides pre-trained VGG16 model.
- **customtkinter 5.2.2**：現代化 GUI 框架 / Modern GUI framework.
- **tkinterdnd2 0.4.3**：支援拖放功能 / Supports drag-and-drop functionality.
- **opencv-python 4.12.0.88**：圖像處理 / Image processing.
- **numpy 2.1.2**：數值計算 / Numerical computations.
- **scikit-image 0.25.2**：計算 SSIM / Computes SSIM.
- **scipy 1.16.1**：科學計算 / Scientific computations.
- **pillow 11.0.0**：圖像處理 / Image processing.
- **tk 0.1.0**：基礎 GUI 支援 / Basic GUI support.

## 程式碼架構 / Code Structure
**中文**：
- **`main.py`**：主程式，初始化 GUI 和處理邏輯。
- **GUI 組件**：
  - 語言選擇下拉選單 / Language dropdown.
  - 輸入/輸出路徑選擇按鈕 / Input/output path selection buttons.
  - 圖片預覽區域（280px）/ Image preview area (280px).
  - 拖放區域 / Drag-and-drop area.
  - 進度條 / Progress bar.
- **圖像處理模組**：
  - 對抗性雜訊生成（使用 PGD，PyTorch）/ Adversarial noise generation (using PGD, PyTorch).
  - 隨機化像素值 / Randomized pixel values.
  - VGG16 特徵提取 / VGG16 feature extraction.
  - SSIM 計算 / SSIM calculation.
- **資料管理**：
  - `settings.json`：儲存語言和路徑 / Stores language and paths.
  - `data/logs/`：儲存日誌 / Stores logs.
- **PyInstaller Hook**（`hook-scipy.py`）：確保打包時包含 scipy 模組。

**English**：
- **`main.py`**: Main program, initializes GUI and processing logic.
- **GUI Components**:
  - Language selection dropdown.
  - Input/output path selection buttons.
  - Image preview area (280px).
  - Drag-and-drop area.
  - Progress bar.
- **Image Processing Module**:
  - Adversarial noise generation (using PGD, PyTorch).
  - Randomized pixel values.
  - VGG16 feature extraction.
  - SSIM calculation.
- **Data Management**:
  - `settings.json`: Stores language and paths.
  - `data/logs/`: Stores logs.
- **PyInstaller Hook** (`hook-scipy.py`): Ensures scipy modules are included during packaging.

## 可執行檔案 / Executable
- 下載 [ANRPV_Anti_ImageTrainingTool.exe](https://github.com/Tadashi0423/ANRPV-Anti-ImageTrainingTool/releases/latest)（從 GitHub Releases 取得最新版本）。

## 許可證 / License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

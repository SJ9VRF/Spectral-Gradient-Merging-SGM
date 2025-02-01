# 🚀 Spectral Gradient Merging (SGM) Fine-Tuning

![Screenshot_2025-01-31_at_7 22 49_AM-removebg-preview](https://github.com/user-attachments/assets/f9aa33fa-b99a-40c3-9e4c-79aeecde7e1c)


**SGM** combines **Signal-to-Noise Ratio (SNR) analysis** with **gradient merging** to optimize fine-tuning. This technique reduces redundant updates by grouping layers with similar SNR values for **shared parameter updates**, making it more memory efficient.

## **📌 Why SGM?**
✅ **SNR-Based Layer Selection** – Identifies high-SNR layers for fine-tuning.  
✅ **Spectral Gradient Merging** – Merges **similar layers** for shared updates.  
✅ **Efficient Fine-Tuning** – Reduces memory usage while maintaining performance.  


## **🚀 Installation**
1️⃣ **Clone the Repository**
```bash
git clone https://github.com/your-username/sgm-finetune.git
cd sgm-finetune
```


Set Up Virtual Environment & Install Dependencies
``` bash
python -m venv env
source env/bin/activate  # On Mac/Linux
env\Scripts\activate  # On Windows

pip install -r requirements.txt
```

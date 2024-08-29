import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io



model = load_model('Dog_Cat_64-64_512Dense.h5') 

def predict_image(img):
    if img is None:
        return None
    
    
    img = Image.fromarray(img.astype('uint8'), 'RGB') 
    img = img.resize((64, 64)) # Modelin giriş boyutu
    img_array = img_to_array(img) 
    img_array = np.expand_dims(img_array, axis=0) 
    
    
    prediction = model.predict(img_array)[0][0] # Tahmin
 
    tahmin = "Köpek" if prediction > 0.5 else "Kedi" 
    dogruluk = prediction if prediction > 0.5 else 1 - prediction
    
    
    sonuc = f"""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h2 style="font-size: 28px; margin-bottom: 15px;">Tahmin Sonucu</h2>
        <p style="font-size: 24px; margin-bottom: 10px;">Bu resim büyük olasılıkla bir <b>{tahmin}</b>!</p>
        <p style="font-size: 18px;">Tahmin Güveni: {dogruluk*100:.2f}%</p>
        <div style="background-color: rgba(255,255,255,0.3); width: 80%; height: 20px; margin: 15px auto; border-radius: 10px;">
            <div style="background-color: #4CAF50; width: {dogruluk*100}%; height: 100%; border-radius: 10px; transition: width 0.5s ease-in-out;"></div>
        </div>
    </div>
    """
    
    return sonuc



descr = """
Bu uygulama, yapay zeka teknolojisini kullanarak kedi ve köpek resimlerini sınıflandırır. 
Yükleyeceğiniz resmin kedi mi yoksa köpek mi olduğunu tahmin eder.

<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&display=swap" rel="stylesheet">

<div style="display: flex; justify-content: space-between; font-family: 'Arial', sans-serif;">
    <div style="flex: 1; padding-right: 10px;">
        <h2>Nasıl Kullanılır?</h2>
        <ol>
            <li>Sağ taraftaki alana bir resim yükleyin veya sürükleyip bırakın.</li>
            <li>Resim yüklendikten sonra, model otomatik olarak tahminde bulunacaktır.</li>
            <li>Sonuç ve doğruluk oranı gözükecektir.</li>
        </ol>
    </div>
    <div style="flex: 1; padding-left: 10px;">
        <h2>Not</h2>
        <ul>
            <li>Desteklenen resim formatları: JPG, JPEG, PNG, WEPB</li>
            <li>Model her zaman %100 doğru olmayabilir. Eğlenmek ve öğrenmek için kullanın.</li>
            <li>Resimlerin net ve anlaşılır olması doğruluk oranını artıracaktır.</li>
        </ul>
    </div>
</div>

<p style="font-size: 20px; font-family: 'Poppins', sans-comic; font-style: bold;">
    Daha fazla açıklama yok bir kedi veya köpek resmi yükleyin ve sonucu görün!
</p>
"""
footer = """
<div style="text-align: center; font-size: 14px; color: #666;">
    <p>
        <a href="https://github.com/yusuffenes" style="margin-right: 15px; text-decoration: none;font-style: bold; color: #666;">GitHub</a>
        <a href="https://www.linkedin.com/in/yusufenesbudak" style="margin-right: 15px; text-decoration: none;font-style: bold; color: #666;">LinkedIn</a>
        <span>yusufenes</span>
    </p>
    <p style="margin-top: 10px;">
        <span>🐾 Kedi-Köpek Tahmin Uygulaması 🐾</span>
        <span style="margin-left: 15px;">© 2024 ©</span>
    </p>
</div>

"""

myGradioAppInterface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy", label="Resim Yükle"),
    outputs=gr.HTML(label="Tahmin Sonucu"),
    title="🐾 Kedi/Köpek Tahmin Uygulaması 🐾",
    
    description=descr,
    article=footer,
    examples=[
        ["resimler/ornek_kedi.jpg"],
        ["resimler/ornek_kopek.jpg"],
        ["resimler/ornek_kedi2.jpg"],
        ["resimler/ornek_kopek2.jpg"]
        
    ],
    theme=gr.themes.Soft().set(
        body_background_fill="#f0f2f6",
        button_primary_background_fill="#4a90e2",
        button_shadow="#205493",
    ),
    allow_flagging=False,
    
)


myGradioAppInterface.launch()
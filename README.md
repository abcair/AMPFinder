# AMPFinder
 
AMPFinder software instruction  
AMPFinder can be downloaded at https://github.com/abcair/AMPFinder .  
AMPFinder aims to distinguish AMPs and non-AMPs and predict their function types by a cascaded computational model.  
Requirements  
Windows / Linux / Mac os  
	Anaconda/miniconda  
Installation  
conda create -n AMPFinder  
conda activate AMPFinder  
conda install python=3.8  
pip install pyside2  
pip install tensorflow==2.8.0  
pip install transformers  
pip install datasets     
# (windows)  
pip3 install torch torchvision torchaudio   
# (Linux)  
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu   
# (Mac os)  
pip3 install torch torchvision torchaudio   
pip install biopython  
pip install scikit-learn  
pip install protlearn  
pip install propy3  
pip install sentencepiece  
pip install blosum  
pip install joblib  
  
Usage  
The prediction page is shown as follow:  
conda activate AMPFinder  
python main_window.py  
Run the python file for AMP identification  
  

Run the python file for AMP function identification  
 
Contact us  
Any questions about AMPFinder, please email to ystillen@gmail.com  

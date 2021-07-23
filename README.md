# Signboard-Transliteration-From-Vernacular-Languages

## Prerequisites
	1. Anaconda
		(For installation guide refer this link:- @https://docs.anaconda.com/anaconda/install/linux/)

	2. Tensorflow(before installation create a conda environment for which you can refer the above link)
		(For installation guide refer this link:- @https://www.tensorflow.org/install/pip)

	3. easyocr
		can be easily installed by:- pip install easyocr
		
##  Running the system:
	Step 1:
		Create a conda environment by:
			-> conda activate (environment-name)

	Step 2:
		Clone the repository into your system by:
			->git clone @https://github.com/anasmohammad611/Signboard-Transliteration-From-Vernacular-Languages.git

	Step 3:
		Now train the model which will then save your model into training_checkpoints folder
			->Open the Transliteration1.ipynb file in jupyter notebook
			->Upload both the datasets(xml file present in dataset folder) in the jupyter notebook of Transliteration1.ipynb
			->Run the file
			->Now model will take around 4-5hrs to train, depending on your system's performance it might take more or less time.
			->After file is executed, the training_checkpoints folder is created where trained model is stored.

	Step 4:
		Now run the project.py file
			->It will open up a tkinter window after few seconds.
			->Choose an image to transliterate
			->Then press on translate option to translate the text from english to hindi

	Step5:
		Deactivate the conda environment by:
			->conda deactivate

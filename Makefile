# GITHUB

commit:
	git commit -am "commit from make file"

push:
	git push origin master

pull:
	git pull origin master

fetch:
	git fetch origin master

reset:
	rm -f .git/index
	git reset

req:
	pip freeze > requirements.txt

compush: req commit push


# CONSOL RUN


run_no_debug:
	python main.py --no-debug

run:
	python main.py


# MODEL TUNING

tuning:
	python scripts/model_tuning.py


# predict.py fonksiyonunu kullanarak train seti değerleri tahmini ve AUC degeri
predict:
	python scripts/predict.py

# predict.py fonksiyonunu kullanarak test seti değerlerini tahmin etme
predict_test:
	python scripts/predict.py --test

# predict.py fonksiyonu ile tahmin edilen sonuçların kaggle'a gönderilmesi
kaggle_submit_predict:
	kaggle competitions submit -c home-credit-default-risk -f outputs/predictions/sub_from_prediction_py.csv -m "Message"

muhat:
	python models/dsmlbc2/muhat.py
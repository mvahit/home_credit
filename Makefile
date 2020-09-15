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

compush: commit push


# CONSOL RUN


run_no_debug:
	python lightgbm-with-simple-features.py --no-debug

run:
	python lightgbm-with-simple-features.py


# MODEL TUNING

tuning:
	python model_tuning/model_tuning.py


# predict.py fonksiyonunu kullanarak train seti değerleri tahmini ve AUC degeri
predict:
	python scripts/predict.py

# predict.py fonksiyonunu kullanarak test seti değerlerini tahmin etme
predict_test:
	python scripts/predict.py --test

# predict.py fonksiyonu ile tahmin edilen sonuçların kaggle'a gönderilmesi
kaggle_submit_predict:
	kaggle competitions submit -c home-credit-default-risk -f /predictions/sub_from_prediction_py.csv -m "Message"


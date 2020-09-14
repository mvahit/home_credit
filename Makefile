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
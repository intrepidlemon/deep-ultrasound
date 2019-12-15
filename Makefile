setup:
	sudo docker run -d \
		--name liver-ultrasound-setup \
	  --mount type=bind,source=$(DATA_DIR),target=/data \
		--gpus all \
		liver-ultrasound \
		bash setup.sh

sort:
	sudo docker run --rm \
		--name liver-ultrasound-sort \
	  --mount type=bind,source=$(DATA_DIR),target=/data \
		--gpus all \
		liver-ultrasound \
		python3.6 data.py --prefix free

train:
	sudo docker run --rm \
		--name liver-ultrasound-train \
	  --mount type=bind,source=$(DATA_DIR),target=/data \
		--gpus all \
		liver-ultrasound \
		python3.6 run.py --model v2 --trials 100 --description $(description)

notebook:
		sudo docker run --rm \
		--name liver-ultrasound-notebook\
	  --mount type=bind,source=$(DATA_DIR),target=/data \
		--gpus all \
		-p 8888:8888 \
		liver-ultrasound \
		jupyter notebook --allow-root --ip=0.0.0.0
